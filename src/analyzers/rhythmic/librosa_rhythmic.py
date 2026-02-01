"""
Librosa rhythmic analyzer for the Audio Sample Analysis Application.

Analyzes rhythmic properties: tempo, beats, and one-shot detection.
"""

from typing import List, Optional

import librosa
import numpy as np

from src.core.analyzer_base import BaseAnalyzer
from src.core.models import AudioSample, RhythmicAnalysis


class LibrosaRhythmicAnalyzer(BaseAnalyzer[RhythmicAnalysis]):
    """
    Librosa-based rhythmic analysis.

    Analyzes:
    - One-shot vs loop detection
    - Tempo estimation
    - Beat tracking
    """

    def __init__(self, tempo_range: tuple = (60, 240)):
        """
        Initialize rhythmic analyzer.

        Args:
            tempo_range: Expected tempo range (min_bpm, max_bpm)
        """
        super().__init__("librosa_rhythmic", "1.0.0")
        self.tempo_range = tempo_range

    def _analyze_impl(self, audio: AudioSample) -> RhythmicAnalysis:
        """
        Analyze rhythmic properties of audio.

        Args:
            audio: AudioSample to analyze

        Returns:
            RhythmicAnalysis: Rhythmic analysis result
        """
        # Step 1: Detect if one-shot (uses cached features)
        is_one_shot = self._detect_one_shot(audio)

        # Step 2: Detect rhythm and tempo
        if is_one_shot:
            # One-shots don't have meaningful rhythm
            return RhythmicAnalysis(
                is_one_shot=True,
                has_rhythm=False,
                tempo_bpm=None,
                num_beats=None,
                rhythm_confidence=0.9,  # High confidence in one-shot detection
                beat_times=None,
                explanation=None
            )

        # Step 3: Estimate tempo and track beats (uses cached onset envelope)
        rhythm_result = self._analyze_rhythm(audio)

        return RhythmicAnalysis(
            is_one_shot=False,
            has_rhythm=rhythm_result['has_rhythm'],
            tempo_bpm=rhythm_result['tempo'],
            num_beats=rhythm_result['num_beats'],
            rhythm_confidence=rhythm_result['confidence'],
            beat_times=rhythm_result['beat_times'],
            explanation=None
        )

    def _detect_one_shot(self, audio: AudioSample) -> bool:
        """
        Detect if audio is a one-shot (single sound event).

        One-shots typically have:
        - Short duration
        - Single onset
        - Quick decay

        Args:
            audio: AudioSample to analyze

        Returns:
            bool: True if one-shot, False if loop/longer audio
        """
        # Short duration is strong indicator
        if audio.duration < 0.5:
            return True

        # Check onset count
        onset_frames = audio.features.onset_frames
        if len(onset_frames) <= 1:
            return True

        # Check if onsets are clustered at the start
        if len(onset_frames) > 0:
            onset_times = librosa.frames_to_time(
                onset_frames,
                sr=audio.sample_rate
            )
            # If most onsets are in first 20% of audio, it's likely one-shot
            early_onsets = sum(1 for t in onset_times if t < audio.duration * 0.2)
            if early_onsets / len(onset_times) > 0.8:
                return True

        # Check envelope shape
        envelope = np.abs(audio.mono_audio)
        peak_idx = np.argmax(envelope)
        peak_position = peak_idx / len(envelope)

        # One-shots typically peak early and decay
        if peak_position < 0.3:
            # Check if it decays to near silence
            end_portion = envelope[int(len(envelope) * 0.8):]
            if np.mean(end_portion) < 0.1 * envelope[peak_idx]:
                return True

        return False

    def _analyze_rhythm(self, audio: AudioSample) -> dict:
        """
        Analyze rhythm and tempo.

        Args:
            audio: AudioSample with pre-computed features

        Returns:
            dict: Rhythm analysis results
        """
        sr = audio.sample_rate
        duration = audio.duration

        # Use cached onset envelope (avoids redundant computation)
        onset_env = audio.features.onset_envelope

        # Estimate tempo
        tempo, beats = librosa.beat.beat_track(
            onset_envelope=onset_env,
            sr=sr,
            units='time'
        )

        # Convert tempo to float if it's an array
        if isinstance(tempo, np.ndarray):
            tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
        else:
            tempo = float(tempo)

        # Validate tempo is in reasonable range
        if tempo < self.tempo_range[0]:
            tempo = tempo * 2  # Double if too slow
        elif tempo > self.tempo_range[1]:
            tempo = tempo / 2  # Halve if too fast

        # Convert beats to list
        beat_times: List[float] = beats.tolist() if isinstance(beats, np.ndarray) else []

        # Calculate number of beats
        num_beats = len(beat_times) if beat_times else 0

        # Also estimate using duration if beats are sparse
        estimated_beats = (duration * tempo) / 60.0

        # Use estimated if detected beats are much lower
        if num_beats < estimated_beats * 0.5:
            num_beats = estimated_beats

        # Determine if audio has rhythm
        # Needs multiple beats to be rhythmic
        has_rhythm = num_beats >= 2

        # Confidence based on beat consistency
        confidence = self._calculate_rhythm_confidence(beat_times, tempo)

        return {
            'has_rhythm': has_rhythm,
            'tempo': float(tempo),
            'num_beats': float(num_beats),
            'beat_times': beat_times if beat_times else None,
            'confidence': confidence
        }

    def _calculate_rhythm_confidence(
        self,
        beat_times: List[float],
        tempo: float
    ) -> float:
        """
        Calculate confidence in rhythm detection.

        High confidence when beats are evenly spaced.

        Args:
            beat_times: Detected beat times
            tempo: Estimated tempo

        Returns:
            float: Confidence score [0.0, 1.0]
        """
        if len(beat_times) < 2:
            return 0.5  # Low confidence with few beats

        # Calculate inter-beat intervals
        intervals = np.diff(beat_times)

        if len(intervals) == 0:
            return 0.5

        # Expected interval based on tempo
        expected_interval = 60.0 / tempo

        # Calculate how consistent intervals are
        interval_std = np.std(intervals)
        interval_mean = np.mean(intervals)

        if interval_mean == 0:
            return 0.5

        # Coefficient of variation
        cv = interval_std / interval_mean

        # Also check how close to expected
        deviation = abs(interval_mean - expected_interval) / expected_interval

        # Confidence: 1.0 if perfect, lower with more variation
        consistency_score = max(0, 1.0 - cv)
        accuracy_score = max(0, 1.0 - deviation)

        # Combine scores
        confidence = 0.6 * consistency_score + 0.4 * accuracy_score

        return float(min(1.0, max(0.0, confidence)))
