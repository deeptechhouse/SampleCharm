"""
Librosa musical analyzer for the Audio Sample Analysis Application.

Analyzes musical properties: pitch, key, and tonality.
"""

from typing import List, Optional, Tuple

import librosa
import numpy as np

from src.core.analyzer_base import BaseAnalyzer
from src.core.models import AudioSample, MusicalAnalysis


# Musical note names
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Key profiles (Krumhansl-Schmuckler)
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


class LibrosaMusicalAnalyzer(BaseAnalyzer[MusicalAnalysis]):
    """
    Librosa-based musical analysis.

    Analyzes:
    - Pitch detection using piptrack
    - Key estimation using chroma correlation
    - Tonality assessment
    """

    def __init__(self):
        """Initialize musical analyzer."""
        super().__init__("librosa_musical", "1.0.0")

    def _analyze_impl(self, audio: AudioSample) -> MusicalAnalysis:
        """
        Analyze musical properties of audio.

        Args:
            audio: AudioSample to analyze

        Returns:
            MusicalAnalysis: Musical analysis result
        """
        # Step 1: Detect pitch (uses pre-computed features)
        pitch_result = self._detect_pitch(audio)

        # Step 2: Estimate key
        key_result = self._estimate_key(audio)

        # Step 3: Assess tonality
        tonality_result = self._assess_tonality(audio)

        return MusicalAnalysis(
            has_pitch=pitch_result['has_pitch'],
            fundamental_frequency=pitch_result['fundamental_frequency'],
            note_name=pitch_result['note_name'],
            pitch_stability=pitch_result['pitch_stability'],
            estimated_key=key_result['key'],
            key_confidence=key_result['confidence'],
            is_ambiguous=key_result['is_ambiguous'],
            alternative_keys=key_result['alternatives'],
            is_atonal=tonality_result['is_atonal'],
            tonality_score=tonality_result['tonality_score'],
            explanation=None
        )

    def _detect_pitch(self, audio: AudioSample) -> dict:
        """
        Detect pitch using librosa piptrack.

        Args:
            audio: AudioSample with pre-computed features

        Returns:
            dict: Pitch detection results
        """
        # Reuse pre-computed harmonic content from features (avoid redundant HPSS)
        harmonic = audio.features.harmonic
        sr = audio.sample_rate

        # Pitch tracking
        pitches, magnitudes = librosa.piptrack(y=harmonic, sr=sr)

        # Get dominant pitch for each frame
        pitch_values = []
        for i in range(pitches.shape[1]):
            index = magnitudes[:, i].argmax()
            pitch = pitches[index, i]
            if pitch > 0:
                pitch_values.append(float(pitch))

        if not pitch_values:
            return {
                'has_pitch': False,
                'fundamental_frequency': None,
                'note_name': None,
                'pitch_stability': 0.0
            }

        # Compute statistics
        mean_pitch = np.mean(pitch_values)
        std_pitch = np.std(pitch_values)

        # Pitch stability: 1.0 if no variance, lower with more variance
        # Using coefficient of variation
        if mean_pitch > 0:
            cv = std_pitch / mean_pitch
            stability = max(0.0, 1.0 - cv)
        else:
            stability = 0.0

        # Convert Hz to note name
        note_name = self._hz_to_note(mean_pitch)

        return {
            'has_pitch': True,
            'fundamental_frequency': float(mean_pitch),
            'note_name': note_name,
            'pitch_stability': float(min(1.0, stability))
        }

    def _hz_to_note(self, frequency: float) -> str:
        """
        Convert frequency in Hz to musical note name.

        Args:
            frequency: Frequency in Hz

        Returns:
            str: Note name (e.g., "A4", "C#5")
        """
        if frequency <= 0:
            return "Unknown"

        # A4 = 440 Hz
        a4 = 440.0

        # Semitones from A4
        semitones = 12 * np.log2(frequency / a4)

        # Round to nearest semitone
        semitones_rounded = round(semitones)

        # Note index (A = 9 in our NOTE_NAMES)
        note_index = (9 + semitones_rounded) % 12

        # Octave (A4 is octave 4)
        octave = 4 + (9 + semitones_rounded) // 12

        return f"{NOTE_NAMES[note_index]}{octave}"

    def _estimate_key(self, audio: AudioSample) -> dict:
        """
        Estimate musical key using chroma correlation.

        Uses Krumhansl-Schmuckler key-finding algorithm.

        Args:
            audio: AudioSample to analyze

        Returns:
            dict: Key estimation results
        """
        # Get chroma features
        chroma = audio.features.chroma

        # Average chroma over time
        chroma_mean = np.mean(chroma, axis=1)

        # Normalize
        if np.sum(chroma_mean) > 0:
            chroma_mean = chroma_mean / np.sum(chroma_mean)

        # Correlate with key profiles for all keys
        correlations: List[Tuple[str, float]] = []

        for i, note in enumerate(NOTE_NAMES):
            # Rotate profile to match key
            major_rotated = np.roll(MAJOR_PROFILE, i)
            minor_rotated = np.roll(MINOR_PROFILE, i)

            # Normalize profiles
            major_rotated = major_rotated / np.sum(major_rotated)
            minor_rotated = minor_rotated / np.sum(minor_rotated)

            # Compute correlation
            major_corr = np.corrcoef(chroma_mean, major_rotated)[0, 1]
            minor_corr = np.corrcoef(chroma_mean, minor_rotated)[0, 1]

            correlations.append((f"{note} major", float(major_corr)))
            correlations.append((f"{note} minor", float(minor_corr)))

        # Sort by correlation
        correlations.sort(key=lambda x: x[1], reverse=True)

        # Best key
        best_key, best_corr = correlations[0]

        # Normalize confidence to [0, 1]
        # Correlation is in [-1, 1], map to [0, 1]
        confidence = (best_corr + 1) / 2

        # Check if ambiguous (close second)
        second_key, second_corr = correlations[1]
        is_ambiguous = (best_corr - second_corr) < 0.05

        # Top alternatives
        alternatives: List[Tuple[str, float]] = [
            (key, (corr + 1) / 2)  # Normalize to [0, 1]
            for key, corr in correlations[1:4]
        ]

        return {
            'key': best_key,
            'confidence': float(confidence),
            'is_ambiguous': is_ambiguous,
            'alternatives': alternatives
        }

    def _assess_tonality(self, audio: AudioSample) -> dict:
        """
        Assess how tonal vs atonal the audio is.

        Args:
            audio: AudioSample to analyze

        Returns:
            dict: Tonality assessment
        """
        # Get chroma features
        chroma = audio.features.chroma

        # Compute chroma entropy over time
        # High entropy = more atonal
        entropies = []
        for frame in range(chroma.shape[1]):
            frame_data = chroma[:, frame]
            if np.sum(frame_data) > 0:
                # Normalize to probability distribution
                p = frame_data / np.sum(frame_data)
                # Compute entropy
                entropy = -np.sum(p * np.log2(p + 1e-10))
                entropies.append(entropy)

        if not entropies:
            return {
                'is_atonal': True,
                'tonality_score': 0.0
            }

        mean_entropy = np.mean(entropies)

        # Maximum entropy for 12 pitch classes
        max_entropy = np.log2(12)

        # Tonality score: 1.0 if no entropy, 0.0 if max entropy
        tonality_score = 1.0 - (mean_entropy / max_entropy)

        # Atonal if tonality score is low
        is_atonal = tonality_score < 0.3

        return {
            'is_atonal': is_atonal,
            'tonality_score': float(max(0.0, min(1.0, tonality_score)))
        }
