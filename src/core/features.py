"""
Feature extractor for the Audio Sample Analysis Application.

Extracts audio features using librosa for use by all analyzers.
"""

from datetime import datetime
from typing import TYPE_CHECKING

import librosa
import numpy as np

from src.core.models import FeatureCache

if TYPE_CHECKING:
    from src.core.models import AudioSample


class FeatureExtractor:
    """
    Stateless feature extraction using librosa.

    All methods are static - no instance state needed.
    """

    @staticmethod
    def extract(audio: "AudioSample") -> FeatureCache:
        """
        Extract all features from audio sample.

        Args:
            audio: AudioSample to extract features from

        Returns:
            FeatureCache: All extracted features

        Time: ~100-500ms for 30s audio
        """
        mono = audio.mono_audio
        sr = audio.sample_rate

        # Spectral features
        mfcc = librosa.feature.mfcc(y=mono, sr=sr, n_mfcc=13)
        spectral_centroid = librosa.feature.spectral_centroid(y=mono, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=mono, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=mono, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(mono)

        # HPSS separation
        harmonic, percussive = librosa.effects.hpss(mono)

        # Pitch features
        chroma = librosa.feature.chroma_cqt(y=harmonic, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)

        # Onset detection
        onset_env = librosa.onset.onset_strength(y=mono, sr=sr)
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr
        )

        # Tempogram
        tempogram = librosa.feature.tempogram(
            onset_envelope=onset_env,
            sr=sr
        )

        return FeatureCache(
            mfcc=mfcc,
            spectral_centroid=spectral_centroid,
            spectral_rolloff=spectral_rolloff,
            spectral_bandwidth=spectral_bandwidth,
            zero_crossing_rate=zcr,
            harmonic=harmonic,
            percussive=percussive,
            chroma=chroma,
            tonnetz=tonnetz,
            onset_envelope=onset_env,
            onset_frames=onset_frames,
            tempogram=tempogram,
            extracted_at=datetime.utcnow()
        )

    @staticmethod
    def extract_mfcc(
        audio_data: np.ndarray,
        sr: int = 22050,
        n_mfcc: int = 13
    ) -> np.ndarray:
        """
        Extract only MFCC features.

        Args:
            audio_data: Audio samples (mono)
            sr: Sample rate
            n_mfcc: Number of MFCCs

        Returns:
            np.ndarray: MFCC features
        """
        return librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc)

    @staticmethod
    def extract_spectral(
        audio_data: np.ndarray,
        sr: int = 22050
    ) -> dict:
        """
        Extract spectral features only.

        Args:
            audio_data: Audio samples (mono)
            sr: Sample rate

        Returns:
            dict: Spectral features
        """
        return {
            'centroid': librosa.feature.spectral_centroid(y=audio_data, sr=sr),
            'rolloff': librosa.feature.spectral_rolloff(y=audio_data, sr=sr),
            'bandwidth': librosa.feature.spectral_bandwidth(y=audio_data, sr=sr),
            'zcr': librosa.feature.zero_crossing_rate(audio_data)
        }

    @staticmethod
    def extract_rhythm(
        audio_data: np.ndarray,
        sr: int = 22050
    ) -> dict:
        """
        Extract rhythm features only.

        Args:
            audio_data: Audio samples (mono)
            sr: Sample rate

        Returns:
            dict: Rhythm features including tempo and beats
        """
        onset_env = librosa.onset.onset_strength(y=audio_data, sr=sr)
        tempo, beats = librosa.beat.beat_track(
            onset_envelope=onset_env,
            sr=sr
        )
        beat_times = librosa.frames_to_time(beats, sr=sr)

        return {
            'tempo': float(tempo),
            'beats': beats,
            'beat_times': beat_times.tolist(),
            'onset_envelope': onset_env
        }

    @staticmethod
    def extract_pitch(
        audio_data: np.ndarray,
        sr: int = 22050
    ) -> dict:
        """
        Extract pitch-related features.

        Args:
            audio_data: Audio samples (mono)
            sr: Sample rate

        Returns:
            dict: Pitch features
        """
        # HPSS to get harmonic content
        harmonic, _ = librosa.effects.hpss(audio_data)

        # Chroma features
        chroma = librosa.feature.chroma_cqt(y=harmonic, sr=sr)

        # Pitch tracking
        pitches, magnitudes = librosa.piptrack(y=harmonic, sr=sr)

        # Get dominant pitch for each frame
        pitch_values = []
        for i in range(pitches.shape[1]):
            index = magnitudes[:, i].argmax()
            pitch = pitches[index, i]
            if pitch > 0:
                pitch_values.append(float(pitch))

        return {
            'chroma': chroma,
            'pitches': pitch_values,
            'mean_pitch': np.mean(pitch_values) if pitch_values else None
        }
