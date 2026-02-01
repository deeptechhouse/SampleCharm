"""
Core data models for the Audio Sample Analysis Application.

Immutable domain models representing audio samples and analysis results.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.core.features import FeatureExtractor

# Thread locks for lazy-loaded properties (shared across all AudioSample instances)
_features_lock = threading.Lock()
_audio_16k_lock = threading.Lock()


@dataclass(frozen=True)
class FeatureCache:
    """
    Cached audio features extracted from AudioSample.

    All features are computed once and cached for reuse across analyzers.
    """

    # Spectral features
    mfcc: np.ndarray  # Shape: (n_mfcc, n_frames)
    spectral_centroid: np.ndarray  # Shape: (1, n_frames)
    spectral_rolloff: np.ndarray  # Shape: (1, n_frames)
    spectral_bandwidth: np.ndarray  # Shape: (1, n_frames)
    zero_crossing_rate: np.ndarray  # Shape: (1, n_frames)

    # Harmonic/Percussive separation
    harmonic: np.ndarray  # Shape: (samples,)
    percussive: np.ndarray  # Shape: (samples,)

    # Pitch features
    chroma: np.ndarray  # Shape: (12, n_frames)
    tonnetz: np.ndarray  # Shape: (6, n_frames)

    # Onset detection
    onset_envelope: np.ndarray  # Shape: (n_frames,)
    onset_frames: np.ndarray  # Shape: (n_onsets,)

    # Tempo/rhythm features
    tempogram: np.ndarray  # Shape: (n_tempo, n_frames)

    # Timestamp
    extracted_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True)
class AudioSample:
    """
    Immutable representation of a loaded audio file.

    Stores audio data and metadata with lazy-loaded features.
    """

    # File identification
    file_path: Path
    file_hash: str  # SHA-256 of original file

    # Processed audio properties
    sample_rate: int  # Always 22050 Hz after resampling
    channels: int  # 1 (mono) or 2 (stereo)
    duration: float  # Duration in seconds
    audio_data: np.ndarray  # Shape depends on channels

    # Original quality metadata
    original_sample_rate: int
    original_bit_depth: str  # e.g., 'PCM_16', 'PCM_24'
    original_format: str  # 'WAV', 'AIFF', 'MP3', 'FLAC'

    # Lazy-loaded features
    _features: Optional[FeatureCache] = field(
        default=None, repr=False, compare=False, hash=False
    )

    # Cached 16kHz mono audio for YAMNet and Whisper (avoids redundant resampling)
    _audio_16k: Optional[np.ndarray] = field(
        default=None, repr=False, compare=False, hash=False
    )

    @property
    def features(self) -> FeatureCache:
        """Lazy-load audio features (thread-safe)."""
        if self._features is None:
            with _features_lock:
                # Double-check after acquiring lock
                if self._features is None:
                    from src.core.features import FeatureExtractor
                    features = FeatureExtractor.extract(self)
                    object.__setattr__(self, '_features', features)
        return self._features

    @property
    def audio_16k(self) -> np.ndarray:
        """
        Get 16kHz mono audio for YAMNet/Whisper (thread-safe).

        Cached to avoid redundant resampling (expensive operation).
        Both YAMNet and Whisper require 16kHz input.
        """
        if self._audio_16k is None:
            with _audio_16k_lock:
                # Double-check after acquiring lock
                if self._audio_16k is None:
                    import librosa
                    mono = self.mono_audio
                    # Resample from 22050 Hz to 16000 Hz
                    audio_16k = librosa.resample(mono, orig_sr=self.sample_rate, target_sr=16000)
                    # Ensure float32 for compatibility
                    audio_16k = audio_16k.astype(np.float32)
                    object.__setattr__(self, '_audio_16k', audio_16k)
        return self._audio_16k

    @property
    def mono_audio(self) -> np.ndarray:
        """Get mono version of audio."""
        if self.channels == 1:
            return self.audio_data
        import numpy as np
        return np.mean(self.audio_data, axis=0)

    @property
    def quality_info(self) -> Dict[str, Any]:
        """Get comprehensive quality information."""
        return {
            'original_sample_rate': self.original_sample_rate,
            'original_bit_depth': self.original_bit_depth,
            'original_format': self.original_format,
            'processed_sample_rate': self.sample_rate,
            'is_high_resolution': self.original_sample_rate >= 96000,
            'is_high_bit_depth': '24' in self.original_bit_depth or '32' in self.original_bit_depth,
            'quality_tier': self._get_quality_tier(),
            'channels': self.channels,
            'duration': self.duration
        }

    def _get_quality_tier(self) -> str:
        """Classify audio quality into tiers."""
        if self.original_sample_rate >= 96000 and (
            '24' in self.original_bit_depth or '32' in self.original_bit_depth
        ):
            return "High-Resolution"
        elif self.original_sample_rate >= 48000 and (
            '24' in self.original_bit_depth or '32' in self.original_bit_depth
        ):
            return "Professional"
        elif self.original_sample_rate >= 44100 and '16' in self.original_bit_depth:
            return "CD Quality"
        return "Standard"


@dataclass
class SpatialInfo:
    """Spatial audio characteristics (stereo only)."""

    is_moving: bool
    direction: Optional[str]  # "left_to_right", "right_to_left", "static"
    stereo_width: float  # [0.0 (mono) to 1.0 (wide)]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'is_moving': self.is_moving,
            'direction': self.direction,
            'stereo_width': self.stereo_width
        }


@dataclass
class SourceClassification:
    """Audio source identification result."""

    source_type: str
    confidence: float  # [0.0, 1.0]
    characteristics: List[str]
    alternatives: List[Tuple[str, float]]
    spatial_info: Optional[SpatialInfo] = None
    explanation: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate fields."""
        validate_confidence(self.confidence)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'source_type': self.source_type,
            'confidence': self.confidence,
            'characteristics': self.characteristics,
            'alternatives': [
                {'source': src, 'confidence': conf}
                for src, conf in self.alternatives
            ],
            'spatial_info': self.spatial_info.to_dict() if self.spatial_info else None,
            'explanation': self.explanation
        }


@dataclass
class MusicalAnalysis:
    """Musical properties (pitch, key, tonality)."""

    has_pitch: bool
    fundamental_frequency: Optional[float]  # Hz
    note_name: Optional[str]  # e.g., "A4", "C#5"
    pitch_stability: float  # [0.0, 1.0]
    estimated_key: Optional[str]  # e.g., "C major"
    key_confidence: float  # [0.0, 1.0]
    is_ambiguous: bool
    alternative_keys: List[Tuple[str, float]]
    is_atonal: bool
    tonality_score: float  # [0.0, 1.0]
    explanation: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate fields."""
        validate_confidence(self.pitch_stability)
        validate_confidence(self.key_confidence)
        validate_confidence(self.tonality_score)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'has_pitch': self.has_pitch,
            'fundamental_frequency': self.fundamental_frequency,
            'note_name': self.note_name,
            'pitch_stability': self.pitch_stability,
            'estimated_key': self.estimated_key,
            'key_confidence': self.key_confidence,
            'is_ambiguous': self.is_ambiguous,
            'alternative_keys': [
                {'key': k, 'confidence': c}
                for k, c in self.alternative_keys
            ],
            'is_atonal': self.is_atonal,
            'tonality_score': self.tonality_score,
            'explanation': self.explanation
        }


@dataclass
class PercussiveAnalysis:
    """Percussion/drum classification."""

    drum_type: str
    confidence: float  # [0.0, 1.0]
    attack_time: float  # milliseconds
    decay_time: float  # milliseconds
    brightness: float  # spectral centroid in Hz
    is_synthesized: bool
    alternatives: List[Tuple[str, float]]
    explanation: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate fields."""
        validate_confidence(self.confidence)
        validate_drum_type(self.drum_type)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'drum_type': self.drum_type,
            'confidence': self.confidence,
            'attack_time': self.attack_time,
            'decay_time': self.decay_time,
            'brightness': self.brightness,
            'is_synthesized': self.is_synthesized,
            'alternatives': [
                {'drum_type': dt, 'confidence': c}
                for dt, c in self.alternatives
            ],
            'explanation': self.explanation
        }


@dataclass
class RhythmicAnalysis:
    """Rhythmic and tempo analysis."""

    is_one_shot: bool
    has_rhythm: bool
    tempo_bpm: Optional[float]
    num_beats: Optional[float]
    rhythm_confidence: float  # [0.0, 1.0]
    beat_times: Optional[List[float]] = None
    explanation: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate fields."""
        validate_confidence(self.rhythm_confidence)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'is_one_shot': self.is_one_shot,
            'has_rhythm': self.has_rhythm,
            'tempo_bpm': self.tempo_bpm,
            'num_beats': self.num_beats,
            'rhythm_confidence': self.rhythm_confidence,
            'beat_times': self.beat_times,
            'explanation': self.explanation
        }


@dataclass
class LLMAnalysis:
    """LLM-based audio analysis (naming, description, speech detection)."""

    # Audio naming
    suggested_name: str
    name_confidence: float  # [0.0, 1.0]

    # Description
    description: str

    # Speech detection
    contains_speech: bool
    detected_words: Optional[List[str]]  # Words if speech present
    speech_language: Optional[str]  # e.g., "en", "es"

    # Metadata
    model_used: str  # e.g., "togetherai/llama-3", "openai/gpt-4"
    
    # Optional fields (with defaults - must come last)
    tags: List[str] = field(default_factory=list)
    transcription: Optional[str] = None  # Full transcription if speech present
    speech_confidence: Optional[float] = None  # Confidence in speech detection (0.0-1.0)
    explanation: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate fields."""
        validate_confidence(self.name_confidence)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'suggested_name': self.suggested_name,
            'name_confidence': self.name_confidence,
            'description': self.description,
            'contains_speech': self.contains_speech,
            'detected_words': self.detected_words,
            'speech_language': self.speech_language,
            'transcription': self.transcription,
            'speech_confidence': self.speech_confidence,
            'tags': self.tags,
            'model_used': self.model_used,
            'explanation': self.explanation
        }


@dataclass
class AnalysisResult:
    """Complete analysis result for an audio sample."""

    # Identification
    audio_sample_hash: str
    timestamp: datetime
    processing_time: float  # seconds

    # Quality metadata
    quality_metadata: Dict[str, Any]

    # Analysis components (optional - may be None if failed)
    source_classification: Optional[SourceClassification]
    musical_analysis: Optional[MusicalAnalysis]
    percussive_analysis: Optional[PercussiveAnalysis]
    rhythmic_analysis: Optional[RhythmicAnalysis]
    llm_analysis: Optional[LLMAnalysis] = None

    # Metadata
    analyzer_versions: Dict[str, str] = field(default_factory=dict)
    used_fallback: Dict[str, bool] = field(default_factory=dict)
    from_cache: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'audio_sample_hash': self.audio_sample_hash,
            'timestamp': self.timestamp.isoformat(),
            'processing_time': self.processing_time,
            'quality_metadata': self.quality_metadata,
            'source_classification': (
                self.source_classification.to_dict()
                if self.source_classification else None
            ),
            'musical_analysis': (
                self.musical_analysis.to_dict()
                if self.musical_analysis else None
            ),
            'percussive_analysis': (
                self.percussive_analysis.to_dict()
                if self.percussive_analysis else None
            ),
            'rhythmic_analysis': (
                self.rhythmic_analysis.to_dict()
                if self.rhythmic_analysis else None
            ),
            'llm_analysis': (
                self.llm_analysis.to_dict()
                if self.llm_analysis else None
            ),
            'analyzer_versions': self.analyzer_versions,
            'used_fallback': self.used_fallback,
            'from_cache': self.from_cache
        }

    def to_json(self, indent: int = 2) -> str:
        """Export as JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def is_complete(self) -> bool:
        """Check if all analyses completed successfully."""
        return (
            self.source_classification is not None and
            self.musical_analysis is not None and
            self.percussive_analysis is not None and
            self.rhythmic_analysis is not None
        )

    def get_summary(self) -> str:
        """Get human-readable summary."""
        parts = []

        # LLM analysis first (name and description)
        if self.llm_analysis:
            parts.append(f"Name: {self.llm_analysis.suggested_name}")

        if self.source_classification:
            parts.append(f"Source: {self.source_classification.source_type}")

        if self.musical_analysis and self.musical_analysis.has_pitch:
            parts.append(f"Note: {self.musical_analysis.note_name}")

        if self.percussive_analysis:
            parts.append(f"Drum: {self.percussive_analysis.drum_type}")

        if self.rhythmic_analysis and self.rhythmic_analysis.has_rhythm:
            parts.append(f"Tempo: {self.rhythmic_analysis.tempo_bpm:.2f} BPM")

        # Add speech transcription if available (from Whisper)
        if self.llm_analysis:
            # Show transcription if available, regardless of contains_speech flag
            # (Whisper may produce transcription even if confidence is low)
            if self.llm_analysis.transcription:
                # Truncate long transcriptions for summary
                transcription = self.llm_analysis.transcription.strip()
                if len(transcription) > 60:
                    transcription = transcription[:57] + "..."
                parts.append(f"Speech: \"{transcription}\"")
            elif self.llm_analysis.contains_speech and self.llm_analysis.detected_words:
                # Fallback to words if no full transcription but words detected
                words_str = ", ".join(self.llm_analysis.detected_words[:5])
                if len(self.llm_analysis.detected_words) > 5:
                    words_str += "..."
                parts.append(f"Words: {words_str}")

        return " | ".join(parts) if parts else "No analysis results"


# Validation helpers

def validate_confidence(confidence: float) -> None:
    """Validate confidence score is in valid range."""
    if not (0.0 <= confidence <= 1.0):
        raise ValueError(f"Confidence must be in [0.0, 1.0], got {confidence}")


def validate_drum_type(drum_type: str) -> None:
    """Validate drum type is one of allowed values."""
    valid_types = {
        "kick", "snare", "hihat", "rim", "tom",
        "cymbal", "shaker", "other", "unknown"
    }
    if drum_type not in valid_types:
        raise ValueError(
            f"Invalid drum type: {drum_type}. Must be one of {valid_types}"
        )
