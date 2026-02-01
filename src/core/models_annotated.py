"""
Core data models for the Audio Sample Analysis Application.

=============================================================================
ANNOTATED VERSION - Extensive comments for educational purposes
=============================================================================

This module contains immutable domain models representing audio samples
and analysis results. These are the core data structures that flow
through the entire application.

OVERVIEW FOR JUNIOR DEVELOPERS:
-------------------------------
Data models are the "nouns" of your application - they represent the
things your system works with. In this case:
- AudioSample: A loaded audio file
- FeatureCache: Extracted audio features
- SourceClassification: What made the sound
- MusicalAnalysis: Pitch, key, tonality
- PercussiveAnalysis: Drum classification
- RhythmicAnalysis: Tempo and beats
- AnalysisResult: Everything combined

KEY CONCEPTS EXPLAINED:

1. DATACLASSES (@dataclass)
   Python's way to create classes that primarily hold data.
   Instead of writing __init__, __repr__, __eq__ manually, the decorator
   generates them for you.

   @dataclass
   class Point:
       x: float
       y: float

   Is equivalent to:
   class Point:
       def __init__(self, x, y):
           self.x = x
           self.y = y
       def __repr__(self):
           return f"Point(x={self.x}, y={self.y})"
       # ... and more

2. IMMUTABILITY (frozen=True)
   Frozen dataclasses cannot be modified after creation.
   This prevents bugs where code accidentally mutates shared data.

   sample = AudioSample(...)
   sample.duration = 10  # ERROR! Cannot modify frozen instance

3. TYPE HINTS
   Tell Python what types are expected. They don't enforce anything
   at runtime (Python is dynamically typed), but:
   - IDEs use them for autocomplete
   - Static analyzers catch type errors
   - Documentation for developers

4. LAZY LOADING
   Expensive operations (like feature extraction) are delayed until
   needed. This improves startup time and memory usage.

5. NUMPY ARRAYS (np.ndarray)
   NumPy is the foundation of scientific computing in Python.
   Arrays are like lists but:
   - Fixed size (more memory efficient)
   - Support vectorized operations (much faster)
   - Required by ML libraries (TensorFlow, librosa)
"""

# =============================================================================
# IMPORTS
# =============================================================================

from __future__ import annotations  # Allows forward references in type hints

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

# TYPE_CHECKING is False at runtime, True when running type checkers
# This prevents circular imports while still allowing type hints
if TYPE_CHECKING:
    from src.core.features import FeatureExtractor


# =============================================================================
# FEATURE CACHE
# =============================================================================

@dataclass(frozen=True)
class FeatureCache:
    """
    Cached audio features extracted from AudioSample.

    All features are computed once and cached for reuse across analyzers.
    This significantly improves performance when multiple analyzers need
    the same features.

    WHAT ARE THESE FEATURES?
    Audio features are numerical representations of audio characteristics.
    They transform raw audio waveforms into meaningful numbers that ML
    models can understand.

    FEATURE DESCRIPTIONS:

    MFCC (Mel-Frequency Cepstral Coefficients):
    - The most important feature for audio classification
    - Represents the "shape" of the audio spectrum
    - Based on how humans perceive sound
    - Shape: (n_mfcc, n_frames) typically (13, ~100)

    Spectral Centroid:
    - The "center of mass" of the spectrum
    - Correlates with perceived "brightness"
    - High centroid = bright sound (cymbals)
    - Low centroid = dark sound (bass)

    Spectral Rolloff:
    - Frequency below which X% of spectrum energy lies
    - Helps distinguish voiced/unvoiced sounds

    Spectral Bandwidth:
    - How "spread out" the spectrum is
    - Wide = complex sound, Narrow = pure tone

    Zero Crossing Rate:
    - How often the signal crosses zero
    - High for noisy/percussive sounds
    - Low for harmonic/tonal sounds

    Harmonic/Percussive:
    - Separation of audio into:
      - Harmonic: Tonal parts (sustained notes)
      - Percussive: Transient parts (drums, attacks)

    Chroma:
    - 12-dimensional pitch class representation
    - Each dimension = one note (C, C#, D, ..., B)
    - Useful for harmony analysis

    Tonnetz:
    - 6-dimensional tonal space representation
    - Based on musical harmony theory
    - Captures relationships between pitches

    Onset Envelope:
    - Strength of "attacks" over time
    - Peaks indicate note/event onsets

    Tempogram:
    - Tempo over time
    - Shows rhythmic structure
    """

    # =========================================================================
    # SPECTRAL FEATURES
    # =========================================================================

    # Mel-Frequency Cepstral Coefficients
    # Shape: (n_mfcc, n_frames) - typically (13, ~86 frames per second)
    # These are the most important features for most audio classification tasks
    mfcc: np.ndarray

    # Spectral centroid - indicates brightness of sound
    # Shape: (1, n_frames)
    # Unit: Hz (frequency)
    spectral_centroid: np.ndarray

    # Spectral rolloff - frequency below which 85% of energy lies
    # Shape: (1, n_frames)
    # Unit: Hz
    spectral_rolloff: np.ndarray

    # Spectral bandwidth - width of the spectrum
    # Shape: (1, n_frames)
    # Unit: Hz
    spectral_bandwidth: np.ndarray

    # Zero crossing rate - how often signal changes sign
    # Shape: (1, n_frames)
    # Unit: ratio (0.0 to ~0.5)
    zero_crossing_rate: np.ndarray

    # =========================================================================
    # HARMONIC/PERCUSSIVE SEPARATION
    # =========================================================================

    # Harmonic component - sustained tonal elements
    # Shape: (samples,) - same length as original audio
    # HPSS (Harmonic-Percussive Source Separation) extracts this
    harmonic: np.ndarray

    # Percussive component - transient drum-like elements
    # Shape: (samples,)
    percussive: np.ndarray

    # =========================================================================
    # PITCH FEATURES
    # =========================================================================

    # Chroma features - pitch class energy distribution
    # Shape: (12, n_frames) - one bin per semitone (C, C#, D, ..., B)
    # Values show how much energy is in each pitch class
    chroma: np.ndarray

    # Tonnetz features - tonal centroid features
    # Shape: (6, n_frames)
    # Based on harmonic theory - captures tonal relationships
    tonnetz: np.ndarray

    # =========================================================================
    # ONSET DETECTION
    # =========================================================================

    # Onset strength envelope - how "attacky" is each moment
    # Shape: (n_frames,)
    # Peaks indicate where new notes/events start
    onset_envelope: np.ndarray

    # Detected onset frame indices
    # Shape: (n_onsets,) - variable length
    # These are the frame numbers where onsets were detected
    onset_frames: np.ndarray

    # =========================================================================
    # TEMPO/RHYTHM FEATURES
    # =========================================================================

    # Tempogram - tempo strength over time for different tempos
    # Shape: (n_tempo_bins, n_frames)
    # Shows which tempos are present at each time
    tempogram: np.ndarray

    # =========================================================================
    # METADATA
    # =========================================================================

    # When features were extracted (for cache validation)
    # default_factory ensures each instance gets its own datetime
    extracted_at: datetime = field(default_factory=datetime.utcnow)


# =============================================================================
# AUDIO SAMPLE
# =============================================================================

@dataclass(frozen=True)
class AudioSample:
    """
    Immutable representation of a loaded audio file.

    This is the primary data structure that flows through the analysis
    pipeline. It contains:
    1. The audio data itself (as numpy array)
    2. Metadata about the original file
    3. Lazy-loaded features

    WHY FROZEN (IMMUTABLE)?
    - Prevents accidental modification
    - Safe to cache and share between threads
    - Guarantees data integrity throughout pipeline

    AUDIO DATA FORMAT:
    - Mono: shape (samples,) - 1D array
    - Stereo: shape (2, samples) - 2D array, first axis is channels
    - Always float32 in range [-1.0, 1.0]
    - Always resampled to 22050 Hz

    EXAMPLE USAGE:
        sample = loader.load(Path("test.wav"))
        print(f"Duration: {sample.duration}s")
        print(f"Quality: {sample.quality_info['quality_tier']}")

        # Features are computed on first access
        mfcc = sample.features.mfcc
    """

    # =========================================================================
    # FILE IDENTIFICATION
    # =========================================================================

    # Path to the original audio file
    # Using Path instead of str for type safety and convenience methods
    file_path: Path

    # SHA-256 hash of file contents
    # Used for:
    # - Cache keys (same content = same hash)
    # - Duplicate detection
    # - Result identification
    file_hash: str

    # =========================================================================
    # PROCESSED AUDIO PROPERTIES
    # =========================================================================

    # Sample rate after resampling
    # Always 22050 Hz in our system (standardized for analysis)
    sample_rate: int

    # Number of audio channels
    # 1 = mono, 2 = stereo
    channels: int

    # Audio duration in seconds
    # Calculated from: len(audio_data) / sample_rate
    duration: float

    # The actual audio data as numpy array
    # Mono: shape (samples,)
    # Stereo: shape (2, samples) or (channels, samples)
    # dtype: float32, range [-1.0, 1.0]
    audio_data: np.ndarray

    # =========================================================================
    # ORIGINAL QUALITY METADATA
    # =========================================================================

    # Original sample rate before resampling
    # Examples: 44100, 48000, 96000, 192000
    original_sample_rate: int

    # Original bit depth/format
    # Examples: 'PCM_16', 'PCM_24', 'FLOAT', 'PCM_32'
    original_bit_depth: str

    # Original file format
    # Examples: 'WAV', 'AIFF', 'MP3', 'FLAC'
    original_format: str

    # =========================================================================
    # LAZY-LOADED FEATURES
    # =========================================================================

    # Features are expensive to compute (~100-500ms)
    # We only compute them when first accessed via the .features property
    #
    # Field options:
    # - default=None: Start with no features computed
    # - repr=False: Don't include in string representation (too large)
    # - compare=False: Don't use for equality comparison
    # - hash=False: Don't use for hashing (arrays aren't hashable anyway)
    _features: Optional[FeatureCache] = field(
        default=None, repr=False, compare=False, hash=False
    )

    # =========================================================================
    # COMPUTED PROPERTIES
    # =========================================================================

    @property
    def features(self) -> FeatureCache:
        """
        Lazy-load audio features.

        Features are expensive to compute (~100-500ms for 30s audio),
        so we only compute them when actually needed.

        LAZY LOADING PATTERN:
        1. First access: features is None
        2. Compute features (expensive)
        3. Cache in _features
        4. Subsequent access: return cached value

        HOW WE MODIFY A FROZEN DATACLASS:
        Normally, frozen dataclasses cannot be modified. But we need
        to cache the features. We use object.__setattr__ to bypass
        the frozen restriction. This is a standard pattern for lazy
        initialization in frozen dataclasses.

        Returns:
            FeatureCache: All extracted features

        EXAMPLE:
            sample = loader.load("test.wav")
            # Features computed here on first access
            mfcc = sample.features.mfcc
            # Features already cached, instant access
            chroma = sample.features.chroma
        """
        if self._features is None:
            # Import here to avoid circular dependency
            # models.py is imported by features.py, so we can't import
            # at module level
            from src.core.features import FeatureExtractor

            # Extract all features (expensive operation)
            features = FeatureExtractor.extract(self)

            # Cache the result by directly setting the attribute
            # object.__setattr__ bypasses the frozen restriction
            object.__setattr__(self, '_features', features)

        return self._features

    @property
    def mono_audio(self) -> np.ndarray:
        """
        Get mono version of audio.

        Many analysis algorithms expect mono audio. This property
        provides a consistent way to get mono data regardless of
        whether the original is mono or stereo.

        HOW STEREO TO MONO WORKS:
        - Average the left and right channels
        - np.mean(data, axis=0) averages along the channel axis

        Returns:
            np.ndarray: Mono audio data, shape (samples,)
        """
        if self.channels == 1:
            # Already mono, return as-is
            return self.audio_data
        # Stereo: average left and right channels
        return np.mean(self.audio_data, axis=0)

    @property
    def quality_info(self) -> Dict[str, Any]:
        """
        Get comprehensive quality information.

        This provides a summary of the audio quality for display
        to users or for quality-aware processing decisions.

        QUALITY TIERS:
        - High-Resolution: 96kHz+, 24/32-bit (audiophile)
        - Professional: 48kHz+, 24/32-bit (studio)
        - CD Quality: 44.1kHz, 16-bit (consumer)
        - Standard: Everything else (compressed, low-fi)

        Returns:
            Dict with quality information
        """
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
        """
        Classify audio quality into tiers.

        This is used for user-facing quality labels and potentially
        for adjusting analysis parameters based on input quality.

        QUALITY CLASSIFICATION LOGIC:
        1. Check sample rate first (higher = better)
        2. Then check bit depth (24/32 > 16)
        3. Assign to appropriate tier

        Returns:
            str: Quality tier name
        """
        # High-Resolution: 96kHz+ at 24/32-bit
        if self.original_sample_rate >= 96000 and (
            '24' in self.original_bit_depth or '32' in self.original_bit_depth
        ):
            return "High-Resolution"

        # Professional: 48kHz+ at 24/32-bit
        elif self.original_sample_rate >= 48000 and (
            '24' in self.original_bit_depth or '32' in self.original_bit_depth
        ):
            return "Professional"

        # CD Quality: 44.1kHz at 16-bit
        elif self.original_sample_rate >= 44100 and '16' in self.original_bit_depth:
            return "CD Quality"

        # Everything else
        return "Standard"


# =============================================================================
# ANALYSIS RESULT MODELS
# =============================================================================

@dataclass
class SpatialInfo:
    """
    Spatial audio characteristics (stereo only).

    For stereo audio, we can analyze the spatial properties:
    - Is the sound moving in the stereo field?
    - Which direction is it moving?
    - How wide is the stereo image?

    STEREO WIDTH EXPLAINED:
    - 0.0 = Mono (left and right are identical)
    - 1.0 = Wide stereo (left and right are uncorrelated)
    - Most music is somewhere in between

    MOVEMENT DETECTION:
    We analyze energy distribution over time in each channel.
    If energy shifts from left to right, the sound is "moving".

    EXAMPLE:
        If a car passes by in stereo recording:
        - is_moving=True
        - direction="left_to_right"
        - stereo_width depends on recording technique
    """

    is_moving: bool  # Does sound move in stereo field?
    direction: Optional[str]  # "left_to_right", "right_to_left", "static"
    stereo_width: float  # [0.0 (mono) to 1.0 (wide)]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'is_moving': self.is_moving,
            'direction': self.direction,
            'stereo_width': self.stereo_width
        }


@dataclass
class SourceClassification:
    """
    Audio source identification result.

    Answers: "What made this sound?"

    EXAMPLES:
    - source_type: "car", characteristics: ["moving left to right"]
    - source_type: "dog", characteristics: ["barking"]
    - source_type: "music", characteristics: ["guitar", "electronic"]

    THE YAMNet MODEL:
    Our primary classifier uses Google's YAMNet model, which
    recognizes 521 audio classes from the AudioSet dataset.
    Classes include: speech, music, animals, vehicles, etc.

    CONFIDENCE EXPLAINED:
    - 0.0 = Model has no idea
    - 0.5 = About as likely as random guess
    - 0.75 = Reasonably confident (our fallback threshold)
    - 1.0 = Completely certain

    In practice:
    - > 0.9 = Very confident, almost always correct
    - 0.7-0.9 = Good confidence, usually correct
    - < 0.7 = May need human verification or LLM fallback
    """

    source_type: str  # Primary classification (e.g., "car", "dog")
    confidence: float  # How confident is the model [0.0, 1.0]
    characteristics: List[str]  # Additional descriptors
    alternatives: List[Tuple[str, float]]  # Other possibilities [(type, conf), ...]
    spatial_info: Optional[SpatialInfo] = None  # For stereo audio
    explanation: Optional[str] = None  # LLM-generated explanation

    def __post_init__(self) -> None:
        """
        Validate fields after initialization.

        __post_init__ is called automatically after the dataclass
        __init__ method. It's the place to add validation logic.
        """
        validate_confidence(self.confidence)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.

        SERIALIZATION:
        We need to convert our dataclass to a dict/JSON for:
        - API responses
        - Database storage
        - Cache serialization
        - Logging
        """
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
    """
    Musical properties (pitch, key, tonality).

    Answers questions like:
    - Does this sound have a pitch?
    - What note is it?
    - What key is the music in?
    - Is it tonal or atonal?

    PITCH DETECTION:
    Not all sounds have pitch. A snare drum doesn't have a clear
    pitch, but a piano note does.
    - has_pitch: True if discernible pitch exists
    - fundamental_frequency: The F0 in Hz (A4 = 440 Hz)
    - note_name: Musical notation (A4, C#5, Bb3, etc.)

    KEY DETECTION:
    For music, we try to identify the key (C major, A minor, etc.)
    - estimated_key: Our best guess
    - key_confidence: How sure we are
    - is_ambiguous: True if multiple keys are equally likely
    - alternative_keys: Other possible keys

    TONALITY:
    Some music is "atonal" - it doesn't have a tonal center.
    - is_atonal: True if no clear tonal center
    - tonality_score: 0.0 (atonal) to 1.0 (strongly tonal)
    """

    # Pitch information
    has_pitch: bool  # Does audio have discernible pitch?
    fundamental_frequency: Optional[float]  # F0 in Hz
    note_name: Optional[str]  # e.g., "A4", "C#5"
    pitch_stability: float  # How stable is pitch? [0.0, 1.0]

    # Key estimation
    estimated_key: Optional[str]  # e.g., "C major", "A minor"
    key_confidence: float  # [0.0, 1.0]
    is_ambiguous: bool  # Is key ambiguous?
    alternative_keys: List[Tuple[str, float]]  # [(key, confidence), ...]

    # Tonality
    is_atonal: bool  # Is audio atonal?
    tonality_score: float  # 0.0 (atonal) to 1.0 (tonal)

    # LLM explanation (optional)
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
    """
    Percussion/drum classification.

    Answers: "What type of drum or percussive sound is this?"

    DRUM TYPES:
    - kick: Bass drum, low thump
    - snare: Snare drum, crack/snap
    - hihat: Hi-hat cymbal (open or closed)
    - rim: Rimshot
    - tom: Tom-tom drum
    - cymbal: Crash, ride, or other cymbal
    - shaker: Shaker, tambourine, etc.
    - other: Other percussion

    TIMBRAL CHARACTERISTICS:
    - attack_time: How quickly sound reaches peak (ms)
      - Short attack (~1-10ms) = sharp, percussive
      - Long attack (~50-100ms) = soft, swelling
    - decay_time: How quickly sound fades (ms)
    - brightness: Spectral centroid (Hz)
      - High = bright (cymbals, ~8000+ Hz)
      - Low = dark (kick drums, ~200-500 Hz)

    SYNTHESIZED DETECTION:
    Real drums have natural variations and noise.
    Synthesized drums are more "perfect" and regular.
    is_synthesized indicates if the sound is likely electronic.
    """

    drum_type: str  # One of the defined types
    confidence: float  # [0.0, 1.0]

    # Timbral characteristics
    attack_time: float  # milliseconds
    decay_time: float  # milliseconds
    brightness: float  # spectral centroid in Hz

    # Synthesis detection
    is_synthesized: bool  # Is this a synthesized drum?

    # Alternatives
    alternatives: List[Tuple[str, float]]  # [(drum_type, confidence), ...]

    # LLM explanation
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
    """
    Rhythmic and tempo analysis.

    Answers:
    - Is this a one-shot (single hit) or a loop?
    - What's the tempo (BPM)?
    - How many beats are there?
    - Where are the beats?

    ONE-SHOT vs LOOP:
    - One-shot: Single sound event (a kick drum hit)
    - Loop: Repeating pattern (a drum beat)

    TEMPO (BPM):
    Beats Per Minute - how fast the rhythm is.
    - 60 BPM = 1 beat per second
    - 120 BPM = 2 beats per second
    - Most music is 60-180 BPM

    BEAT TRACKING:
    We detect where beats occur in the audio.
    beat_times is a list of timestamps (in seconds) where
    beats were detected.
    """

    is_one_shot: bool  # Is this a single event?
    has_rhythm: bool  # Does audio have rhythmic pattern?

    # Tempo (if has_rhythm)
    tempo_bpm: Optional[float]  # Beats per minute
    num_beats: Optional[float]  # Number of beats (can be fractional)
    rhythm_confidence: float  # How confident in rhythm detection

    # Beat locations (if has_rhythm)
    beat_times: Optional[List[float]] = None  # Timestamps in seconds

    # LLM explanation
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


# =============================================================================
# COMPLETE ANALYSIS RESULT
# =============================================================================

@dataclass
class AnalysisResult:
    """
    Complete analysis result for an audio sample.

    This is the final output of the analysis pipeline, containing
    all analysis components plus metadata.

    STRUCTURE:
    - Identification: hash and timestamp
    - Quality: original audio quality info
    - Analysis: source, musical, percussive, rhythmic
    - Metadata: versions, fallback usage

    PARTIAL RESULTS:
    Analysis components are Optional because some may fail.
    A failed analyzer returns None for its component.
    The system returns partial results rather than failing completely.

    FALLBACK TRACKING:
    used_fallback dict shows which analyzers used the OpenAI fallback.
    This is important for:
    - Cost tracking (LLM calls cost money)
    - Quality assessment (fallback may be more accurate)
    - Debugging (understanding analysis path)
    """

    # =========================================================================
    # IDENTIFICATION
    # =========================================================================

    audio_sample_hash: str  # SHA-256 of audio file
    timestamp: datetime  # When analysis was performed
    processing_time: float  # Total time in seconds

    # =========================================================================
    # QUALITY METADATA
    # =========================================================================

    quality_metadata: Dict[str, Any]  # From AudioSample.quality_info

    # =========================================================================
    # ANALYSIS COMPONENTS
    # =========================================================================

    # Each is Optional - None if that analysis failed
    source_classification: Optional[SourceClassification]
    musical_analysis: Optional[MusicalAnalysis]
    percussive_analysis: Optional[PercussiveAnalysis]
    rhythmic_analysis: Optional[RhythmicAnalysis]

    # =========================================================================
    # METADATA
    # =========================================================================

    # Version of each analyzer used
    analyzer_versions: Dict[str, str]  # {"source": "1.0.0", ...}

    # Whether LLM fallback was used for each analyzer
    used_fallback: Dict[str, bool]  # {"source": True, "musical": False, ...}

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.

        This is used for:
        - API responses
        - Database storage
        - Cache serialization
        """
        return {
            'audio_sample_hash': self.audio_sample_hash,
            'timestamp': self.timestamp.isoformat(),  # Convert datetime to string
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
            'analyzer_versions': self.analyzer_versions,
            'used_fallback': self.used_fallback
        }

    def to_json(self, indent: int = 2) -> str:
        """
        Export as JSON string.

        Args:
            indent: Number of spaces for indentation (0 for compact)

        Returns:
            str: JSON-formatted string

        The default=str argument handles any non-serializable types
        (like datetime or Path) by converting them to strings.
        """
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def is_complete(self) -> bool:
        """
        Check if all analyses completed successfully.

        Returns:
            bool: True if all four analyses have results

        USAGE:
            if not result.is_complete():
                logger.warning("Some analyses failed")
        """
        return (
            self.source_classification is not None and
            self.musical_analysis is not None and
            self.percussive_analysis is not None and
            self.rhythmic_analysis is not None
        )

    def get_summary(self) -> str:
        """
        Get human-readable summary.

        Returns:
            str: One-line summary of results

        EXAMPLE OUTPUT:
        "Source: music | Note: A4 | Drum: kick | Tempo: 120.00 BPM"
        """
        parts = []

        if self.source_classification:
            parts.append(f"Source: {self.source_classification.source_type}")

        if self.musical_analysis and self.musical_analysis.has_pitch:
            parts.append(f"Note: {self.musical_analysis.note_name}")

        if self.percussive_analysis:
            parts.append(f"Drum: {self.percussive_analysis.drum_type}")

        if self.rhythmic_analysis and self.rhythmic_analysis.has_rhythm:
            parts.append(f"Tempo: {self.rhythmic_analysis.tempo_bpm:.2f} BPM")

        return " | ".join(parts) if parts else "No analysis results"


# =============================================================================
# VALIDATION HELPERS
# =============================================================================

def validate_confidence(confidence: float) -> None:
    """
    Validate confidence score is in valid range.

    Confidence scores must be in [0.0, 1.0] by convention.
    This validation catches bugs early.

    Args:
        confidence: Score to validate

    Raises:
        ValueError: If confidence is out of range
    """
    if not (0.0 <= confidence <= 1.0):
        raise ValueError(f"Confidence must be in [0.0, 1.0], got {confidence}")


def validate_drum_type(drum_type: str) -> None:
    """
    Validate drum type is one of allowed values.

    This ensures consistency in drum classification results.

    Args:
        drum_type: Type to validate

    Raises:
        ValueError: If drum_type is not in valid set
    """
    valid_types = {
        "kick", "snare", "hihat", "rim", "tom",
        "cymbal", "shaker", "other", "unknown"
    }
    if drum_type not in valid_types:
        raise ValueError(
            f"Invalid drum type: {drum_type}. Must be one of {valid_types}"
        )


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

"""
EXAMPLE 1: Working with AudioSample

    from src.core.loader import AudioLoader

    loader = AudioLoader()
    sample = loader.load(Path("test.wav"))

    # Basic properties
    print(f"Duration: {sample.duration}s")
    print(f"Channels: {sample.channels}")
    print(f"Quality: {sample.quality_info['quality_tier']}")

    # Lazy-loaded features (computed on first access)
    mfcc = sample.features.mfcc
    print(f"MFCC shape: {mfcc.shape}")

EXAMPLE 2: Working with AnalysisResult

    from src.core.engine import create_analysis_engine

    engine = create_analysis_engine(config)
    result = engine.analyze(Path("test.wav"))

    # Check completeness
    if not result.is_complete():
        print("Warning: Some analyses failed")

    # Get summary
    print(result.get_summary())

    # Export to JSON
    json_str = result.to_json()

    # Access specific results
    if result.source_classification:
        print(f"Source: {result.source_classification.source_type}")
        print(f"Confidence: {result.source_classification.confidence}")

EXAMPLE 3: Creating analysis results manually (for testing)

    from datetime import datetime

    source = SourceClassification(
        source_type="music",
        confidence=0.92,
        characteristics=["electronic", "synthesized"],
        alternatives=[("speech", 0.05), ("noise", 0.03)]
    )

    result = AnalysisResult(
        audio_sample_hash="abc123",
        timestamp=datetime.utcnow(),
        processing_time=0.5,
        quality_metadata={"quality_tier": "CD Quality"},
        source_classification=source,
        musical_analysis=None,
        percussive_analysis=None,
        rhythmic_analysis=None,
        analyzer_versions={"source": "1.0.0"},
        used_fallback={"source": False}
    )
"""
