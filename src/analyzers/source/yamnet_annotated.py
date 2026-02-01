"""
YAMNet source analyzer for the Audio Sample Analysis Application.

=============================================================================
ANNOTATED VERSION - Extensive comments for educational purposes
=============================================================================

This analyzer uses Google's YAMNet model to classify audio sources.
YAMNet is a pre-trained neural network that can recognize 521 different
types of sounds.

OVERVIEW FOR JUNIOR DEVELOPERS:
-------------------------------
YAMNet (Yet Another Mobile Net) is a deep learning model designed for
audio classification. It was trained on the AudioSet dataset, which
contains millions of labeled audio clips from YouTube.

WHAT CAN YAMNET RECOGNIZE?
YAMNet recognizes 521 classes including:
- Human sounds: Speech, laughter, coughing, snoring
- Music: Various instruments, genres, singing
- Animals: Dogs, cats, birds, insects
- Vehicles: Cars, motorcycles, airplanes
- Nature: Rain, wind, thunder, ocean
- Domestic: Doorbell, alarm, vacuum cleaner
- And many more!

HOW YAMNET WORKS:
1. Audio is converted to a spectrogram (visual representation)
2. The spectrogram is divided into 0.96-second frames
3. Each frame is processed by a MobileNet-based neural network
4. Output: 521 class probabilities per frame
5. We average across frames to get overall classification

TECHNICAL DETAILS:
- Input: 16kHz mono audio (we resample from 22kHz)
- Architecture: MobileNet V1
- Output: 521 class probabilities
- Model size: ~5 MB
- Inference time: ~50-100ms per second of audio

TENSORFLOW HUB:
YAMNet is distributed via TensorFlow Hub, a repository of pre-trained
models. We load it with a URL and it downloads automatically.
"""

# =============================================================================
# IMPORTS
# =============================================================================

import csv          # For reading the class names CSV file
import logging      # For logging messages
import urllib.request  # For downloading the class names file
from pathlib import Path
from typing import List, Optional, Tuple

import librosa      # For audio resampling
import numpy as np  # For numerical operations

from src.core.analyzer_base import BaseAnalyzer
from src.core.models import AudioSample, SourceClassification, SpatialInfo
from src.utils.errors import ModelLoadError


# =============================================================================
# CONSTANTS
# =============================================================================

# URL for the YAMNet model on TensorFlow Hub
# This is Google's official hosted version
YAMNET_MODEL_URL = 'https://tfhub.dev/google/yamnet/1'

# URL for the class names mapping
# Maps class index (0-520) to human-readable names
YAMNET_CLASSES_URL = 'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'


# =============================================================================
# YAMNET ANALYZER CLASS
# =============================================================================

class YAMNetAnalyzer(BaseAnalyzer[SourceClassification]):
    """
    YAMNet-based source classification.

    Uses Google's YAMNet model to classify what type of sound is in
    the audio. This is the "source analyzer" - it identifies the
    source that made the sound.

    INHERITANCE:
    This class inherits from BaseAnalyzer[SourceClassification].
    - BaseAnalyzer provides: timing, logging, error handling
    - [SourceClassification] specifies the return type

    The BaseAnalyzer template method pattern works like this:
    1. analyze() is called (in BaseAnalyzer)
    2. analyze() calls _analyze_impl() (which we implement here)
    3. BaseAnalyzer handles timing and error wrapping

    LAZY LOADING:
    The model and class names are loaded "lazily" - only when first
    needed. This improves startup time because the model (which
    requires downloading and loading into memory) isn't loaded until
    an analysis is actually requested.

    EXAMPLE USAGE:
        analyzer = YAMNetAnalyzer()

        # First analysis - model loads (slow)
        result1 = analyzer.analyze(audio1)

        # Second analysis - model already loaded (fast)
        result2 = analyzer.analyze(audio2)

        print(f"This sounds like: {result1.source_type}")
        print(f"Confidence: {result1.confidence:.2%}")
    """

    def __init__(self, model_url: str = YAMNET_MODEL_URL):
        """
        Initialize YAMNet analyzer.

        Args:
            model_url: TensorFlow Hub URL for YAMNet model.
                      You can use a different URL if you have a
                      custom or local model.

        NOTE ON LAZY LOADING:
        We don't load the model here in __init__. Instead, we set
        _model to None and load it on first use via the property.
        This is called "lazy initialization".
        """
        # Call parent constructor
        # "yamnet" is the analyzer name, "1.0.0" is version
        super().__init__("yamnet", "1.0.0")

        # Store the model URL for later loading
        self.model_url = model_url

        # Model and class names will be loaded on first use
        # Using None as initial value for lazy loading
        self._model = None
        self._class_names: Optional[List[str]] = None

    # =========================================================================
    # LAZY-LOADED PROPERTIES
    # =========================================================================

    @property
    def model(self):
        """
        Lazy-load YAMNet model.

        The model is loaded on first access and then cached.
        This is a common pattern for expensive resources.

        Returns:
            The loaded TensorFlow Hub model

        WHY LAZY LOADING?
        - Model download is slow (~5 MB download)
        - Model loading uses memory
        - Not all code paths need the model
        - Faster startup time
        """
        if self._model is None:
            self._model = self._load_model()
        return self._model

    @property
    def class_names(self) -> List[str]:
        """
        Lazy-load class names.

        The class names map YAMNet's output indices (0-520) to
        human-readable names like "Speech", "Dog", "Piano", etc.

        Returns:
            List[str]: 521 class names
        """
        if self._class_names is None:
            self._class_names = self._load_class_names()
        return self._class_names

    # =========================================================================
    # MODEL LOADING
    # =========================================================================

    def _load_model(self):
        """
        Load YAMNet model from TensorFlow Hub.

        TensorFlow Hub is a repository of pre-trained models.
        hub.load() downloads the model and loads it into memory.

        Returns:
            The loaded YAMNet model

        Raises:
            ModelLoadError: If model cannot be loaded

        CACHING:
        TensorFlow Hub caches downloaded models in ~/.tfhub_cache/
        So the first load downloads, subsequent loads use cache.
        """
        try:
            # Import here to avoid TensorFlow import at module level
            # This speeds up import if YAMNet isn't used
            import tensorflow_hub as hub

            self.logger.info("Loading YAMNet model from TensorFlow Hub...")

            # hub.load() downloads and loads the model
            # This may take a while on first run
            model = hub.load(self.model_url)

            self.logger.info(f"YAMNet model loaded successfully")
            return model

        except Exception as e:
            raise ModelLoadError(
                f"Failed to load YAMNet model: {e}",
                model_name="yamnet"
            )

    def _load_class_names(self) -> List[str]:
        """
        Load YAMNet class names from AudioSet ontology.

        The class names CSV file maps indices to names:
        0,/m/09x0r,Speech
        1,/m/05zppz,Male speech, man speaking
        ...

        We extract the third column (display_name).

        Returns:
            List[str]: Class names indexed by class ID
        """
        try:
            # Path to store downloaded class map
            class_map_path = Path("models/yamnet_class_map.csv")
            class_map_path.parent.mkdir(parents=True, exist_ok=True)

            # Download if not already present
            if not class_map_path.exists():
                self.logger.info("Downloading YAMNet class map...")
                urllib.request.urlretrieve(YAMNET_CLASSES_URL, class_map_path)

            # Parse CSV file
            class_names = []
            with open(class_map_path, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header row
                for row in reader:
                    if len(row) >= 3:
                        # Third column is display_name
                        class_names.append(row[2])

            self.logger.info(f"Loaded {len(class_names)} class names")
            return class_names

        except Exception as e:
            self.logger.warning(f"Could not load class names: {e}")
            # Return fallback if download fails
            return self._get_fallback_class_names()

    def _get_fallback_class_names(self) -> List[str]:
        """
        Return fallback class names if download fails.

        This ensures the analyzer can still work even without
        the official class map - we just use generic names.

        Returns:
            List[str]: Basic fallback class names
        """
        return [
            "Speech", "Music", "Animal", "Vehicle", "Natural sounds",
            "Domestic sounds", "Urban sounds", "Silence", "Other"
        ] + [f"Class_{i}" for i in range(512)]  # Pad to 521

    # =========================================================================
    # ANALYSIS IMPLEMENTATION
    # =========================================================================

    def _analyze_impl(self, audio: AudioSample) -> SourceClassification:
        """
        Classify audio source using YAMNet.

        This is the main analysis method (called by BaseAnalyzer.analyze()).

        Process:
        1. Resample audio to 16kHz (YAMNet's expected format)
        2. Run inference through YAMNet
        3. Average scores across frames
        4. Get top 5 predictions
        5. Analyze spatial characteristics (if stereo)
        6. Build and return SourceClassification

        Args:
            audio: AudioSample to classify

        Returns:
            SourceClassification: Classification result with:
                - source_type: Most likely class
                - confidence: Probability (0-1)
                - alternatives: Other possible classes
                - spatial_info: Movement/stereo information (if stereo)
        """
        # =================================================================
        # STEP 1: PREPARE AUDIO FOR YAMNET
        # =================================================================
        # YAMNet expects 16kHz mono audio
        # Our audio is 22050 Hz, so we resample
        audio_16k = self._resample_to_16k(audio.mono_audio)

        # =================================================================
        # STEP 2: RUN YAMNET INFERENCE
        # =================================================================
        # YAMNet returns three outputs:
        # - scores: Class probabilities per frame, shape (N_frames, 521)
        # - embeddings: Feature vectors (not used here)
        # - spectrogram: Mel spectrogram (not used here)
        scores, embeddings, spectrogram = self.model(audio_16k)

        # =================================================================
        # STEP 3: PROCESS SCORES
        # =================================================================
        # Convert TensorFlow tensor to numpy array
        scores_np = scores.numpy()

        # Average scores across all frames
        # This gives us one score per class for the entire audio
        mean_scores = np.mean(scores_np, axis=0)

        # Get indices of top 5 classes (sorted highest to lowest)
        # argsort gives ascending order, [::-1] reverses it
        top_indices = np.argsort(mean_scores)[::-1][:5]

        # =================================================================
        # STEP 4: GET PRIMARY AND ALTERNATIVE PREDICTIONS
        # =================================================================
        # Primary prediction (highest scoring class)
        primary_idx = top_indices[0]
        primary_class = self._get_class_name(primary_idx)
        primary_confidence = float(mean_scores[primary_idx])

        # Alternative predictions (next 4 highest)
        alternatives: List[Tuple[str, float]] = [
            (self._get_class_name(idx), float(mean_scores[idx]))
            for idx in top_indices[1:5]
        ]

        # =================================================================
        # STEP 5: ANALYZE SPATIAL CHARACTERISTICS (STEREO ONLY)
        # =================================================================
        # If audio is stereo, analyze movement and stereo width
        spatial_info = None
        if audio.channels == 2:
            spatial_info = self._analyze_spatial(audio)

        # =================================================================
        # STEP 6: EXTRACT DESCRIPTIVE CHARACTERISTICS
        # =================================================================
        # Add spatial info as characteristics if relevant
        characteristics = self._extract_characteristics(primary_class, spatial_info)

        # =================================================================
        # STEP 7: BUILD AND RETURN RESULT
        # =================================================================
        return SourceClassification(
            source_type=primary_class,
            confidence=primary_confidence,
            characteristics=characteristics,
            alternatives=alternatives,
            spatial_info=spatial_info,
            explanation=None  # No LLM explanation in local mode
        )

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _get_class_name(self, index: int) -> str:
        """
        Get class name by index with bounds checking.

        Args:
            index: Class index (0-520)

        Returns:
            str: Class name or "Unknown_N" if out of bounds
        """
        if 0 <= index < len(self.class_names):
            return self.class_names[index]
        return f"Unknown_{index}"

    def _resample_to_16k(self, audio: np.ndarray) -> np.ndarray:
        """
        Resample from 22050 Hz to 16000 Hz for YAMNet.

        YAMNet was trained on 16kHz audio, so we need to
        resample our 22050 Hz audio to match.

        Args:
            audio: Audio at 22050 Hz

        Returns:
            np.ndarray: Audio at 16000 Hz

        LIBROSA RESAMPLING:
        librosa.resample uses high-quality polyphase filtering
        to change sample rates without introducing artifacts.
        """
        return librosa.resample(audio, orig_sr=22050, target_sr=16000)

    def _analyze_spatial(self, audio: AudioSample) -> SpatialInfo:
        """
        Analyze spatial characteristics of stereo audio.

        For stereo audio, we can detect:
        1. Stereo width: How different are left and right channels?
        2. Movement: Does the sound move from one side to the other?

        Args:
            audio: Stereo AudioSample

        Returns:
            SpatialInfo: Spatial characteristics

        HOW MOVEMENT DETECTION WORKS:
        We compare energy distribution in first and second halves:
        - If left starts strong and right ends strong: left_to_right
        - If right starts strong and left ends strong: right_to_left
        - Otherwise: static

        This is a simple heuristic that works for panning sounds.
        """
        # Get left and right channels
        left = audio.audio_data[0]
        right = audio.audio_data[1]

        # =====================================================================
        # STEREO WIDTH
        # =====================================================================
        # Correlation measures how similar two signals are
        # Correlation of 1.0 = identical (mono)
        # Correlation of 0.0 = unrelated (wide stereo)
        correlation = np.corrcoef(left, right)[0, 1]

        # Convert to width (1 - correlation)
        # Clamp to [0, 1] range
        stereo_width = 1.0 - abs(correlation)

        # =====================================================================
        # MOVEMENT DETECTION
        # =====================================================================
        # Compute energy (RMS) envelope for each channel
        left_env = librosa.feature.rms(y=left)[0]
        right_env = librosa.feature.rms(y=right)[0]

        # Split into first and second halves
        half = len(left_env) // 2
        left_first = np.mean(left_env[:half])
        left_second = np.mean(left_env[half:])
        right_first = np.mean(right_env[:half])
        right_second = np.mean(right_env[half:])

        # Detect movement direction
        is_moving = False
        direction = "static"

        # Left-to-right: Left strong first, right strong second
        if (left_first > right_first * 1.5) and (right_second > left_second * 1.5):
            is_moving = True
            direction = "left_to_right"
        # Right-to-left: Right strong first, left strong second
        elif (right_first > left_first * 1.5) and (left_second > right_second * 1.5):
            is_moving = True
            direction = "right_to_left"

        return SpatialInfo(
            is_moving=is_moving,
            direction=direction,
            stereo_width=float(stereo_width)
        )

    def _extract_characteristics(
        self,
        class_name: str,
        spatial_info: Optional[SpatialInfo]
    ) -> List[str]:
        """
        Extract descriptive characteristics.

        Build a list of characteristics that describe the sound
        beyond just the class name.

        Args:
            class_name: Primary class name
            spatial_info: Spatial characteristics (if stereo)

        Returns:
            List[str]: Descriptive characteristics

        EXAMPLE:
        For a car sound with movement:
        ["moving left to right", "wide stereo"]
        """
        characteristics = []

        # Add movement if detected
        if spatial_info and spatial_info.is_moving:
            # Convert "left_to_right" to "left to right"
            characteristics.append(
                f"moving {spatial_info.direction.replace('_', ' ')}"
            )

        # Add stereo width description
        if spatial_info:
            if spatial_info.stereo_width > 0.7:
                characteristics.append("wide stereo")
            elif spatial_info.stereo_width < 0.3:
                characteristics.append("narrow stereo")

        return characteristics
