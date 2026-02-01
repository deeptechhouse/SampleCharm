"""
YAMNet source analyzer for the Audio Sample Analysis Application.

Uses Google's YAMNet model for audio source classification.
"""

import csv
import logging
import os
import ssl
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

import librosa
import numpy as np

from src.core.analyzer_base import BaseAnalyzer
from src.core.models import AudioSample, SourceClassification, SpatialInfo
from src.utils.errors import ModelLoadError


def _setup_ssl_certificates():
    """
    Setup SSL certificates for HTTPS requests.

    Fixes the common macOS issue:
    'SSL: CERTIFICATE_VERIFY_FAILED certificate verify failed'
    """
    # Try to use certifi certificates if available
    try:
        import certifi
        os.environ['SSL_CERT_FILE'] = certifi.where()
        os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
    except ImportError:
        pass

    # Also set the default SSL context to be more permissive if needed
    try:
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
    except AttributeError:
        pass

# YAMNet model URL and class names
YAMNET_MODEL_URL = 'https://tfhub.dev/google/yamnet/1'
YAMNET_CLASSES_URL = 'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'


class YAMNetAnalyzer(BaseAnalyzer[SourceClassification]):
    """
    YAMNet-based source classification.

    Model: https://tfhub.dev/google/yamnet/1
    Classes: 521 AudioSet classes
    Input: 16kHz mono audio
    Output: Class probabilities
    """

    def __init__(self, model_url: str = YAMNET_MODEL_URL):
        """
        Initialize YAMNet analyzer.

        Args:
            model_url: TensorFlow Hub URL for YAMNet model
        """
        super().__init__("yamnet", "1.0.0")
        self.model_url = model_url
        self._model = None
        self._class_names: Optional[List[str]] = None

    @property
    def model(self):
        """Lazy-load YAMNet model."""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    @property
    def class_names(self) -> List[str]:
        """Lazy-load class names."""
        if self._class_names is None:
            self._class_names = self._load_class_names()
        return self._class_names

    def _load_model(self):
        """Load YAMNet model from TensorFlow Hub."""
        try:
            # Setup SSL certificates to avoid certificate verification errors
            _setup_ssl_certificates()

            import tensorflow_hub as hub
            self.logger.info("Loading YAMNet model from TensorFlow Hub...")
            model = hub.load(self.model_url)
            self.logger.info(f"YAMNet model loaded successfully")
            return model
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load YAMNet model: {e}",
                model_name="yamnet"
            )

    def _load_class_names(self) -> List[str]:
        """Load YAMNet class names from AudioSet ontology."""
        try:
            # Setup SSL certificates
            _setup_ssl_certificates()

            # Try to download class map
            class_map_path = Path("models/yamnet_class_map.csv")
            class_map_path.parent.mkdir(parents=True, exist_ok=True)

            if not class_map_path.exists():
                self.logger.info("Downloading YAMNet class map...")
                # Create SSL context that doesn't verify certificates
                context = ssl.create_default_context()
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE

                # Download with the unverified context
                with urllib.request.urlopen(YAMNET_CLASSES_URL, context=context) as response:
                    with open(class_map_path, 'wb') as f:
                        f.write(response.read())

            class_names = []
            with open(class_map_path, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    if len(row) >= 3:
                        class_names.append(row[2])  # display_name column

            self.logger.info(f"Loaded {len(class_names)} class names")
            return class_names

        except Exception as e:
            self.logger.warning(f"Could not load class names: {e}")
            # Return basic fallback class names
            return self._get_fallback_class_names()

    def _get_fallback_class_names(self) -> List[str]:
        """Return fallback class names if download fails."""
        return [
            "Speech", "Music", "Animal", "Vehicle", "Natural sounds",
            "Domestic sounds", "Urban sounds", "Silence", "Other"
        ] + [f"Class_{i}" for i in range(512)]  # Pad to 521

    def _analyze_impl(self, audio: AudioSample) -> SourceClassification:
        """
        Classify audio source using YAMNet.

        Args:
            audio: AudioSample to analyze

        Returns:
            SourceClassification: Classification result
        """
        # Step 1: Use cached 16kHz audio (avoids redundant resampling)
        audio_16k = audio.audio_16k

        # Step 2: Run inference
        scores, embeddings, spectrogram = self.model(audio_16k)

        # Step 3: Get top predictions
        scores_np = scores.numpy()
        mean_scores = np.mean(scores_np, axis=0)
        top_indices = np.argsort(mean_scores)[::-1][:5]

        # Primary prediction
        primary_idx = top_indices[0]
        primary_class = self._get_class_name(primary_idx)
        primary_confidence = float(mean_scores[primary_idx])

        # Alternative predictions
        alternatives: List[Tuple[str, float]] = [
            (self._get_class_name(idx), float(mean_scores[idx]))
            for idx in top_indices[1:5]
        ]

        # Step 4: Analyze spatial characteristics (if stereo)
        spatial_info = None
        if audio.channels == 2:
            spatial_info = self._analyze_spatial(audio)

        # Step 5: Extract characteristics
        characteristics = self._extract_characteristics(primary_class, spatial_info)

        return SourceClassification(
            source_type=primary_class,
            confidence=primary_confidence,
            characteristics=characteristics,
            alternatives=alternatives,
            spatial_info=spatial_info,
            explanation=None
        )

    def _get_class_name(self, index: int) -> str:
        """Get class name by index with bounds checking."""
        if 0 <= index < len(self.class_names):
            return self.class_names[index]
        return f"Unknown_{index}"

    def _resample_to_16k(self, audio: np.ndarray) -> np.ndarray:
        """Resample from 22050 Hz to 16000 Hz for YAMNet."""
        return librosa.resample(audio, orig_sr=22050, target_sr=16000)

    def _analyze_spatial(self, audio: AudioSample) -> SpatialInfo:
        """Analyze spatial characteristics of stereo audio."""
        left = audio.audio_data[0]
        right = audio.audio_data[1]

        # Compute stereo width (correlation)
        correlation = np.corrcoef(left, right)[0, 1]
        stereo_width = 1.0 - abs(correlation)

        # Detect movement using energy envelope
        left_env = librosa.feature.rms(y=left)[0]
        right_env = librosa.feature.rms(y=right)[0]

        # Check if energy shifts from one channel to other
        half = len(left_env) // 2
        left_first = np.mean(left_env[:half])
        left_second = np.mean(left_env[half:])
        right_first = np.mean(right_env[:half])
        right_second = np.mean(right_env[half:])

        # Detect direction
        is_moving = False
        direction = "static"

        if (left_first > right_first * 1.5) and (right_second > left_second * 1.5):
            is_moving = True
            direction = "left_to_right"
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
        """Extract descriptive characteristics."""
        characteristics = []

        # Add spatial characteristics
        if spatial_info and spatial_info.is_moving:
            characteristics.append(
                f"moving {spatial_info.direction.replace('_', ' ')}"
            )

        if spatial_info:
            if spatial_info.stereo_width > 0.7:
                characteristics.append("wide stereo")
            elif spatial_info.stereo_width < 0.3:
                characteristics.append("narrow stereo")

        return characteristics
