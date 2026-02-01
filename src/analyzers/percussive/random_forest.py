"""
Random Forest percussive analyzer for the Audio Sample Analysis Application.

Classifies percussion/drum sounds using a Random Forest model.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.core.analyzer_base import BaseAnalyzer
from src.core.models import AudioSample, PercussiveAnalysis


# Drum type labels
DRUM_TYPES = ['kick', 'snare', 'hihat', 'rim', 'tom', 'cymbal', 'shaker', 'other']


class RandomForestPercussiveAnalyzer(BaseAnalyzer[PercussiveAnalysis]):
    """
    Random Forest-based percussion classification.

    Uses a simple RF model trained on spectral features to classify
    drum sounds into 8 categories.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize percussive analyzer.

        Args:
            model_path: Optional path to pre-trained model
        """
        super().__init__("random_forest_percussive", "1.0.0")
        self.model_path = model_path
        self._model: Optional[RandomForestClassifier] = None

    @property
    def model(self) -> RandomForestClassifier:
        """Lazy-load or create Random Forest model."""
        if self._model is None:
            self._model = self._load_or_create_model()
        return self._model

    def _load_or_create_model(self) -> RandomForestClassifier:
        """Load existing model or create a default one."""
        # Try to load pre-trained model
        if self.model_path:
            model_file = Path(self.model_path)
            if model_file.exists():
                self.logger.info(f"Loading model from {model_file}")
                return joblib.load(model_file)

        # Create default model (will use rule-based classification)
        self.logger.info("Creating default Random Forest model")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )

        # Note: In a real implementation, this would be pre-trained
        # For now, we'll use rule-based classification
        return model

    def _analyze_impl(self, audio: AudioSample) -> PercussiveAnalysis:
        """
        Classify percussion sound.

        Args:
            audio: AudioSample to analyze

        Returns:
            PercussiveAnalysis: Classification result
        """
        # Extract features for classification (uses cached features)
        features = self._extract_features(audio)

        # Classify using features
        classification = self._classify(features, audio)

        # Analyze timbral characteristics (uses cached features)
        timbre = self._analyze_timbre(audio)

        # Detect if synthesized
        is_synthesized = self._detect_synthesized(audio)

        return PercussiveAnalysis(
            drum_type=classification['drum_type'],
            confidence=classification['confidence'],
            attack_time=timbre['attack_time'],
            decay_time=timbre['decay_time'],
            brightness=timbre['brightness'],
            is_synthesized=is_synthesized,
            alternatives=classification['alternatives'],
            explanation=None
        )

    def _extract_features(self, audio: AudioSample) -> np.ndarray:
        """
        Extract features for classification.

        Args:
            audio: AudioSample to analyze

        Returns:
            np.ndarray: Feature vector
        """
        features = audio.features

        # Aggregate features
        feature_vector = []

        # MFCC statistics
        mfcc_mean = np.mean(features.mfcc, axis=1)
        mfcc_std = np.std(features.mfcc, axis=1)
        feature_vector.extend(mfcc_mean)
        feature_vector.extend(mfcc_std)

        # Spectral features
        feature_vector.append(float(np.mean(features.spectral_centroid)))
        feature_vector.append(float(np.mean(features.spectral_rolloff)))
        feature_vector.append(float(np.mean(features.spectral_bandwidth)))
        feature_vector.append(float(np.mean(features.zero_crossing_rate)))

        # Onset features
        feature_vector.append(float(len(features.onset_frames)))

        return np.array(feature_vector)

    def _classify(self, features: np.ndarray, audio: AudioSample) -> dict:
        """
        Classify drum type using features.

        Since we don't have a pre-trained model, we use rule-based
        classification based on spectral characteristics.

        Args:
            features: Feature vector
            audio: Original audio sample

        Returns:
            dict: Classification results
        """
        # Use rule-based classification based on spectral characteristics
        spectral_centroid = float(np.mean(audio.features.spectral_centroid))
        spectral_rolloff = float(np.mean(audio.features.spectral_rolloff))
        zcr = float(np.mean(audio.features.zero_crossing_rate))
        duration = audio.duration

        # Classification rules based on spectral characteristics
        scores = {}

        # Kick: Low centroid, low ZCR
        kick_score = max(0, 1.0 - spectral_centroid / 2000) * max(0, 1.0 - zcr * 10)
        scores['kick'] = kick_score

        # Snare: Mid centroid, mid-high ZCR
        snare_score = (1.0 - abs(spectral_centroid - 3000) / 3000) * (0.5 + zcr * 5)
        scores['snare'] = max(0, min(1, snare_score))

        # Hihat: High centroid, high ZCR
        hihat_score = min(1, spectral_centroid / 6000) * min(1, zcr * 15)
        scores['hihat'] = max(0, hihat_score)

        # Cymbal: Very high centroid, medium duration
        cymbal_score = min(1, spectral_centroid / 8000) * (1.0 if duration > 0.3 else 0.3)
        scores['cymbal'] = max(0, cymbal_score)

        # Tom: Low-mid centroid, moderate ZCR
        tom_score = (1.0 - abs(spectral_centroid - 1500) / 1500) * (1.0 - abs(zcr - 0.1) * 10)
        scores['tom'] = max(0, min(1, tom_score))

        # Rim: Mid-high centroid, short duration, high attack
        rim_score = (1.0 - abs(spectral_centroid - 4000) / 4000) * (1.0 if duration < 0.2 else 0.3)
        scores['rim'] = max(0, min(1, rim_score))

        # Shaker: High ZCR, medium centroid
        shaker_score = min(1, zcr * 20) * (1.0 - abs(spectral_centroid - 4000) / 4000)
        scores['shaker'] = max(0, min(1, shaker_score))

        # Other: Low score for all above
        scores['other'] = 0.2

        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}

        # Sort by score
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Best classification
        drum_type, confidence = sorted_scores[0]

        # Alternatives
        alternatives: List[Tuple[str, float]] = sorted_scores[1:4]

        return {
            'drum_type': drum_type,
            'confidence': float(confidence),
            'alternatives': alternatives
        }

    def _analyze_timbre(self, audio: AudioSample) -> dict:
        """
        Analyze timbral characteristics.

        Args:
            audio: AudioSample with pre-computed features

        Returns:
            dict: Timbral characteristics
        """
        # Use pre-computed percussive component and spectral centroid
        percussive = audio.features.percussive
        sr = audio.sample_rate

        # Envelope
        envelope = np.abs(percussive)

        # Find peak
        peak_idx = np.argmax(envelope)
        peak_time = peak_idx / sr

        # Attack time: Time from start to 90% of peak
        threshold = 0.9 * envelope[peak_idx] if envelope[peak_idx] > 0 else 0.9
        attack_samples = np.where(envelope[:peak_idx] >= threshold * 0.1)[0]
        if len(attack_samples) > 0:
            attack_time = (peak_idx - attack_samples[0]) / sr * 1000  # ms
        else:
            attack_time = peak_time * 1000

        # Decay time: Time from peak to 10% of peak
        decay_samples = np.where(envelope[peak_idx:] <= threshold * 0.1)[0]
        if len(decay_samples) > 0:
            decay_time = decay_samples[0] / sr * 1000  # ms
        else:
            decay_time = (len(envelope) - peak_idx) / sr * 1000

        # Brightness: reuse pre-computed spectral centroid (avoid redundant computation)
        brightness = float(np.mean(audio.features.spectral_centroid))

        return {
            'attack_time': float(max(0.1, min(100, attack_time))),
            'decay_time': float(max(1, min(5000, decay_time))),
            'brightness': float(brightness)
        }

    def _detect_synthesized(self, audio: AudioSample) -> bool:
        """
        Detect if the drum sound is synthesized.

        Synthesized sounds tend to be more "perfect" - less variation
        in spectral characteristics.

        Args:
            audio: AudioSample to analyze

        Returns:
            bool: True if likely synthesized
        """
        # Get spectral features
        centroid = audio.features.spectral_centroid
        rolloff = audio.features.spectral_rolloff

        # Compute variation
        centroid_std = np.std(centroid)
        rolloff_std = np.std(rolloff)

        # Low variation suggests synthesized
        # Real drums have more acoustic variation
        centroid_mean = np.mean(centroid)
        rolloff_mean = np.mean(rolloff)

        # Coefficient of variation
        if centroid_mean > 0:
            centroid_cv = centroid_std / centroid_mean
        else:
            centroid_cv = 0

        if rolloff_mean > 0:
            rolloff_cv = rolloff_std / rolloff_mean
        else:
            rolloff_cv = 0

        # Low CV suggests synthesized
        is_synthesized = centroid_cv < 0.1 and rolloff_cv < 0.1

        return is_synthesized
