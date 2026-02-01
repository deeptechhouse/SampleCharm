"""
Whisper-based speech recognition analyzer for high-accuracy speech-to-text.

Uses OpenAI Whisper for accurate speech detection and transcription.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional

import numpy as np

from src.core.analyzer_base import BaseAnalyzer
from src.core.models import AudioSample
from src.utils.errors import AnalysisError, ModelLoadError


class WhisperSpeechAnalyzer(BaseAnalyzer[dict]):
    """
    Speech recognition analyzer using OpenAI Whisper.
    
    Provides high-accuracy speech detection and word-level transcription.
    """

    def __init__(
        self,
        model_size: str = "base",
        language: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize Whisper speech analyzer.
        
        Args:
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
                       Smaller = faster, less accurate. Larger = slower, more accurate.
            language: Language code (e.g., 'en', 'es', 'fr'). None = auto-detect.
            device: Device to use ('cpu', 'cuda'). None = auto-detect.
        """
        super().__init__("whisper_speech", "1.0.0")
        
        self.model_size = model_size
        self.language = language
        self.device = device
        self._model = None
        self.logger = logging.getLogger(__name__)
        
    @property
    def model(self):
        """Lazy-load Whisper model."""
        if self._model is None:
            self._model = self._load_model()
        return self._model
    
    def _load_model(self):
        """Load Whisper model."""
        try:
            import whisper
        except ImportError:
            raise ModelLoadError(
                "whisper package required for speech recognition. "
                "Install with: pip install openai-whisper",
                model_name="whisper"
            )
        
        try:
            self.logger.info(f"Loading Whisper model: {self.model_size}")
            model = whisper.load_model(
                self.model_size,
                device=self.device
            )
            self.logger.info("Whisper model loaded successfully")
            return model
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load Whisper model: {e}",
                model_name=f"whisper-{self.model_size}"
            )
    
    def _analyze_impl(self, audio: AudioSample) -> dict:
        """
        Analyze audio for speech and transcribe.

        Args:
            audio: AudioSample to analyze

        Returns:
            Dict with speech detection and transcription results
        """
        try:
            import whisper

            # Use cached 16kHz audio (avoids redundant resampling)
            # AudioSample.audio_16k is already mono, 16kHz, float32
            audio_data = audio.audio_16k
            
            # Run Whisper transcription
            self.logger.info(f"Running Whisper transcription (model: {self.model_size})...")
            result = self.model.transcribe(
                audio_data,
                language=self.language,
                task="transcribe",
                verbose=False
            )
            self.logger.debug(f"Whisper transcription completed: text_length={len(result.get('text', ''))}")
            
            # Extract information
            text = result.get("text", "").strip()
            language_detected = result.get("language", None)
            segments = result.get("segments", [])
            
            # Log raw transcription for debugging
            self.logger.debug(f"Raw Whisper text: '{text[:100]}...' (length: {len(text)})")
            
            # Determine if speech is present
            # Filter out common false positives
            text_lower = text.lower().strip()
            false_positives = [
                "", 
                "thank you for watching!", 
                "thanks for watching!",
                "you",
                "thank you",
                "thank you.",
                "thanks."
            ]
            # More lenient check - if we have substantial text, consider it speech
            contains_speech = len(text) > 0 and text_lower not in false_positives and len(text_lower) > 1
            
            # Log for debugging
            self.logger.debug(f"Whisper transcription: text_length={len(text)}, contains_speech={contains_speech}, language={language_detected}")
            
            # Extract words from segments
            words = []
            if segments:
                for segment in segments:
                    segment_words = segment.get("words", [])
                    for word_info in segment_words:
                        word = word_info.get("word", "").strip()
                        if word:
                            words.append(word)
            
            # If no word-level timestamps, extract words from text
            if not words and text:
                words = text.split()
            
            # Calculate confidence (average of segment confidences if available)
            confidence = 0.0
            if segments:
                confidences = [s.get("no_speech_prob", 0.0) for s in segments]
                if confidences:
                    # Convert no_speech_prob to speech confidence
                    avg_no_speech = sum(confidences) / len(confidences)
                    confidence = 1.0 - avg_no_speech
            else:
                # If we have text, assume some confidence
                confidence = 0.7 if contains_speech else 0.0
            
            return {
                "contains_speech": contains_speech,
                "transcription": text,
                "detected_words": words if contains_speech else [],
                "speech_language": language_detected,
                "confidence": confidence,
                "segments": segments,
                "model_used": f"whisper-{self.model_size}"
            }
            
        except Exception as e:
            self.logger.error(f"Whisper transcription failed: {e}")
            # Return empty result on error (don't break analysis)
            return {
                "contains_speech": False,
                "transcription": "",
                "detected_words": [],
                "speech_language": None,
                "confidence": 0.0,
                "segments": [],
                "model_used": f"whisper-{self.model_size}",
                "error": str(e)
            }
    
    def _prepare_audio(self, audio: AudioSample) -> np.ndarray:
        """
        Prepare audio data for Whisper.
        
        Whisper expects:
        - Mono audio
        - 16kHz sample rate
        - Float32 format
        - Normalized to [-1, 1]
        
        Args:
            audio: AudioSample to prepare
            
        Returns:
            Prepared numpy array
        """
        # Get mono audio
        if audio.channels == 1:
            mono_audio = audio.audio_data
        else:
            mono_audio = np.mean(audio.audio_data, axis=0)
        
        # Resample to 16kHz if needed (Whisper's expected sample rate)
        if audio.sample_rate != 16000:
            import librosa
            mono_audio = librosa.resample(
                mono_audio,
                orig_sr=audio.sample_rate,
                target_sr=16000
            )
        
        # Ensure float32 and normalized
        mono_audio = mono_audio.astype(np.float32)
        
        # Normalize to [-1, 1] if needed
        max_val = np.max(np.abs(mono_audio))
        if max_val > 1.0:
            mono_audio = mono_audio / max_val
        
        return mono_audio


def create_whisper_speech_analyzer(config: dict) -> Optional[WhisperSpeechAnalyzer]:
    """
    Factory function to create Whisper speech analyzer.
    
    Args:
        config: Configuration dict with 'speech' section
        
    Returns:
        WhisperSpeechAnalyzer instance or None if disabled
    """
    speech_config = config.get("speech", {})
    
    if not speech_config.get("enabled", False):
        return None
    
    model_size = speech_config.get("model_size", "base")
    language = speech_config.get("language", None)
    device = speech_config.get("device", None)
    
    try:
        return WhisperSpeechAnalyzer(
            model_size=model_size,
            language=language,
            device=device
        )
    except Exception as e:
        logging.getLogger(__name__).warning(
            f"Failed to create Whisper analyzer: {e}. Speech recognition disabled."
        )
        return None
