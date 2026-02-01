"""
Speech recognition analyzers.
"""

from src.analyzers.speech.whisper_speech import (
    WhisperSpeechAnalyzer,
    create_whisper_speech_analyzer
)

__all__ = [
    "WhisperSpeechAnalyzer",
    "create_whisper_speech_analyzer"
]
