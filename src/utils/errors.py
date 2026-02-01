"""
Custom exceptions for the Audio Sample Analysis Application.

This module defines a hierarchy of exceptions for handling various
error conditions throughout the application.
"""

from typing import Optional, Any


class AudioAnalysisError(Exception):
    """Base exception for all audio analysis errors."""

    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(message)
        self.message = message
        self.details = details

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} (Details: {self.details})"
        return self.message


class AudioLoadError(AudioAnalysisError):
    """Raised when audio file cannot be loaded."""

    def __init__(self, message: str, file_path: Optional[str] = None):
        super().__init__(message, details={"file_path": file_path})
        self.file_path = file_path


class UnsupportedFormatError(AudioLoadError):
    """Raised when audio format is not supported."""

    def __init__(self, message: str, format: Optional[str] = None):
        super().__init__(message)
        self.format = format
        self.details = {"format": format}


class FileTooLargeError(AudioLoadError):
    """Raised when audio file exceeds size limit."""

    def __init__(
        self,
        message: str,
        file_size: Optional[int] = None,
        max_size: Optional[int] = None,
    ):
        super().__init__(message)
        self.file_size = file_size
        self.max_size = max_size
        self.details = {"file_size": file_size, "max_size": max_size}


class AnalysisError(AudioAnalysisError):
    """Raised when audio analysis fails."""

    def __init__(
        self,
        message: str,
        analyzer_name: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.analyzer_name = analyzer_name
        self.original_error = original_error
        self.details = {
            "analyzer_name": analyzer_name,
            "original_error": str(original_error) if original_error else None,
        }


class ConfigurationError(AudioAnalysisError):
    """Raised when configuration is invalid or missing."""

    def __init__(self, message: str, config_key: Optional[str] = None):
        super().__init__(message)
        self.config_key = config_key
        self.details = {"config_key": config_key}


class CacheError(AudioAnalysisError):
    """Raised when cache operations fail."""

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        key: Optional[str] = None,
    ):
        super().__init__(message)
        self.operation = operation
        self.key = key
        self.details = {"operation": operation, "key": key}


class FeatureExtractionError(AnalysisError):
    """Raised when feature extraction fails."""

    def __init__(self, message: str, feature_name: Optional[str] = None):
        super().__init__(message, analyzer_name="feature_extractor")
        self.feature_name = feature_name
        self.details["feature_name"] = feature_name


class ModelLoadError(AudioAnalysisError):
    """Raised when ML model cannot be loaded."""

    def __init__(self, message: str, model_name: Optional[str] = None):
        super().__init__(message)
        self.model_name = model_name
        self.details = {"model_name": model_name}


class FeatureDisabledError(AudioAnalysisError):
    """Raised when an LLM feature is toggled off in configuration."""

    def __init__(self, feature_id: str):
        super().__init__(
            f"Feature '{feature_id}' is disabled in configuration.",
            details={"feature_id": feature_id},
        )
        self.feature_id = feature_id


class EntitlementError(AudioAnalysisError):
    """Raised when user is not entitled to use an LLM feature (paywall)."""

    def __init__(self, feature_id: str, user_id: str = "anonymous"):
        super().__init__(
            f"User '{user_id}' is not entitled to use feature '{feature_id}'.",
            details={"feature_id": feature_id, "user_id": user_id},
        )
        self.feature_id = feature_id
        self.user_id = user_id
