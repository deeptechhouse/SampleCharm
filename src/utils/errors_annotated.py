"""
Custom exceptions for the Audio Sample Analysis Application.

=============================================================================
ANNOTATED VERSION - Extensive comments for educational purposes
=============================================================================

This module defines a hierarchy of exceptions for handling various
error conditions throughout the application.

OVERVIEW FOR JUNIOR DEVELOPERS:
-------------------------------
Exceptions are a way to handle errors in Python. Instead of returning error
codes (like -1 or None), we "raise" exceptions when something goes wrong.
This makes error handling cleaner and more explicit.

WHY CUSTOM EXCEPTIONS?
- More descriptive error messages
- Can carry additional context (like file paths, sizes, etc.)
- Allow catching specific error types
- Easier debugging and logging
- Better user experience with clear error explanations

EXCEPTION HIERARCHY IN THIS MODULE:
                    AudioAnalysisError (base)
                            |
            +---------------+---------------+
            |               |               |
    AudioLoadError    AnalysisError   ConfigurationError
            |               |
    +-------+-------+       +
    |       |       |       |
Unsupported FileToo FeatureExtraction
Format      Large    Error

BEST PRACTICES DEMONSTRATED:
1. Inherit from a base exception class
2. Store relevant context as attributes
3. Override __str__ for readable messages
4. Use Optional typing for flexibility
"""

# =============================================================================
# IMPORTS
# =============================================================================

# typing module provides type hints for better code documentation and IDE support
# Optional[X] means the value can be either type X or None
# Any means any type is acceptable
from typing import Optional, Any


# =============================================================================
# BASE EXCEPTION CLASS
# =============================================================================

class AudioAnalysisError(Exception):
    """
    Base exception for all audio analysis errors.

    This is the parent class for ALL custom exceptions in our application.
    By having a single base class, users can catch ALL our exceptions with:

        try:
            result = engine.analyze(file)
        except AudioAnalysisError as e:
            # Catches any of our custom exceptions
            print(f"Analysis failed: {e}")

    ATTRIBUTES:
        message (str): Human-readable error description
        details (Any): Additional context about the error (optional)

    EXAMPLE USAGE:
        raise AudioAnalysisError("Something went wrong", details={"code": 123})

    WHY INHERIT FROM Exception?
    - Exception is Python's built-in base class for errors
    - It provides standard behavior like traceback support
    - It's catchable with try/except blocks
    """

    def __init__(self, message: str, details: Optional[Any] = None):
        """
        Initialize the exception.

        Args:
            message: Human-readable description of what went wrong.
                     Keep this clear and actionable.
            details: Optional dictionary or object with additional context.
                     Useful for logging and debugging.

        IMPLEMENTATION NOTE:
        We call super().__init__(message) to ensure the parent Exception
        class is properly initialized. This is REQUIRED for exceptions
        to work correctly with Python's exception handling.
        """
        # Call parent class constructor - ALWAYS do this first
        super().__init__(message)

        # Store our custom attributes
        self.message = message
        self.details = details

    def __str__(self) -> str:
        """
        Return string representation of the exception.

        This method is called when you do:
        - print(exception)
        - str(exception)
        - f"{exception}"

        Returns:
            str: Formatted error message with optional details

        DESIGN DECISION:
        We include details in the string representation to make
        debugging easier. When you see an error in logs, you'll
        have all the context you need.
        """
        # If we have additional details, include them
        if self.details:
            return f"{self.message} (Details: {self.details})"
        # Otherwise, just return the message
        return self.message


# =============================================================================
# AUDIO LOADING EXCEPTIONS
# =============================================================================

class AudioLoadError(AudioAnalysisError):
    """
    Raised when audio file cannot be loaded.

    This is the base class for all file-loading related errors.
    It adds the file_path attribute so we know which file caused the issue.

    WHEN TO USE:
    - File doesn't exist
    - File is corrupted
    - File contains invalid data
    - Read permissions denied

    EXAMPLE:
        if not file_path.exists():
            raise AudioLoadError(
                f"File not found: {file_path}",
                file_path=str(file_path)
            )

    WHY SEPARATE FROM BASE CLASS?
    File loading is a distinct operation that often fails due to
    external factors (missing files, permissions). Having a specific
    exception type lets callers handle file issues differently from
    analysis issues.
    """

    def __init__(self, message: str, file_path: Optional[str] = None):
        """
        Initialize audio load error.

        Args:
            message: Description of the loading error
            file_path: Path to the file that failed to load (optional)

        IMPLEMENTATION NOTE:
        We pass details as a dict to the parent class, making it
        easy to log structured information.
        """
        # Pass file_path in details dict to parent class
        super().__init__(message, details={"file_path": file_path})

        # Also store as direct attribute for easy access
        # This way callers can do: error.file_path
        self.file_path = file_path


class UnsupportedFormatError(AudioLoadError):
    """
    Raised when audio format is not supported.

    Our application supports specific formats: WAV, AIFF, MP3, FLAC.
    When a user tries to load an unsupported format (like OGG), we
    raise this exception.

    WHEN TO USE:
    - File extension not in supported list
    - File header indicates unsupported codec
    - Cannot find decoder for format

    EXAMPLE:
        if suffix not in SUPPORTED_FORMATS:
            raise UnsupportedFormatError(
                f"Format {suffix} not supported. Use: WAV, AIFF, MP3, FLAC",
                format=suffix
            )

    WHY A SEPARATE CLASS?
    This is a recoverable error - the user just needs to convert
    their file. Having a specific exception lets the UI show
    appropriate guidance.
    """

    def __init__(self, message: str, format: Optional[str] = None):
        """
        Initialize unsupported format error.

        Args:
            message: Description including supported formats
            format: The unsupported format that was attempted
        """
        # Call parent constructor
        super().__init__(message)

        # Store the problematic format
        self.format = format

        # Override details to include format info
        self.details = {"format": format}


class FileTooLargeError(AudioLoadError):
    """
    Raised when audio file exceeds size limit.

    We limit file sizes to prevent:
    - Memory exhaustion
    - Long processing times
    - Denial of service attacks (in web deployments)

    Default limit: 50 MB

    WHEN TO USE:
    - File size > max_file_size configuration
    - Estimated memory usage would exceed limits

    EXAMPLE:
        if file_size > MAX_FILE_SIZE:
            raise FileTooLargeError(
                f"File too large: {file_size / 1024 / 1024:.1f} MB",
                file_size=file_size,
                max_size=MAX_FILE_SIZE
            )

    WHY STORE BOTH SIZES?
    By storing both the actual size and the limit, we can give
    users clear feedback: "Your file is 75MB but max is 50MB"
    """

    def __init__(
        self,
        message: str,
        file_size: Optional[int] = None,
        max_size: Optional[int] = None,
    ):
        """
        Initialize file too large error.

        Args:
            message: Description with size information
            file_size: Actual file size in bytes
            max_size: Maximum allowed size in bytes
        """
        super().__init__(message)

        # Store size information for programmatic access
        self.file_size = file_size
        self.max_size = max_size

        # Include both sizes in details for logging
        self.details = {"file_size": file_size, "max_size": max_size}


# =============================================================================
# ANALYSIS EXCEPTIONS
# =============================================================================

class AnalysisError(AudioAnalysisError):
    """
    Raised when audio analysis fails.

    Analysis can fail for many reasons:
    - Model prediction error
    - Invalid input data
    - Resource exhaustion
    - Unexpected audio characteristics

    This is the base class for analysis-related failures.

    WHEN TO USE:
    - ML model throws an error
    - Analysis produces invalid results
    - Timeout during analysis

    EXAMPLE:
        try:
            result = model.predict(audio)
        except Exception as e:
            raise AnalysisError(
                "YAMNet classification failed",
                analyzer_name="yamnet",
                original_error=e
            ) from e

    KEY ATTRIBUTE: original_error
    We preserve the original exception so debugging can trace
    the actual cause. The 'from e' syntax in Python creates
    an exception chain for full traceback.
    """

    def __init__(
        self,
        message: str,
        analyzer_name: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        """
        Initialize analysis error.

        Args:
            message: Description of the analysis failure
            analyzer_name: Name of the analyzer that failed (e.g., "yamnet")
            original_error: The underlying exception that caused this error

        DESIGN PATTERN: Exception Chaining
        By storing original_error, we maintain the full error chain.
        This is crucial for debugging complex issues where the root
        cause might be several layers deep.
        """
        super().__init__(message)

        # Which analyzer failed?
        self.analyzer_name = analyzer_name

        # What was the original error?
        self.original_error = original_error

        # Build detailed context for logging
        self.details = {
            "analyzer_name": analyzer_name,
            # Convert exception to string for serialization
            "original_error": str(original_error) if original_error else None,
        }


class ConfigurationError(AudioAnalysisError):
    """
    Raised when configuration is invalid or missing.

    Configuration issues prevent the application from starting
    or functioning correctly.

    WHEN TO USE:
    - Required config key missing
    - Config value invalid (wrong type, out of range)
    - Config file not found or unparseable

    EXAMPLE:
        if "openai" not in config and openai_required:
            raise ConfigurationError(
                "OpenAI configuration required but not found",
                config_key="openai"
            )

    WHY FAIL FAST ON CONFIG ERRORS?
    It's better to fail immediately with a clear error than to
    fail later with a confusing error. Configuration should be
    validated at startup.
    """

    def __init__(self, message: str, config_key: Optional[str] = None):
        """
        Initialize configuration error.

        Args:
            message: Description of the configuration problem
            config_key: The specific config key that's problematic
        """
        super().__init__(message)
        self.config_key = config_key
        self.details = {"config_key": config_key}


class CacheError(AudioAnalysisError):
    """
    Raised when cache operations fail.

    Caching is used to speed up repeated analyses. Cache errors
    should be recoverable (just re-run the analysis).

    WHEN TO USE:
    - Cache read/write failure
    - Cache connection lost (Redis)
    - Serialization/deserialization error

    EXAMPLE:
        try:
            cached = cache.get(key)
        except Exception as e:
            raise CacheError(
                "Failed to retrieve from cache",
                operation="get",
                key=key
            ) from e

    DESIGN DECISION: Non-Fatal Errors
    Cache errors should NOT crash the application. We log them
    and continue without caching. This is called "graceful degradation".
    """

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        key: Optional[str] = None,
    ):
        """
        Initialize cache error.

        Args:
            message: Description of the cache problem
            operation: Which operation failed ("get", "set", "delete")
            key: The cache key involved (be careful not to log sensitive data)
        """
        super().__init__(message)
        self.operation = operation
        self.key = key
        self.details = {"operation": operation, "key": key}


class FeatureExtractionError(AnalysisError):
    """
    Raised when feature extraction fails.

    Feature extraction is a crucial step that transforms raw audio
    into numerical features (MFCC, chroma, etc.). If this fails,
    no analysis can proceed.

    WHEN TO USE:
    - librosa feature extraction throws error
    - Invalid audio data (NaN, Inf values)
    - Audio too short for feature extraction

    EXAMPLE:
        try:
            mfcc = librosa.feature.mfcc(y=audio, sr=sr)
        except Exception as e:
            raise FeatureExtractionError(
                "MFCC extraction failed",
                feature_name="mfcc"
            ) from e

    WHY INHERIT FROM AnalysisError?
    Feature extraction is conceptually part of analysis. By inheriting
    from AnalysisError, catching AnalysisError will also catch this.
    """

    def __init__(self, message: str, feature_name: Optional[str] = None):
        """
        Initialize feature extraction error.

        Args:
            message: Description of the extraction failure
            feature_name: Which feature failed to extract (e.g., "mfcc", "chroma")
        """
        # Pass "feature_extractor" as analyzer_name to parent
        super().__init__(message, analyzer_name="feature_extractor")

        self.feature_name = feature_name

        # Add feature_name to existing details
        self.details["feature_name"] = feature_name


class ModelLoadError(AudioAnalysisError):
    """
    Raised when ML model cannot be loaded.

    ML models (YAMNet, Random Forest) must be loaded before analysis.
    Loading can fail due to:
    - Missing model files
    - Incompatible model version
    - Insufficient memory
    - Network error (TensorFlow Hub)

    WHEN TO USE:
    - TensorFlow Hub download fails
    - Model file not found
    - Model deserialization error

    EXAMPLE:
        try:
            model = hub.load(MODEL_URL)
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load YAMNet from {MODEL_URL}",
                model_name="yamnet"
            ) from e

    RECOVERY STRATEGY:
    Model load errors are often transient (network issues).
    Consider implementing retry logic with exponential backoff.
    """

    def __init__(self, message: str, model_name: Optional[str] = None):
        """
        Initialize model load error.

        Args:
            message: Description of the loading failure
            model_name: Name of the model that failed to load
        """
        super().__init__(message)
        self.model_name = model_name
        self.details = {"model_name": model_name}


# =============================================================================
# USAGE EXAMPLES (for educational purposes)
# =============================================================================

"""
EXAMPLE 1: Basic exception handling

    from src.utils.errors import AudioLoadError, UnsupportedFormatError

    try:
        audio = loader.load("test.ogg")
    except UnsupportedFormatError as e:
        print(f"Format not supported: {e.format}")
        print("Please convert to WAV, AIFF, MP3, or FLAC")
    except AudioLoadError as e:
        print(f"Could not load file: {e.file_path}")
        print(f"Error: {e.message}")

EXAMPLE 2: Catching all our exceptions

    from src.utils.errors import AudioAnalysisError

    try:
        result = engine.analyze(file_path)
    except AudioAnalysisError as e:
        # This catches ALL our custom exceptions
        logger.error(f"Analysis failed: {e}", extra=e.details)
        return None

EXAMPLE 3: Re-raising with context

    from src.utils.errors import AnalysisError

    try:
        predictions = model(audio)
    except Exception as e:
        # Add context and re-raise
        raise AnalysisError(
            f"Model inference failed: {e}",
            analyzer_name="yamnet",
            original_error=e
        ) from e  # 'from e' preserves the traceback chain

EXAMPLE 4: Graceful degradation with cache errors

    from src.utils.errors import CacheError

    try:
        cached_result = cache.get(audio_hash)
        if cached_result:
            return cached_result
    except CacheError:
        # Log but don't fail - just continue without cache
        logger.warning("Cache unavailable, computing fresh result")

    # Continue with analysis even if cache failed
    result = analyze_audio(audio)
"""
