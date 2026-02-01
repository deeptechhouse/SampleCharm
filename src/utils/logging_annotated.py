"""
Structured logging utilities for the Audio Sample Analysis Application.

=============================================================================
ANNOTATED VERSION - Extensive comments for educational purposes
=============================================================================

This module provides logging functionality with two main formats:
1. JSON logging for production (machine-readable)
2. Colored text logging for development (human-readable)

OVERVIEW FOR JUNIOR DEVELOPERS:
-------------------------------
Logging is ESSENTIAL for any production application. It helps you:
- Debug issues after they happen ("What was the state when this crashed?")
- Monitor application health ("How many errors per hour?")
- Audit actions ("Who did what and when?")
- Understand behavior ("Why did this take so long?")

PYTHON'S LOGGING HIERARCHY:
                    Root Logger (logging.getLogger())
                            |
            +---------------+---------------+
            |               |               |
    src.core.engine   src.analyzers   src.utils
            |
    +-------+-------+
    |       |       |
  loader  models  features

Messages propagate UP the hierarchy. Setting root logger level affects all.

LOG LEVELS (from least to most severe):
- DEBUG: Detailed info for debugging (verbose)
- INFO: General operational messages
- WARNING: Something unexpected but not an error
- ERROR: Something failed but app continues
- CRITICAL: App is about to crash

STRUCTURED VS TEXT LOGGING:
- Text: "2024-01-24 10:30:15 | INFO | Loaded file test.wav"
- JSON: {"timestamp": "2024-01-24T10:30:15Z", "level": "INFO", "message": "Loaded file", "file": "test.wav"}

JSON is better for production because log aggregation tools (like Splunk,
ELK Stack, or CloudWatch) can parse and query structured logs efficiently.
"""

# =============================================================================
# IMPORTS
# =============================================================================

import json                    # For JSON serialization
import logging                 # Python's built-in logging framework
import sys                     # For stdout access
from datetime import datetime  # For timestamps
from logging.handlers import RotatingFileHandler  # Auto-rotate log files
from pathlib import Path       # Modern path handling
from typing import Any, Dict, Optional  # Type hints


# =============================================================================
# JSON FORMATTER
# =============================================================================

class JSONFormatter(logging.Formatter):
    """
    Custom formatter that outputs log records as JSON.

    This is useful for:
    - Log aggregation services (ELK, Splunk, CloudWatch)
    - Structured log analysis
    - Machine-parseable log files

    WHAT IS A FORMATTER?
    A Formatter converts a LogRecord (Python's internal log representation)
    into a string that gets written to the output (console, file, etc.).

    Python's default formatter produces text like:
        "INFO:module:message"

    Our JSON formatter produces:
        {"level": "INFO", "module": "module", "message": "message", ...}

    WHY INHERIT FROM logging.Formatter?
    We need to override the format() method to customize output.
    The parent class handles things like exception formatting.

    EXAMPLE OUTPUT:
    {
        "timestamp": "2024-01-24T10:30:15.123456Z",
        "level": "INFO",
        "logger": "src.core.loader",
        "message": "Loaded audio file: test.wav",
        "module": "loader",
        "function": "load",
        "line": 45
    }
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON string.

        Args:
            record: LogRecord object containing all log information
                   (level, message, exception info, etc.)

        Returns:
            str: JSON-formatted string representation

        WHAT'S IN A LogRecord?
        - record.levelname: "DEBUG", "INFO", etc.
        - record.name: Logger name (e.g., "src.core.loader")
        - record.getMessage(): The log message
        - record.module: Module name
        - record.funcName: Function name
        - record.lineno: Line number in source
        - record.exc_info: Exception tuple if logging an exception
        """
        # Build the base log object with standard fields
        log_obj: Dict[str, Any] = {
            # ISO format timestamp with 'Z' to indicate UTC
            # Example: "2024-01-24T10:30:15.123456Z"
            "timestamp": datetime.utcnow().isoformat() + "Z",

            # Log level as string
            "level": record.levelname,

            # Logger name (shows the module hierarchy)
            "logger": record.name,

            # The actual log message (may have been formatted with %)
            "message": record.getMessage(),

            # Source code location - useful for debugging
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # If this log includes exception info, add formatted traceback
        # This happens when you call logger.exception() or pass exc_info=True
        if record.exc_info:
            # formatException() returns the full traceback as a string
            log_obj["exception"] = self.formatException(record.exc_info)

        # If the log call included extra fields, add them
        # Example: logger.info("msg", extra={"user_id": "123"})
        if hasattr(record, "extra") and record.extra:
            log_obj["extra"] = record.extra

        # Convert to JSON string
        # default=str handles any non-serializable objects (like Path)
        return json.dumps(log_obj, default=str)


# =============================================================================
# COLORED TEXT FORMATTER
# =============================================================================

class ColoredFormatter(logging.Formatter):
    """
    Formatter that adds colors to log levels for terminal output.

    Makes logs easier to read during development by color-coding
    different severity levels.

    HOW TERMINAL COLORS WORK:
    Terminal colors use "ANSI escape codes" - special character sequences
    that terminals interpret as formatting commands.

    Format: \033[XXm where XX is a color code
    - \033 is the escape character
    - [ starts the sequence
    - XX is the color code (31=red, 32=green, etc.)
    - m ends the sequence

    To reset to default: \033[0m

    EXAMPLE:
    "\033[31mThis is red\033[0m and this is normal"

    WHY COLORED LOGS?
    In a sea of log output, colors help you instantly spot:
    - Errors (red) - Something's wrong!
    - Warnings (yellow) - Pay attention
    - Info (green) - Normal operation
    - Debug (cyan) - Detailed trace info

    NOTE: Colors don't work in all terminals. The logging output
    may look garbled if the terminal doesn't support ANSI codes.
    """

    # ANSI color codes for each log level
    # Format: level_name -> escape_sequence
    COLORS = {
        "DEBUG": "\033[36m",     # Cyan - stands out but not alarming
        "INFO": "\033[32m",      # Green - everything is OK
        "WARNING": "\033[33m",   # Yellow - pay attention
        "ERROR": "\033[31m",     # Red - something's wrong
        "CRITICAL": "\033[35m",  # Magenta - very wrong
    }

    # Reset code to return to default terminal color
    RESET = "\033[0m"

    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None):
        """
        Initialize the colored formatter.

        Args:
            fmt: Log message format string (uses % formatting)
            datefmt: Date/time format string

        EXAMPLE FORMAT STRING:
        "%(asctime)s | %(levelname)s | %(message)s"

        Available placeholders:
        - %(asctime)s: Timestamp
        - %(levelname)s: DEBUG/INFO/etc.
        - %(name)s: Logger name
        - %(message)s: The log message
        - %(module)s: Module name
        - %(funcName)s: Function name
        - %(lineno)d: Line number
        """
        super().__init__(fmt, datefmt)

    def format(self, record: logging.LogRecord) -> str:
        """
        Format with color codes.

        Process:
        1. Get color for this log level
        2. Wrap levelname in color codes
        3. Call parent format() to build final string

        Args:
            record: LogRecord to format

        Returns:
            str: Colored formatted string
        """
        # Get color for this level, default to empty string (no color)
        color = self.COLORS.get(record.levelname, "")

        # Wrap the level name in color codes
        # Before: "INFO"
        # After:  "\033[32mINFO\033[0m" (green INFO then reset)
        record.levelname = f"{color}{record.levelname}{self.RESET}"

        # Let parent class handle the rest of the formatting
        return super().format(record)


# =============================================================================
# LOGGING SETUP FUNCTION
# =============================================================================

def setup_logging(
    level: str = "INFO",
    log_format: str = "json",
    log_file: Optional[str] = None,
    max_bytes: int = 10485760,   # 10 MB
    backup_count: int = 5,
    console_enabled: bool = True,
    colored: bool = True,
) -> None:
    """
    Configure application-wide logging.

    This function sets up the root logger, which affects ALL loggers
    in the application. Call this ONCE at application startup.

    Args:
        level: Minimum log level to capture
               "DEBUG" - Everything (very verbose)
               "INFO" - Normal operation messages
               "WARNING" - Only warnings and above
               "ERROR" - Only errors and critical

        log_format: Output format
                   "json" - Machine-readable JSON (for production)
                   "text" - Human-readable text (for development)

        log_file: Optional file path for log output
                  If provided, logs are written to this file
                  Files are rotated when they reach max_bytes

        max_bytes: Maximum log file size before rotation (default: 10 MB)
                   When exceeded, current log is renamed (log.1, log.2, etc.)
                   and a new log file is created

        backup_count: Number of rotated files to keep (default: 5)
                      Older files are deleted automatically

        console_enabled: Whether to also log to console (stdout)

        colored: Whether to use colors in console output
                 Only applies to text format

    WHAT IS LOG ROTATION?
    Without rotation, log files would grow forever and fill your disk.
    RotatingFileHandler automatically:
    1. Monitors file size
    2. When max_bytes exceeded: renames current to .1
    3. Shifts existing backups (.1 -> .2, .2 -> .3, etc.)
    4. Deletes oldest if > backup_count
    5. Creates fresh log file

    EXAMPLE FILE ROTATION:
    Before: app.log (11 MB), app.log.1, app.log.2
    After:  app.log (empty), app.log.1 (was app.log), app.log.2 (was .1), app.log.3 (was .2)

    EXAMPLE USAGE:
        # Development setup
        setup_logging(
            level="DEBUG",
            log_format="text",
            colored=True,
            console_enabled=True
        )

        # Production setup
        setup_logging(
            level="INFO",
            log_format="json",
            log_file="logs/app.log",
            console_enabled=False
        )
    """
    # Get the root logger - this is the ancestor of ALL loggers
    root_logger = logging.getLogger()

    # Set the minimum level to capture
    # getattr(logging, "INFO") returns logging.INFO (integer 20)
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove any existing handlers to prevent duplicate logs
    # This is important if setup_logging is called multiple times
    root_logger.handlers = []

    # Create the appropriate formatter based on configuration
    if log_format == "json":
        # Machine-readable JSON format
        formatter = JSONFormatter()
    else:
        # Human-readable text format
        # Define the format string with placeholders
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        # %-8s means left-align in 8-character field (for alignment)

        datefmt = "%Y-%m-%d %H:%M:%S"  # ISO-ish date format

        if colored and console_enabled:
            # Use colored formatter for terminal
            formatter = ColoredFormatter(fmt, datefmt)
        else:
            # Plain text formatter
            formatter = logging.Formatter(fmt, datefmt)

    # Set up console logging (stdout)
    if console_enabled:
        # StreamHandler sends logs to a stream (file-like object)
        # sys.stdout is the console output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)

        # Add handler to root logger
        root_logger.addHandler(console_handler)

    # Set up file logging with rotation
    if log_file:
        # Ensure the log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Create rotating file handler
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,      # Rotate when file exceeds this size
            backupCount=backup_count,  # Keep this many backup files
        )

        # Always use JSON for file logging
        # Reason: File logs are usually parsed by tools, so structured
        # format is more useful than human-readable text
        file_handler.setFormatter(JSONFormatter())

        root_logger.addHandler(file_handler)


# =============================================================================
# LOGGER FACTORY FUNCTION
# =============================================================================

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.

    This is the standard way to get a logger in your modules.
    The name should typically be the module name.

    Args:
        name: Logger name (typically module or class name)
              Recommended: use __name__ to get the module name

    Returns:
        logging.Logger: Configured logger instance

    LOGGER NAMING CONVENTION:
    Use dot-separated names matching your module structure:
    - "src.core.loader"
    - "src.analyzers.yamnet"
    - "src.utils.config"

    This creates a hierarchy where you can control logging
    at different levels of granularity.

    EXAMPLE USAGE:
        # In src/core/loader.py
        logger = get_logger(__name__)  # Gets "src.core.loader"

        def load(path):
            logger.info(f"Loading {path}")
            try:
                data = read_file(path)
                logger.debug(f"Loaded {len(data)} bytes")
            except Exception as e:
                logger.error(f"Failed to load: {e}")

    WHY USE get_logger() INSTEAD OF logging.getLogger()?
    It's the same thing! This wrapper is provided for consistency
    and potential future enhancements (like adding default context).
    """
    return logging.getLogger(name)


# =============================================================================
# LOGGER ADAPTER FOR CONTEXT
# =============================================================================

class LoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that adds extra context to all log messages.

    Useful for adding request IDs, user IDs, or other context
    that should appear in every log message from a request.

    WHAT IS A LoggerAdapter?
    It wraps a logger and intercepts all log calls to add extra info.
    The underlying logger does the actual logging.

    WHY USE IT?
    Imagine tracking a web request through multiple modules:
    - Without adapter: Each log() call must include request_id
    - With adapter: Set context once, it's added automatically

    EXAMPLE:
        Without adapter:
            logger.info("Loading file", extra={"request_id": rid})
            logger.info("Processing", extra={"request_id": rid})
            logger.info("Done", extra={"request_id": rid})

        With adapter:
            ctx_logger = LoggerAdapter(logger, {"request_id": rid})
            ctx_logger.info("Loading file")  # request_id added automatically
            ctx_logger.info("Processing")
            ctx_logger.info("Done")
    """

    def process(
        self, msg: str, kwargs: Dict[str, Any]
    ) -> tuple[str, Dict[str, Any]]:
        """
        Process a log message to add context.

        This method is called for every log message. We use it to
        merge our stored context (self.extra) with any extra fields
        passed to the log call.

        Args:
            msg: The log message
            kwargs: Keyword arguments passed to log call

        Returns:
            tuple: (message, modified_kwargs)

        HOW IT WORKS:
        1. Get existing 'extra' from kwargs (if any)
        2. Add our stored context (self.extra) to it
        3. Return modified kwargs

        The logging framework will include 'extra' in the LogRecord.
        """
        # Get any extra fields already in kwargs
        extra = kwargs.get("extra", {})

        # Merge our stored context (self.extra comes from __init__)
        extra.update(self.extra)

        # Put merged extra back in kwargs
        kwargs["extra"] = extra

        return msg, kwargs


def create_logger_with_context(
    name: str, context: Dict[str, Any]
) -> LoggerAdapter:
    """
    Create a logger with persistent context.

    This is a convenience function for creating a LoggerAdapter
    with a given context dictionary.

    Args:
        name: Logger name (e.g., "src.core.engine")
        context: Dictionary of context to add to all logs
                 Example: {"request_id": "abc123", "user_id": "user456"}

    Returns:
        LoggerAdapter: Logger that includes context in all messages

    EXAMPLE USAGE:
        # At the start of request handling
        request_id = str(uuid.uuid4())
        logger = create_logger_with_context(
            "api.handler",
            {"request_id": request_id, "user_id": current_user.id}
        )

        # All subsequent logs include this context automatically
        logger.info("Processing request")
        # Output: {"message": "Processing request", "request_id": "abc123", ...}

        logger.info("Calling analyzer")
        # Output: {"message": "Calling analyzer", "request_id": "abc123", ...}

    WHY IS THIS USEFUL?
    When debugging issues in production, you can filter logs by
    request_id to see everything that happened for a specific request.
    This is crucial for tracing issues in concurrent systems.
    """
    # Get the base logger
    base_logger = get_logger(name)

    # Wrap it in adapter with the context
    return LoggerAdapter(base_logger, context)


# =============================================================================
# USAGE EXAMPLES (for educational purposes)
# =============================================================================

"""
EXAMPLE 1: Basic setup and usage

    from src.utils.logging import setup_logging, get_logger

    # Configure logging at app startup
    setup_logging(level="DEBUG", log_format="text", colored=True)

    # In your module
    logger = get_logger(__name__)

    def process_file(path):
        logger.debug(f"Starting to process {path}")
        logger.info(f"Processing file: {path}")
        logger.warning("This file is quite large")
        logger.error("Something went wrong!")

EXAMPLE 2: Production setup with file logging

    setup_logging(
        level="INFO",
        log_format="json",
        log_file="logs/app.log",
        max_bytes=50 * 1024 * 1024,  # 50 MB
        backup_count=10,
        console_enabled=False  # No console output in production
    )

EXAMPLE 3: Logging exceptions

    logger = get_logger(__name__)

    try:
        risky_operation()
    except Exception as e:
        # logger.exception() automatically includes the traceback
        logger.exception("Operation failed")

        # Or with explicit exc_info:
        logger.error("Operation failed", exc_info=True)

EXAMPLE 4: Request tracking with context

    from src.utils.logging import create_logger_with_context
    import uuid

    def handle_request(request):
        # Create request-specific logger
        request_id = str(uuid.uuid4())
        logger = create_logger_with_context(
            "api.handler",
            {
                "request_id": request_id,
                "user_id": request.user.id,
                "endpoint": request.path
            }
        )

        logger.info("Request received")
        # All logs will include request_id, user_id, endpoint

        result = process_request(request)

        logger.info("Request completed")

EXAMPLE 5: Different log levels for different modules

    # In setup code:
    setup_logging(level="INFO")

    # Set specific module to DEBUG for detailed tracing
    logging.getLogger("src.core.loader").setLevel(logging.DEBUG)

    # Set another module to WARNING to reduce noise
    logging.getLogger("src.utils").setLevel(logging.WARNING)
"""
