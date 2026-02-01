"""
Structured logging utilities for the Audio Sample Analysis Application.

Provides JSON-formatted logging for production environments and
human-readable logging for development.
"""

import json
import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional


class JSONFormatter(logging.Formatter):
    """
    Custom formatter that outputs log records as JSON.

    This is useful for:
    - Log aggregation services (ELK, Splunk, CloudWatch)
    - Structured log analysis
    - Machine-parseable log files
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON string."""
        log_obj: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        # Add extra fields if present
        if hasattr(record, "extra") and record.extra:
            log_obj["extra"] = record.extra

        return json.dumps(log_obj, default=str)


class ColoredFormatter(logging.Formatter):
    """
    Formatter that adds colors to log levels for terminal output.

    Makes logs easier to read during development.
    """

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None):
        super().__init__(fmt, datefmt)

    def format(self, record: logging.LogRecord) -> str:
        """Format with color codes."""
        color = self.COLORS.get(record.levelname, "")
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(
    level: str = "INFO",
    log_format: str = "json",
    log_file: Optional[str] = None,
    max_bytes: int = 10485760,
    backup_count: int = 5,
    console_enabled: bool = True,
    colored: bool = True,
) -> None:
    """
    Configure application-wide logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Output format ("json" or "text")
        log_file: Optional file path for log output
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup files to keep
        console_enabled: Whether to log to console
        colored: Whether to use colored output (console only, text format only)
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    root_logger.handlers = []

    # Create formatters
    if log_format == "json":
        formatter = JSONFormatter()
    else:
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"
        if colored and console_enabled:
            formatter = ColoredFormatter(fmt, datefmt)
        else:
            formatter = logging.Formatter(fmt, datefmt)

    # Console handler
    if console_enabled:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler with rotation
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        # Always use JSON for file logging (easier to parse)
        file_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.

    Args:
        name: Logger name (typically module or class name)

    Returns:
        logging.Logger: Configured logger instance
    """
    return logging.getLogger(name)


class LoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that adds extra context to all log messages.

    Useful for adding request IDs, user IDs, or other context
    that should appear in every log message.
    """

    def process(
        self, msg: str, kwargs: Dict[str, Any]
    ) -> tuple[str, Dict[str, Any]]:
        """Add extra context to log message."""
        extra = kwargs.get("extra", {})
        extra.update(self.extra)
        kwargs["extra"] = extra
        return msg, kwargs


def create_logger_with_context(
    name: str, context: Dict[str, Any]
) -> LoggerAdapter:
    """
    Create a logger with persistent context.

    Args:
        name: Logger name
        context: Dictionary of context to add to all logs

    Returns:
        LoggerAdapter: Logger that includes context in all messages

    Example:
        logger = create_logger_with_context(
            "api",
            {"request_id": "abc123", "user_id": "user456"}
        )
        logger.info("Processing request")
        # Logs: {"message": "Processing request", "request_id": "abc123", ...}
    """
    base_logger = get_logger(name)
    return LoggerAdapter(base_logger, context)
