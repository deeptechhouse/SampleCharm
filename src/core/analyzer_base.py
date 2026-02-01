"""
Analyzer base interface for the Audio Sample Analysis Application.

Defines the contract for all analyzers using Protocol (structural subtyping).
"""

import logging
import time
from abc import abstractmethod
from typing import Generic, Protocol, TypeVar

from src.core.models import AudioSample
from src.utils.errors import AnalysisError

# Type variable for result types
T = TypeVar('T')


class Analyzer(Protocol[T]):
    """
    Base protocol for all analyzers.

    All analyzers must implement:
    - analyze(audio) -> T
    - name property
    - version property

    This uses structural subtyping (duck typing with type hints).
    A class doesn't need to explicitly inherit from Analyzer to be
    compatible - it just needs to have the required methods.
    """

    @property
    def name(self) -> str:
        """Analyzer name (e.g., 'yamnet', 'librosa_musical')."""
        ...

    @property
    def version(self) -> str:
        """Analyzer version for result tracking."""
        ...

    def analyze(self, audio: AudioSample) -> T:
        """
        Analyze audio sample and return typed result.

        Args:
            audio: AudioSample to analyze

        Returns:
            T: Analysis result (type depends on analyzer)

        Raises:
            AnalysisError: If analysis fails
        """
        ...


class BaseAnalyzer(Generic[T]):
    """
    Optional base class providing common functionality.

    Subclasses can inherit this for shared logic like logging,
    error handling, and timing.

    Uses Template Method pattern - analyze() provides the template,
    subclasses implement _analyze_impl().
    """

    def __init__(self, name: str, version: str):
        """
        Initialize analyzer with name and version.

        Args:
            name: Unique analyzer name
            version: Version string for tracking
        """
        self._name = name
        self._version = version
        self.logger = logging.getLogger(f"analyzer.{name}")

    @property
    def name(self) -> str:
        """Return analyzer name."""
        return self._name

    @property
    def version(self) -> str:
        """Return analyzer version."""
        return self._version

    def analyze(self, audio: AudioSample) -> T:
        """
        Template method with timing and error handling.

        Subclasses implement _analyze_impl().

        Args:
            audio: AudioSample to analyze

        Returns:
            T: Analysis result

        Raises:
            AnalysisError: If analysis fails
        """
        start_time = time.time()

        try:
            self.logger.debug(f"Starting analysis: {audio.file_path}")

            result = self._analyze_impl(audio)

            elapsed = time.time() - start_time
            self.logger.info(f"Analysis complete in {elapsed:.3f}s")

            return result

        except AnalysisError:
            # Re-raise AnalysisError as-is
            raise

        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            raise AnalysisError(
                f"{self.name} analysis failed: {e}",
                analyzer_name=self.name,
                original_error=e
            ) from e

    @abstractmethod
    def _analyze_impl(self, audio: AudioSample) -> T:
        """
        Subclasses implement actual analysis logic.

        Args:
            audio: AudioSample to analyze

        Returns:
            T: Analysis result
        """
        raise NotImplementedError


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for an analyzer.

    Args:
        name: Analyzer name

    Returns:
        logging.Logger: Configured logger
    """
    return logging.getLogger(f"analyzer.{name}")
