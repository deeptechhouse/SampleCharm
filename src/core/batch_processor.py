"""
Batch processor for analyzing multiple audio files.

Follows SOLID principles:
- Single Responsibility: Only handles batch orchestration
- Open/Closed: Extends functionality without modifying existing engine
- Dependency Inversion: Depends on engine abstraction, not implementation
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from src.core.models import AnalysisResult


@dataclass
class BatchResult:
    """Result of a batch processing operation."""
    successful: Dict[Path, AnalysisResult] = field(default_factory=dict)
    failed: Dict[Path, str] = field(default_factory=dict)
    total_files: int = 0
    total_time: float = 0.0

    @property
    def success_count(self) -> int:
        """Number of successfully processed files."""
        return len(self.successful)

    @property
    def failure_count(self) -> int:
        """Number of failed files."""
        return len(self.failed)

    @property
    def success_rate(self) -> float:
        """Success rate as percentage."""
        if self.total_files == 0:
            return 0.0
        return (self.success_count / self.total_files) * 100


class BatchProcessor:
    """
    Processes multiple audio files using an analysis engine.

    This class follows the Single Responsibility Principle by focusing
    solely on batch orchestration, delegating actual analysis to the engine.
    """

    # Supported audio extensions
    AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a', '.aiff', '.aif'}

    def __init__(
        self,
        engine,
        max_workers: int = 4,
        progress_callback: Optional[Callable[[int, int, Path], None]] = None
    ):
        """
        Initialize batch processor.

        Args:
            engine: Analysis engine instance (dependency injection)
            max_workers: Maximum parallel workers for processing
            progress_callback: Optional callback(current, total, file_path) for progress updates
        """
        self.engine = engine
        self.max_workers = max_workers
        self.progress_callback = progress_callback
        self.logger = logging.getLogger("batch_processor")

    def process(
        self,
        inputs: Union[Path, List[Path]],
        recursive: bool = False
    ) -> BatchResult:
        """
        Process one or more audio files or directories.

        Args:
            inputs: Single path or list of paths (files or directories)
            recursive: If True, search directories recursively

        Returns:
            BatchResult containing all results and any errors
        """
        import time
        start_time = time.time()

        # Collect all audio files
        files = self._collect_files(inputs, recursive)

        if not files:
            self.logger.warning("No audio files found to process")
            return BatchResult(total_files=0, total_time=0.0)

        self.logger.info(f"Processing {len(files)} audio files")

        # Process files
        result = self._process_files(files)
        result.total_time = time.time() - start_time

        self.logger.info(
            f"Batch complete: {result.success_count}/{result.total_files} succeeded "
            f"in {result.total_time:.2f}s"
        )

        return result

    def _collect_files(
        self,
        inputs: Union[Path, List[Path]],
        recursive: bool
    ) -> List[Path]:
        """Collect all audio files from inputs."""
        if isinstance(inputs, Path):
            inputs = [inputs]

        files = []
        for path in inputs:
            path = Path(path)
            if path.is_file():
                if self._is_audio_file(path):
                    files.append(path)
                else:
                    self.logger.warning(f"Skipping non-audio file: {path}")
            elif path.is_dir():
                files.extend(self._scan_directory(path, recursive))
            else:
                self.logger.warning(f"Path not found: {path}")

        return sorted(set(files))  # Remove duplicates and sort

    def _scan_directory(self, directory: Path, recursive: bool) -> List[Path]:
        """Scan directory for audio files."""
        pattern = "**/*" if recursive else "*"
        files = []

        for path in directory.glob(pattern):
            if path.is_file() and self._is_audio_file(path):
                files.append(path)

        return files

    def _is_audio_file(self, path: Path) -> bool:
        """Check if path is a supported audio file."""
        return path.suffix.lower() in self.AUDIO_EXTENSIONS

    def _process_files(self, files: List[Path]) -> BatchResult:
        """Process list of files, optionally in parallel."""
        result = BatchResult(total_files=len(files))
        processed = 0

        # Use single-threaded for now to avoid TensorFlow threading issues
        # Can be extended to use ThreadPoolExecutor for I/O-bound operations
        for file_path in files:
            processed += 1

            if self.progress_callback:
                self.progress_callback(processed, len(files), file_path)

            try:
                analysis = self.engine.analyze(file_path)
                result.successful[file_path] = analysis
                self.logger.debug(f"Successfully processed: {file_path}")
            except Exception as e:
                error_msg = str(e)
                result.failed[file_path] = error_msg
                self.logger.error(f"Failed to process {file_path}: {error_msg}")

        return result
