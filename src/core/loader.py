"""
Audio loader for the Audio Sample Analysis Application.

Loads and validates audio files from various formats and quality levels.
"""

import asyncio
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

import librosa
import numpy as np
import soundfile as sf

from src.core.models import AudioSample
from src.utils.errors import AudioLoadError, FileTooLargeError, UnsupportedFormatError


# Constants
SUPPORTED_FORMATS: Dict[str, str] = {
    '.wav': 'soundfile',
    '.aif': 'soundfile',
    '.aiff': 'soundfile',
    '.mp3': 'audioread',
    '.flac': 'soundfile'
}

TARGET_SAMPLE_RATE: int = 22050  # Hz
MAX_FILE_SIZE: int = 524288000  # 500 MB (supports large audio files)
MAX_DURATION: float = 30.0  # seconds

logger = logging.getLogger(__name__)


class AudioLoader:
    """
    Loads audio files and creates AudioSample instances.

    Thread-safe and stateless - can be used concurrently.
    """

    def __init__(
        self,
        target_sr: int = TARGET_SAMPLE_RATE,
        max_file_size: int = MAX_FILE_SIZE,
        max_duration: float = MAX_DURATION,
        truncate_long_files: bool = False
    ):
        """
        Initialize loader with configuration.

        Args:
            target_sr: Target sample rate for resampling
            max_file_size: Maximum file size in bytes
            max_duration: Maximum audio duration in seconds
            truncate_long_files: If True, truncate files longer than max_duration
                                 instead of rejecting them
        """
        self.target_sr = target_sr
        self.max_file_size = max_file_size
        self.max_duration = max_duration
        self.truncate_long_files = truncate_long_files
        self.supported_suffixes: Set[str] = set(SUPPORTED_FORMATS.keys())

    def load(self, file_path: Path) -> AudioSample:
        """
        Load audio file and create AudioSample.

        Args:
            file_path: Path to audio file

        Returns:
            AudioSample: Loaded and validated audio sample

        Raises:
            FileNotFoundError: File doesn't exist
            UnsupportedFormatError: File format not supported
            FileTooLargeError: File exceeds size limit
            AudioLoadError: Audio data is invalid
        """
        file_path = Path(file_path)

        # Step 1: Validate file
        self._validate_file(file_path)

        # Step 2: Load original metadata
        original_metadata = self._load_metadata(file_path)

        # Step 3: Load and resample audio
        audio_data, sample_rate = self._load_audio_data(file_path)

        # Step 4: Validate audio data
        audio_data = self._validate_audio_data(audio_data, file_path)

        # Step 5: Compute file hash
        file_hash = self._compute_file_hash(file_path)

        # Step 6: Determine channels and duration
        if audio_data.ndim == 1:
            channels = 1
            duration = len(audio_data) / sample_rate
        else:
            channels = audio_data.shape[0]
            duration = audio_data.shape[1] / sample_rate

        return AudioSample(
            file_path=file_path,
            file_hash=file_hash,
            sample_rate=sample_rate,
            channels=channels,
            duration=duration,
            audio_data=audio_data,
            original_sample_rate=original_metadata['sample_rate'],
            original_bit_depth=original_metadata['bit_depth'],
            original_format=original_metadata['format']
        )

    def _validate_file(self, file_path: Path) -> None:
        """Validate file exists, has supported format, and is within size limit."""
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix not in self.supported_suffixes:
            raise UnsupportedFormatError(
                f"Format {suffix} not supported. "
                f"Supported formats: {', '.join(self.supported_suffixes)}",
                format=suffix
            )

        file_size = file_path.stat().st_size
        if file_size > self.max_file_size:
            raise FileTooLargeError(
                f"File too large: {file_size / 1024 / 1024:.1f} MB. "
                f"Maximum: {self.max_file_size / 1024 / 1024:.1f} MB",
                file_size=file_size,
                max_size=self.max_file_size
            )

    def _validate_audio_data(
        self, audio_data: np.ndarray, file_path: Path
    ) -> np.ndarray:
        """Validate audio data integrity and quality."""
        if audio_data.size == 0:
            raise AudioLoadError(f"Audio file is empty: {file_path}")

        # Note: Duration check is now handled at a higher level (queue system)
        # We don't reject long files here anymore

        # Check for silence
        rms = np.sqrt(np.mean(audio_data ** 2))
        if rms < 1e-6:
            logger.warning(f"Audio appears to be silent: {file_path}")

        # Check for clipping and normalize if needed
        max_abs = np.max(np.abs(audio_data))
        if max_abs > 1.0:
            logger.warning(
                f"Audio contains clipping (max: {max_abs:.2f}), normalizing: {file_path}"
            )
            audio_data = audio_data / max_abs

        return audio_data

    def _load_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Load original audio metadata before resampling."""
        try:
            with sf.SoundFile(str(file_path)) as f:
                metadata = {
                    'sample_rate': f.samplerate,
                    'bit_depth': f.subtype,
                    'format': file_path.suffix.lstrip('.').upper(),
                    'channels': f.channels
                }

                logger.info(
                    f"Loading audio: {metadata['sample_rate']} Hz, "
                    f"{metadata['channels']} ch, {metadata['bit_depth']}"
                )

                return metadata

        except Exception as e:
            # For formats soundfile can't read metadata (like some MP3s)
            logger.warning(f"Could not read metadata with soundfile: {e}")
            return {
                'sample_rate': 44100,  # Assume CD quality
                'bit_depth': 'PCM_16',
                'format': file_path.suffix.lstrip('.').upper(),
                'channels': 2
            }

    def _load_audio_data(self, file_path: Path) -> Tuple[np.ndarray, int]:
        """
        Load audio data with resampling.
        
        Note: Truncation is no longer used. Long files are handled via queue system.
        """
        try:
            audio_data, sample_rate = librosa.load(
                str(file_path),
                sr=self.target_sr,
                mono=False,
                dtype=np.float32
            )

            return audio_data, sample_rate

        except Exception as e:
            raise AudioLoadError(
                f"Failed to load audio data from {file_path}: {e}"
            )
    
    def get_duration(self, file_path: Path) -> float:
        """
        Get duration of audio file without loading full audio data.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Duration in seconds
        """
        try:
            # Use librosa to get duration without loading full audio
            import librosa
            duration = librosa.get_duration(path=str(file_path))
            return duration
        except Exception as e:
            logger.warning(f"Could not get duration for {file_path}: {e}")
            # Fallback: try to load metadata
            try:
                with sf.SoundFile(str(file_path)) as f:
                    return f.frames / f.samplerate
            except Exception:
                # Last resort: return 0 (will be caught during actual load)
                return 0.0
    
    def is_long_file(self, file_path: Path) -> bool:
        """
        Check if file exceeds max_duration without loading full audio.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            True if file exceeds max_duration
        """
        duration = self.get_duration(file_path)
        return duration > self.max_duration

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA-256 hash of file content."""
        sha256 = hashlib.sha256()

        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                sha256.update(chunk)

        return sha256.hexdigest()


class AsyncAudioLoader:
    """Async wrapper around AudioLoader for non-blocking I/O."""

    def __init__(
        self,
        loader: Optional[AudioLoader] = None,
        executor: Optional[ThreadPoolExecutor] = None
    ):
        """
        Initialize async loader.

        Args:
            loader: AudioLoader instance (creates default if None)
            executor: ThreadPoolExecutor (creates default if None)
        """
        self.loader = loader or AudioLoader()
        self.executor = executor or ThreadPoolExecutor(max_workers=4)

    async def load(self, file_path: Path) -> AudioSample:
        """Load audio asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.loader.load,
            file_path
        )

    def shutdown(self) -> None:
        """Shutdown the executor."""
        self.executor.shutdown(wait=True)


def create_audio_loader(config: Optional[Dict[str, Any]] = None) -> AudioLoader:
    """
    Factory function to create AudioLoader with configuration.

    Args:
        config: Optional configuration dict

    Returns:
        AudioLoader: Configured loader instance
    """
    if config is None:
        config = {}

    return AudioLoader(
        target_sr=config.get('target_sample_rate', TARGET_SAMPLE_RATE),
        max_file_size=config.get('max_file_size', MAX_FILE_SIZE),
        max_duration=config.get('max_duration', MAX_DURATION)
    )
