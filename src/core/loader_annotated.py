"""
Audio loader for the Audio Sample Analysis Application.

=============================================================================
ANNOTATED VERSION - Extensive comments for educational purposes
=============================================================================

This module is responsible for loading audio files from disk and converting
them into AudioSample objects that can be processed by the analysis system.

OVERVIEW FOR JUNIOR DEVELOPERS:
-------------------------------
Loading audio files is more complex than you might think:
1. Different formats (WAV, MP3, FLAC) use different codecs
2. Sample rates vary (44.1kHz, 48kHz, 96kHz, etc.)
3. Bit depths vary (16-bit, 24-bit, 32-bit float)
4. Audio can be mono or stereo
5. Files can be corrupted, too large, or invalid

This loader handles all these cases and normalizes everything to a
consistent format for analysis:
- Sample rate: 22050 Hz (good balance of quality vs. processing speed)
- Format: float32 in range [-1.0, 1.0]
- Channels: preserved (mono or stereo)

KEY LIBRARIES:
- librosa: Audio loading and resampling
- soundfile: Metadata reading (sample rate, bit depth)
- numpy: Array operations

WHY 22050 Hz?
- Nyquist theorem: Can capture frequencies up to 11 kHz
- Most audio analysis doesn't need higher frequencies
- Reduces processing time and memory by ~50% vs 44.1 kHz
- Most ML models (like YAMNet) expect 16 kHz anyway
"""

# =============================================================================
# IMPORTS
# =============================================================================

import asyncio       # For async audio loading
import hashlib       # For computing file hashes
import logging       # For logging messages
from concurrent.futures import ThreadPoolExecutor  # For running blocking I/O in threads
from pathlib import Path   # Modern path handling
from typing import Any, Dict, Optional, Set, Tuple  # Type hints

import librosa       # Audio loading library
import numpy as np   # Numerical operations
import soundfile as sf   # Audio metadata reading

from src.core.models import AudioSample
from src.utils.errors import AudioLoadError, FileTooLargeError, UnsupportedFormatError


# =============================================================================
# CONSTANTS
# =============================================================================

# Mapping of file extensions to the library used to load them
# soundfile: High-quality, supports WAV, FLAC, AIFF
# audioread: Fallback for compressed formats like MP3
SUPPORTED_FORMATS: Dict[str, str] = {
    '.wav': 'soundfile',    # Uncompressed, highest quality
    '.aif': 'soundfile',    # Apple's uncompressed format
    '.aiff': 'soundfile',   # Same as .aif
    '.mp3': 'audioread',    # Compressed, uses external decoder
    '.flac': 'soundfile'    # Lossless compression
}

# Target sample rate for all analysis
# 22050 Hz is standard for audio analysis:
# - Captures up to ~11 kHz (Nyquist)
# - Half of CD quality (44100 Hz)
# - Good balance of quality vs. speed
TARGET_SAMPLE_RATE: int = 22050

# Maximum file size: 50 MB
# Prevents memory issues with very large files
# 50 MB is ~9 minutes of CD-quality stereo audio
MAX_FILE_SIZE: int = 52428800

# Maximum duration: 30 seconds
# Analysis is designed for audio samples, not full songs
# Longer files take too long to process
MAX_DURATION: float = 30.0

# Get logger for this module
logger = logging.getLogger(__name__)


# =============================================================================
# AUDIO LOADER CLASS
# =============================================================================

class AudioLoader:
    """
    Loads audio files and creates AudioSample instances.

    This class is the primary interface for loading audio files.
    It handles:
    - File validation (exists, format, size)
    - Metadata extraction (original sample rate, bit depth)
    - Audio loading and resampling
    - Quality validation (clipping, silence)
    - Hash computation for caching

    DESIGN PRINCIPLES:
    1. Stateless: No mutable state, safe for concurrent use
    2. Fail-fast: Validate early, raise clear errors
    3. Quality preservation: Keep original metadata even after resampling

    THREAD SAFETY:
    This class is thread-safe because it has no mutable state.
    Multiple threads can use the same loader instance safely.

    EXAMPLE USAGE:
        loader = AudioLoader()
        sample = loader.load(Path("my_audio.wav"))
        print(f"Duration: {sample.duration}s")
        print(f"Original quality: {sample.original_sample_rate} Hz")
    """

    def __init__(
        self,
        target_sr: int = TARGET_SAMPLE_RATE,
        max_file_size: int = MAX_FILE_SIZE,
        max_duration: float = MAX_DURATION
    ):
        """
        Initialize loader with configuration.

        Args:
            target_sr: Target sample rate for resampling.
                      All audio will be resampled to this rate.
                      Default: 22050 Hz

            max_file_size: Maximum file size in bytes.
                          Files larger than this will be rejected.
                          Default: 50 MB

            max_duration: Maximum audio duration in seconds.
                         Audio longer than this will be rejected.
                         Default: 30 seconds

        CONFIGURATION STRATEGY:
        These parameters can be customized per-instance, allowing
        different loaders for different use cases:
        - Web API: Strict limits (small files, short duration)
        - Batch processing: Relaxed limits
        - Testing: Very strict limits for fast tests
        """
        self.target_sr = target_sr
        self.max_file_size = max_file_size
        self.max_duration = max_duration

        # Pre-compute set of supported extensions for O(1) lookup
        # Using a set instead of checking dict keys is slightly faster
        self.supported_suffixes: Set[str] = set(SUPPORTED_FORMATS.keys())

    def load(self, file_path: Path) -> AudioSample:
        """
        Load audio file and create AudioSample.

        This is the main entry point for loading audio. It orchestrates
        the entire loading process:
        1. Validate file (exists, format, size)
        2. Load original metadata (before resampling)
        3. Load and resample audio data
        4. Validate audio data (quality checks)
        5. Compute file hash (for caching)
        6. Create AudioSample

        Args:
            file_path: Path to audio file (can be str or Path)

        Returns:
            AudioSample: Loaded and validated audio sample

        Raises:
            FileNotFoundError: File doesn't exist
            UnsupportedFormatError: File format not supported
            FileTooLargeError: File exceeds size limit
            AudioLoadError: Audio data is invalid or corrupted

        EXAMPLE:
            loader = AudioLoader()
            try:
                sample = loader.load(Path("test.wav"))
                print(f"Loaded: {sample.duration}s, {sample.channels} channels")
            except FileNotFoundError:
                print("File not found!")
            except UnsupportedFormatError as e:
                print(f"Bad format: {e.format}")

        PERFORMANCE:
        Typical loading times:
        - Small file (< 1 MB): 50-100 ms
        - Medium file (1-10 MB): 100-300 ms
        - Large file (10-50 MB): 300-500 ms

        Most time is spent on:
        1. File I/O (reading from disk)
        2. Decoding (especially MP3)
        3. Resampling (if needed)
        """
        # Ensure we have a Path object, not a string
        file_path = Path(file_path)

        # Step 1: Validate file exists, has valid format, within size limit
        # This is fast and catches obvious problems early
        self._validate_file(file_path)

        # Step 2: Load metadata BEFORE resampling
        # We want to preserve the original quality info
        original_metadata = self._load_metadata(file_path)

        # Step 3: Load audio data with resampling
        # This is the most time-consuming step
        audio_data, sample_rate = self._load_audio_data(file_path)

        # Step 4: Validate the loaded audio data
        # Check for empty, silence, clipping, duration
        audio_data = self._validate_audio_data(audio_data, file_path)

        # Step 5: Compute hash for caching/identification
        # SHA-256 hash of the original file content
        file_hash = self._compute_file_hash(file_path)

        # Step 6: Determine channel count and duration
        # NumPy array shapes:
        # - Mono: (samples,) - 1D array
        # - Stereo: (2, samples) - 2D array
        if audio_data.ndim == 1:
            channels = 1
            duration = len(audio_data) / sample_rate
        else:
            channels = audio_data.shape[0]  # First dimension is channels
            duration = audio_data.shape[1] / sample_rate  # Second is samples

        # Create and return the AudioSample
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
        """
        Validate file exists, has supported format, and is within size limit.

        This is the first validation step - quick checks that catch
        obvious problems before we try to load the file.

        Args:
            file_path: Path to validate

        Raises:
            FileNotFoundError: File doesn't exist
            UnsupportedFormatError: Format not supported
            FileTooLargeError: File too large

        WHY VALIDATE EARLY?
        It's better to fail immediately with a clear error than to
        spend time loading a file and then fail. This is called
        "fail-fast" programming.
        """
        # Check file exists
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        # Check format is supported
        # .suffix returns the extension with the dot (e.g., ".wav")
        suffix = file_path.suffix.lower()
        if suffix not in self.supported_suffixes:
            raise UnsupportedFormatError(
                f"Format {suffix} not supported. "
                f"Supported formats: {', '.join(self.supported_suffixes)}",
                format=suffix
            )

        # Check file size
        # stat().st_size returns file size in bytes
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
        """
        Validate audio data integrity and quality.

        After loading the raw audio data, we check:
        1. Not empty (has samples)
        2. Duration within limits
        3. Not silent (has some energy)
        4. Not clipped (within [-1, 1] range)

        Args:
            audio_data: Loaded audio data (numpy array)
            file_path: Original file path (for error messages)

        Returns:
            np.ndarray: Validated (possibly normalized) audio data

        Raises:
            AudioLoadError: Audio data is invalid

        NORMALIZATION:
        If audio exceeds the [-1, 1] range (clipping), we normalize
        it to prevent analysis errors. We log a warning so the user
        knows the audio was modified.
        """
        # Check not empty
        if audio_data.size == 0:
            raise AudioLoadError(f"Audio file is empty: {file_path}")

        # Check duration
        # Duration calculation depends on array shape
        if audio_data.ndim == 1:
            duration = len(audio_data) / self.target_sr
        else:
            duration = audio_data.shape[1] / self.target_sr

        if duration > self.max_duration:
            raise AudioLoadError(
                f"Audio too long: {duration:.1f}s. Maximum: {self.max_duration}s"
            )

        # Check for silence
        # RMS (Root Mean Square) measures average energy
        # Very low RMS indicates silent or nearly silent audio
        rms = np.sqrt(np.mean(audio_data ** 2))
        if rms < 1e-6:
            logger.warning(f"Audio appears to be silent: {file_path}")
            # Note: We warn but don't reject - some use cases may want this

        # Check for clipping and normalize if needed
        # Audio should be in range [-1.0, 1.0]
        max_abs = np.max(np.abs(audio_data))
        if max_abs > 1.0:
            logger.warning(
                f"Audio contains clipping (max: {max_abs:.2f}), normalizing: {file_path}"
            )
            # Normalize by dividing by max value
            audio_data = audio_data / max_abs

        return audio_data

    def _load_metadata(self, file_path: Path) -> Dict[str, Any]:
        """
        Load original audio metadata before resampling.

        We use soundfile to read metadata because:
        - It's fast (doesn't decode audio)
        - It provides detailed format info
        - It works with most formats

        Args:
            file_path: Path to audio file

        Returns:
            dict: Metadata including sample_rate, bit_depth, format

        METADATA PRESERVATION:
        We save the original quality metadata even though we resample
        the audio. This is important for:
        - Displaying quality info to users
        - Adjusting analysis parameters based on input quality
        - Debugging and logging
        """
        try:
            # soundfile.SoundFile provides metadata without loading audio
            # Using context manager ensures file is properly closed
            with sf.SoundFile(str(file_path)) as f:
                metadata = {
                    'sample_rate': f.samplerate,  # e.g., 44100, 48000
                    'bit_depth': f.subtype,       # e.g., 'PCM_16', 'PCM_24'
                    'format': file_path.suffix.lstrip('.').upper(),
                    'channels': f.channels
                }

                logger.info(
                    f"Loading audio: {metadata['sample_rate']} Hz, "
                    f"{metadata['channels']} ch, {metadata['bit_depth']}"
                )

                return metadata

        except Exception as e:
            # Some formats (like certain MP3s) can't be read by soundfile
            # In this case, we use default assumptions
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

        Uses librosa.load() which handles:
        - Multiple formats (via audioread backend)
        - Automatic resampling to target_sr
        - Type conversion to float32

        Args:
            file_path: Path to audio file

        Returns:
            Tuple[np.ndarray, int]: (audio_data, sample_rate)
                - audio_data: Shape (samples,) for mono, (2, samples) for stereo
                - sample_rate: Always equals target_sr after resampling

        LIBROSA.LOAD PARAMETERS:
        - sr: Target sample rate (None = keep original)
        - mono: False to preserve stereo
        - dtype: np.float32 for efficiency

        PERFORMANCE TIP:
        If you only need mono audio (which most analysis does),
        set mono=True for faster loading. We keep stereo for
        spatial analysis features.
        """
        try:
            audio_data, sample_rate = librosa.load(
                str(file_path),     # librosa expects string path
                sr=self.target_sr,  # Resample to target
                mono=False,         # Preserve stereo
                dtype=np.float32    # Efficient float format
            )

            return audio_data, sample_rate

        except Exception as e:
            raise AudioLoadError(
                f"Failed to load audio data from {file_path}: {e}"
            )

    def _compute_file_hash(self, file_path: Path) -> str:
        """
        Compute SHA-256 hash of file content.

        The hash is used for:
        1. Cache keys: Same content = same hash = cache hit
        2. Duplicate detection: Find identical files
        3. Result identification: Link results to source files

        Args:
            file_path: Path to file

        Returns:
            str: Hex-encoded SHA-256 hash (64 characters)

        WHY SHA-256?
        - Cryptographically secure (no collisions in practice)
        - Fast enough for our use case
        - Standard format, well-supported

        CHUNKED READING:
        We read in 8KB chunks to handle large files without
        loading the entire file into memory.
        """
        sha256 = hashlib.sha256()

        # Read file in chunks for memory efficiency
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(8192)  # 8KB chunks
                if not chunk:
                    break
                sha256.update(chunk)

        return sha256.hexdigest()


# =============================================================================
# ASYNC WRAPPER
# =============================================================================

class AsyncAudioLoader:
    """
    Async wrapper around AudioLoader for non-blocking I/O.

    In async applications (like web servers), blocking I/O operations
    can block the entire event loop. This wrapper runs the blocking
    AudioLoader.load() in a thread pool, allowing other tasks to run.

    WHEN TO USE:
    - FastAPI/Starlette web applications
    - Async batch processing
    - Any async context where you need to load audio

    EXAMPLE:
        async def handle_upload(file):
            loader = AsyncAudioLoader()
            sample = await loader.load(file.path)
            return await analyze_async(sample)
    """

    def __init__(
        self,
        loader: Optional[AudioLoader] = None,
        executor: Optional[ThreadPoolExecutor] = None
    ):
        """
        Initialize async loader.

        Args:
            loader: AudioLoader instance (creates default if None)
            executor: ThreadPoolExecutor for running blocking I/O
                     (creates default with 4 workers if None)

        THREAD POOL SIZING:
        Default of 4 workers is usually good because:
        - File I/O is the bottleneck, not CPU
        - More workers = more memory for loaded audio
        - 4 parallel loads is usually enough
        """
        self.loader = loader or AudioLoader()
        self.executor = executor or ThreadPoolExecutor(max_workers=4)

    async def load(self, file_path: Path) -> AudioSample:
        """
        Load audio asynchronously.

        This method runs the blocking load() in a thread pool,
        allowing the async event loop to continue processing
        other tasks while the file is being loaded.

        Args:
            file_path: Path to audio file

        Returns:
            AudioSample: Loaded audio sample

        HOW IT WORKS:
        1. Get current event loop
        2. Submit load() to thread pool
        3. await the result (non-blocking)
        4. Return when load() completes

        The 'await' keyword makes this non-blocking - the event
        loop can run other coroutines while we wait.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.loader.load,
            file_path
        )

    def shutdown(self) -> None:
        """
        Shutdown the executor.

        Always call this when done to clean up thread resources.
        """
        self.executor.shutdown(wait=True)


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_audio_loader(config: Optional[Dict[str, Any]] = None) -> AudioLoader:
    """
    Factory function to create AudioLoader with configuration.

    Factory functions are preferred over direct instantiation because:
    1. They can apply configuration consistently
    2. They're easier to mock in tests
    3. They can return different implementations based on config

    Args:
        config: Optional configuration dict with keys:
            - target_sample_rate: int (default: 22050)
            - max_file_size: int in bytes (default: 50 MB)
            - max_duration: float in seconds (default: 30)

    Returns:
        AudioLoader: Configured loader instance

    EXAMPLE:
        config = {
            'target_sample_rate': 16000,  # For YAMNet
            'max_file_size': 10 * 1024 * 1024,  # 10 MB
            'max_duration': 10.0  # 10 seconds
        }
        loader = create_audio_loader(config)
    """
    if config is None:
        config = {}

    return AudioLoader(
        target_sr=config.get('target_sample_rate', TARGET_SAMPLE_RATE),
        max_file_size=config.get('max_file_size', MAX_FILE_SIZE),
        max_duration=config.get('max_duration', MAX_DURATION)
    )
