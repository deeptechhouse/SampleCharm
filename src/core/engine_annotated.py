"""
Analysis engine for the Audio Sample Analysis Application.

=============================================================================
ANNOTATED VERSION - Extensive comments for educational purposes
=============================================================================

This is the main orchestration engine that coordinates all analyzers.
Think of it as the "conductor" of an orchestra - it doesn't play any
instruments (analyzers) itself, but coordinates them to produce the
final result.

OVERVIEW FOR JUNIOR DEVELOPERS:
-------------------------------
The Analysis Engine is the central component that:
1. Loads audio files using the AudioLoader
2. Checks the cache for existing results
3. Runs all 4 analyzers in parallel
4. Aggregates results into a single AnalysisResult
5. Caches the result for future use

KEY DESIGN PATTERNS:

1. DEPENDENCY INJECTION
   Instead of creating dependencies inside the class, we "inject"
   them through the constructor. This makes the class:
   - Testable: We can inject mock objects in tests
   - Flexible: Different configurations without code changes
   - Decoupled: Engine doesn't know how analyzers are created

2. PARALLEL EXECUTION
   All 4 analyzers run simultaneously using ThreadPoolExecutor.
   This reduces total time from ~800ms (sequential) to ~200ms.

   Sequential: Load -> Source -> Musical -> Percussive -> Rhythmic
   Parallel:   Load -> [All analyzers simultaneously]

3. GRACEFUL DEGRADATION
   If one analyzer fails, the others continue. We return partial
   results rather than failing completely. This is more useful
   for users - they get some information even if not everything.

4. CACHING
   Results are cached by file hash. If you analyze the same file
   twice, the second time is instant (cache hit).
"""

# =============================================================================
# IMPORTS
# =============================================================================

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.core.cache import CacheManager, create_cache_manager
from src.core.loader import AudioLoader, create_audio_loader
from src.core.models import AnalysisResult, AudioSample
from src.core.multi_strategy import (
    create_musical_analyzer,
    create_percussive_analyzer,
    create_rhythmic_analyzer,
    create_source_analyzer,
)


# =============================================================================
# ANALYSIS ENGINE CLASS
# =============================================================================

class AudioAnalysisEngine:
    """
    Main analysis engine - orchestrates all components.

    This class coordinates the entire analysis pipeline:
    1. Audio loading
    2. Caching
    3. Parallel analyzer execution
    4. Result aggregation

    DESIGN PRINCIPLES:

    1. Dependency Injection
       All components (loader, analyzers, cache) are injected through
       the constructor. The engine doesn't create anything itself.
       This makes testing easy - just inject mock objects.

    2. Parallel Execution
       Analyzers run in parallel using ThreadPoolExecutor.
       4 analyzers x 200ms = 200ms total (not 800ms sequential)

    3. Graceful Degradation
       If one analyzer fails, others continue. We return partial
       results rather than failing completely.

    4. Caching
       Results cached by file hash. Analyzing the same file twice
       is instant (cache hit).

    THREAD SAFETY:
    The engine itself is thread-safe, but should be used as a
    context manager or explicitly shutdown() to clean up resources.

    EXAMPLE USAGE:
        # Using context manager (recommended)
        with create_analysis_engine(config) as engine:
            result = engine.analyze(Path("audio.wav"))
            print(result.get_summary())

        # Manual lifecycle management
        engine = create_analysis_engine(config)
        try:
            result = engine.analyze(Path("audio.wav"))
        finally:
            engine.shutdown()
    """

    def __init__(
        self,
        loader: AudioLoader,
        source_analyzer: Any,
        musical_analyzer: Any,
        percussive_analyzer: Any,
        rhythmic_analyzer: Any,
        cache: Optional[CacheManager] = None,
        max_workers: int = 4
    ):
        """
        Initialize analysis engine.

        All dependencies are injected, following the Dependency Injection
        pattern. This makes the engine:
        - Testable: Inject mocks for testing
        - Configurable: Different analyzers without code changes
        - Decoupled: Engine doesn't know how components are created

        Args:
            loader: AudioLoader instance for loading audio files
            source_analyzer: Source classification analyzer (MultiStrategy)
            musical_analyzer: Musical analysis analyzer (MultiStrategy)
            percussive_analyzer: Percussion analysis analyzer (MultiStrategy)
            rhythmic_analyzer: Rhythmic analysis analyzer
            cache: Optional CacheManager for result caching
            max_workers: Number of parallel workers (default: 4)

        WHY 4 WORKERS?
        We have 4 analyzers, so 4 workers allows maximum parallelism.
        More workers wouldn't help (nothing else to parallelize).
        Fewer workers would serialize some analysis.
        """
        # Store injected dependencies
        self.loader = loader

        # Store analyzers in a dict for easy iteration
        self.analyzers = {
            'source': source_analyzer,
            'musical': musical_analyzer,
            'percussive': percussive_analyzer,
            'rhythmic': rhythmic_analyzer
        }

        # Optional cache manager
        self.cache = cache

        # Thread pool for parallel execution
        # max_workers determines how many analyzers can run simultaneously
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # Logger for this engine instance
        self.logger = logging.getLogger('engine')

    def analyze(self, file_path: Path) -> AnalysisResult:
        """
        Analyze audio file completely.

        This is the main entry point for analysis. It orchestrates:
        1. Loading the audio file
        2. Checking the cache
        3. Running analyzers in parallel
        4. Aggregating results
        5. Caching the result

        Args:
            file_path: Path to audio file to analyze

        Returns:
            AnalysisResult: Complete analysis result with all components

        PERFORMANCE BREAKDOWN (typical):
        - Load audio: ~200ms
        - Cache check: ~1ms
        - Extract features: ~300ms (first access in analyzers)
        - Run analyzers (parallel): ~200ms
        - Total: ~500-700ms

        With cache hit:
        - Cache check: ~1ms
        - Return cached: ~0ms
        - Total: ~1ms

        EXAMPLE:
            engine = create_analysis_engine(config)
            result = engine.analyze(Path("my_song.wav"))

            print(f"Source: {result.source_classification.source_type}")
            print(f"Tempo: {result.rhythmic_analysis.tempo_bpm} BPM")
        """
        # Ensure we have a Path object
        file_path = Path(file_path)

        # Start timing
        start_time = time.time()

        # =================================================================
        # STEP 1: LOAD AUDIO
        # =================================================================
        # This is the first significant operation. It:
        # - Validates the file
        # - Reads and resamples the audio
        # - Computes the file hash
        self.logger.info(f"Loading audio: {file_path}")
        audio = self.loader.load(file_path)

        # =================================================================
        # STEP 2: CHECK CACHE
        # =================================================================
        # If we've analyzed this file before (same content hash),
        # return the cached result immediately.
        if self.cache:
            cached_result = self.cache.get(audio.file_hash)
            if cached_result:
                self.logger.info(f"Cache hit: {audio.file_hash[:8]}...")
                return cached_result

        # =================================================================
        # STEP 3: RUN ANALYZERS IN PARALLEL
        # =================================================================
        # All 4 analyzers run simultaneously. This is where most of
        # the actual work happens.
        self.logger.info("Running analyzers in parallel")
        analysis_results = self._run_analyzers_parallel(audio)

        # =================================================================
        # STEP 4: AGGREGATE RESULTS
        # =================================================================
        # Combine individual analyzer results into one AnalysisResult
        processing_time = time.time() - start_time
        result = self._create_analysis_result(
            audio,
            analysis_results,
            processing_time
        )

        # =================================================================
        # STEP 5: CACHE RESULT
        # =================================================================
        # Store for future queries with the same file
        if self.cache:
            self.cache.set(audio.file_hash, result)

        self.logger.info(f"Analysis complete in {processing_time:.3f}s")
        return result

    def analyze_batch(self, file_paths: List[Path]) -> List[Optional[AnalysisResult]]:
        """
        Analyze multiple files.

        For batch processing, files are analyzed in parallel up to
        max_workers limit. This is more efficient than calling
        analyze() in a loop.

        Args:
            file_paths: List of file paths to analyze

        Returns:
            List[AnalysisResult]: Results in same order as input
                                 (None for files that failed)

        ORDER PRESERVATION:
        Results are returned in the same order as input paths,
        even though processing order may differ due to parallelism.

        ERROR HANDLING:
        If one file fails, others continue. Failed files return None.

        EXAMPLE:
            files = [Path("a.wav"), Path("b.wav"), Path("c.wav")]
            results = engine.analyze_batch(files)

            for path, result in zip(files, results):
                if result:
                    print(f"{path}: {result.get_summary()}")
                else:
                    print(f"{path}: Analysis failed")
        """
        self.logger.info(f"Analyzing batch of {len(file_paths)} files")

        # Submit all analysis tasks to the executor
        # futures dict maps Future -> original Path (for result ordering)
        futures = {
            self.executor.submit(self.analyze, path): path
            for path in file_paths
        }

        # Collect results as they complete
        # as_completed() yields futures as they finish (not in order)
        results: Dict[Path, Optional[AnalysisResult]] = {}
        for future in as_completed(futures):
            path = futures[future]
            try:
                result = future.result()
                results[path] = result
            except Exception as e:
                self.logger.error(f"Failed to analyze {path}: {e}")
                results[path] = None  # Mark as failed

        # Return in original order
        return [results[path] for path in file_paths]

    def _run_analyzers_parallel(self, audio: AudioSample) -> Dict[str, Tuple[Any, bool]]:
        """
        Run all analyzers in parallel.

        This is the core parallelization logic. Each analyzer runs
        in its own thread, processing the same AudioSample.

        Args:
            audio: AudioSample to analyze

        Returns:
            dict: {
                'source': (SourceClassification, used_fallback),
                'musical': (MusicalAnalysis, used_fallback),
                'percussive': (PercussiveAnalysis, used_fallback),
                'rhythmic': (RhythmicAnalysis, used_fallback)
            }

        HOW PARALLEL EXECUTION WORKS:
        1. Submit all 4 analyzer tasks to the executor
        2. Each task runs in a separate thread
        3. as_completed() yields results as they finish
        4. We collect all results before returning

        THREAD SAFETY:
        AudioSample is immutable (frozen dataclass), so it's safe
        to share between threads. Each analyzer reads from it but
        cannot modify it.
        """
        # Submit all analyzer tasks
        # Each submission returns a Future object that will hold the result
        futures = {
            self.executor.submit(self._run_analyzer, name, analyzer, audio): name
            for name, analyzer in self.analyzers.items()
        }

        # Collect results as they complete
        results: Dict[str, Tuple[Any, bool]] = {}

        for future in as_completed(futures):
            name = futures[future]
            try:
                # future.result() blocks until this specific task completes
                result, used_fallback = future.result()
                results[name] = (result, used_fallback)
                self.logger.debug(f"{name} complete (fallback: {used_fallback})")
            except Exception as e:
                # If analyzer fails, log error and store None
                # This allows other analyzers to continue
                self.logger.error(f"{name} failed: {e}")
                results[name] = (None, False)

        return results

    def _run_analyzer(
        self,
        name: str,
        analyzer: Any,
        audio: AudioSample
    ) -> Tuple[Any, bool]:
        """
        Run single analyzer with error handling.

        This method is called by the executor for each analyzer.
        It handles both MultiStrategyAnalyzer (returns tuple) and
        regular analyzers (returns just result).

        Args:
            name: Analyzer name (for logging)
            analyzer: The analyzer instance
            audio: AudioSample to analyze

        Returns:
            Tuple: (result, used_fallback)
                  - result: Analysis result or None on failure
                  - used_fallback: True if LLM fallback was used

        DUCK TYPING:
        We check for 'fallback' attribute to determine if this is
        a MultiStrategyAnalyzer. This is "duck typing" - we care
        about what it can do, not what type it is.
        """
        try:
            # Check if analyzer is MultiStrategyAnalyzer
            # MultiStrategyAnalyzer has a 'fallback' attribute
            if hasattr(analyzer, 'analyze') and hasattr(analyzer, 'fallback'):
                # MultiStrategyAnalyzer returns (result, used_fallback)
                return analyzer.analyze(audio)
            else:
                # Regular analyzer returns just result
                result = analyzer.analyze(audio)
                return (result, False)

        except Exception as e:
            self.logger.error(f"{name} analyzer failed: {e}")
            raise  # Re-raise to be caught by _run_analyzers_parallel

    def _create_analysis_result(
        self,
        audio: AudioSample,
        analysis_results: Dict[str, Tuple[Any, bool]],
        processing_time: float
    ) -> AnalysisResult:
        """
        Create AnalysisResult from individual analyzer results.

        This aggregates all the pieces into a single result object.

        Args:
            audio: Original AudioSample
            analysis_results: Dict of (result, used_fallback) tuples
            processing_time: Total processing time in seconds

        Returns:
            AnalysisResult: Complete aggregated result

        OPTIONAL COMPONENTS:
        Each analysis component is Optional - it may be None if
        that analyzer failed. This allows returning partial results
        rather than failing completely.
        """
        # Extract individual results and fallback flags
        # .get() returns (None, False) if key doesn't exist
        source_result, source_fallback = analysis_results.get('source', (None, False))
        musical_result, musical_fallback = analysis_results.get('musical', (None, False))
        percussive_result, percussive_fallback = analysis_results.get('percussive', (None, False))
        rhythmic_result, rhythmic_fallback = analysis_results.get('rhythmic', (None, False))

        # Build analyzer versions dict
        # Used for tracking which version produced the results
        analyzer_versions = {}
        for name, analyzer in self.analyzers.items():
            if hasattr(analyzer, 'version'):
                analyzer_versions[name] = analyzer.version
            else:
                analyzer_versions[name] = "1.0.0"

        # Build used_fallback dict
        # Tracks whether LLM fallback was used for each analyzer
        used_fallback = {
            'source': source_fallback,
            'musical': musical_fallback,
            'percussive': percussive_fallback,
            'rhythmic': rhythmic_fallback
        }

        # Create and return the complete result
        return AnalysisResult(
            audio_sample_hash=audio.file_hash,
            timestamp=datetime.utcnow(),
            processing_time=processing_time,
            quality_metadata=audio.quality_info,
            source_classification=source_result,
            musical_analysis=musical_result,
            percussive_analysis=percussive_result,
            rhythmic_analysis=rhythmic_result,
            analyzer_versions=analyzer_versions,
            used_fallback=used_fallback
        )

    # =========================================================================
    # LIFECYCLE MANAGEMENT
    # =========================================================================

    def shutdown(self) -> None:
        """
        Shutdown thread pool gracefully.

        IMPORTANT: Always call this when done with the engine!
        The thread pool holds system resources that need cleanup.

        wait=True means we wait for all pending tasks to complete
        before shutting down.
        """
        self.logger.info("Shutting down analysis engine")
        self.executor.shutdown(wait=True)

    def __enter__(self) -> "AudioAnalysisEngine":
        """
        Context manager entry - returns self.

        Allows using the engine with 'with' statement:
            with create_analysis_engine(config) as engine:
                result = engine.analyze(file)
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Context manager exit - cleanup resources.

        Called automatically when leaving 'with' block, even if
        an exception occurred. This ensures proper cleanup.

        Args:
            exc_type: Exception type if exception occurred, None otherwise
            exc_val: Exception value
            exc_tb: Exception traceback
        """
        self.shutdown()


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_analysis_engine(config: Dict[str, Any]) -> AudioAnalysisEngine:
    """
    Factory function to create fully configured analysis engine.

    This is the main entry point for creating an engine. It:
    1. Creates the AudioLoader with configuration
    2. Creates all 4 analyzers with configuration
    3. Creates the cache manager if enabled
    4. Assembles everything into an AudioAnalysisEngine

    Args:
        config: Configuration dictionary with sections:
            - audio: Audio loading settings
            - analyzers: Analyzer-specific settings
            - cache: Caching settings
            - performance: Thread pool settings

    Returns:
        AudioAnalysisEngine: Fully configured and ready to use

    EXAMPLE:
        config = load_config("config/config.yaml")
        engine = create_analysis_engine(config)

        with engine:
            result = engine.analyze(Path("audio.wav"))

    WHY A FACTORY FUNCTION?
    1. Simplifies engine creation - one call does everything
    2. Encapsulates complex initialization logic
    3. Makes testing easier - can mock the factory
    4. Configuration-driven - just change config, not code
    """
    # Create audio loader with configuration
    loader = create_audio_loader(config.get('audio', {}))

    # OpenAI client (not implemented in initial version)
    # This would be used for LLM fallback analyzers
    openai_client = None
    # if config.get('openai', {}).get('enabled'):
    #     openai_client = create_openai_client(config['openai'])

    # Create all 4 analyzers
    # Each factory function handles its specific configuration
    source_analyzer = create_source_analyzer(config, openai_client)
    musical_analyzer = create_musical_analyzer(config, openai_client)
    percussive_analyzer = create_percussive_analyzer(config, openai_client)
    rhythmic_analyzer = create_rhythmic_analyzer(config)

    # Create cache manager if enabled
    cache = None
    cache_config = config.get('cache', {})
    if cache_config.get('enabled', True):
        cache = create_cache_manager(cache_config)

    # Get performance settings
    performance_config = config.get('performance', {})
    max_workers = performance_config.get('max_workers', 4)

    # Assemble and return the engine
    return AudioAnalysisEngine(
        loader=loader,
        source_analyzer=source_analyzer,
        musical_analyzer=musical_analyzer,
        percussive_analyzer=percussive_analyzer,
        rhythmic_analyzer=rhythmic_analyzer,
        cache=cache,
        max_workers=max_workers
    )
