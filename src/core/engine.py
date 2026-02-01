"""
Analysis engine for the Audio Sample Analysis Application.

Main orchestration engine that coordinates all analyzers.
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    # Load .env from project root
    env_path = Path(__file__).parent.parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    # python-dotenv not installed, skip
    pass

from src.core.cache import CacheManager, create_cache_manager
from src.core.loader import AudioLoader, create_audio_loader
from src.core.models import AnalysisResult, AudioSample
from src.core.multi_strategy import (
    create_llm_analyzer,
    create_musical_analyzer,
    create_percussive_analyzer,
    create_rhythmic_analyzer,
    create_source_analyzer,
)
from src.features.manager import LLMFeatureManager
from src.features.models import FeatureResult

# Speech analyzer (optional - requires whisper)
try:
    from src.analyzers.speech import create_whisper_speech_analyzer
except ImportError:
    # Speech analyzer not available (whisper not installed)
    create_whisper_speech_analyzer = None


class AudioAnalysisEngine:
    """
    Main analysis engine - orchestrates all components.

    Design:
    - Dependency Injection: All dependencies injected (testable)
    - Parallel Execution: Analyzers run concurrently
    - Caching: Results cached by file hash
    - Error Handling: Partial results on failure
    """

    def __init__(
        self,
        loader: AudioLoader,
        source_analyzer: Any,
        musical_analyzer: Any,
        percussive_analyzer: Any,
        rhythmic_analyzer: Any,
        llm_analyzer: Optional[Any] = None,
        speech_analyzer: Optional[Any] = None,
        cache: Optional[CacheManager] = None,
        max_workers: int = 4,
        feature_manager: Optional[LLMFeatureManager] = None,
    ):
        """
        Initialize analysis engine.

        Args:
            loader: AudioLoader instance
            source_analyzer: Source classification analyzer
            musical_analyzer: Musical analysis analyzer
            percussive_analyzer: Percussion classification analyzer
            rhythmic_analyzer: Rhythmic analysis analyzer
            llm_analyzer: Optional LLM-based analyzer for naming/description
            speech_analyzer: Optional speech recognition analyzer (Whisper)
            cache: Optional cache manager
            max_workers: Max parallel workers
            feature_manager: Optional LLM feature manager for downstream AI features
        """
        self.loader = loader
        self.analyzers = {
            'source': source_analyzer,
            'musical': musical_analyzer,
            'percussive': percussive_analyzer,
            'rhythmic': rhythmic_analyzer
        }
        # LLM analyzer is optional
        if llm_analyzer is not None:
            self.analyzers['llm'] = llm_analyzer
        # Speech analyzer is optional
        if speech_analyzer is not None:
            self.analyzers['speech'] = speech_analyzer
        self.cache = cache
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.logger = logging.getLogger('engine')
        self.feature_manager = feature_manager

    def analyze(self, file_path: Path) -> AnalysisResult:
        """
        Analyze audio file completely.

        Args:
            file_path: Path to audio file

        Returns:
            AnalysisResult: Complete analysis result
        """
        file_path = Path(file_path)
        start_time = time.time()

        # Step 1: Load audio
        self.logger.info(f"Loading audio: {file_path}")
        audio = self.loader.load(file_path)

        # Step 2: Check cache
        if self.cache:
            cached_result = self.cache.get(audio.file_hash)
            if cached_result:
                self.logger.info(f"Cache hit: {audio.file_hash[:8]}...")
                return cached_result

        # Step 3: Run analyzers in parallel
        self.logger.info("Running analyzers in parallel")
        analysis_results = self._run_analyzers_parallel(audio)

        # Step 4: Aggregate results
        processing_time = time.time() - start_time
        result = self._create_analysis_result(
            audio,
            analysis_results,
            processing_time
        )

        # Step 5: Cache result
        if self.cache:
            self.cache.set(audio.file_hash, result)

        self.logger.info(f"Analysis complete in {processing_time:.3f}s")
        return result

    def analyze_batch(self, file_paths: List[Path]) -> List[Optional[AnalysisResult]]:
        """
        Analyze multiple files.

        Args:
            file_paths: List of file paths

        Returns:
            List[AnalysisResult]: Results in same order as input
        """
        self.logger.info(f"Analyzing batch of {len(file_paths)} files")

        # Submit all tasks
        futures = {
            self.executor.submit(self.analyze, path): path
            for path in file_paths
        }

        # Collect results in order
        results: Dict[Path, Optional[AnalysisResult]] = {}
        for future in as_completed(futures):
            path = futures[future]
            try:
                result = future.result()
                results[path] = result
            except Exception as e:
                self.logger.error(f"Failed to analyze {path}: {e}")
                results[path] = None

        # Return in original order
        return [results[path] for path in file_paths]

    def _run_analyzers_parallel(self, audio: AudioSample) -> Dict[str, Tuple[Any, bool]]:
        """
        Run all analyzers in parallel with optimized speech handling.

        Speech and LLM analyzers are coordinated: speech runs in parallel with
        other analyzers, and LLM waits for speech results if available.

        Returns:
            dict: {name: (result, used_fallback)}
        """
        results: Dict[str, Tuple[Any, bool]] = {}

        # Separate analyzers into groups for optimal parallelization
        # Group 1: Analyzers that don't need speech data (can run immediately)
        independent_analyzers = {
            name: analyzer for name, analyzer in self.analyzers.items()
            if name not in ('speech', 'llm')
        }

        # Group 2: Speech analyzer (runs in parallel with Group 1)
        speech_analyzer = self.analyzers.get('speech')

        # Group 3: LLM analyzer (needs speech results for enhancement)
        llm_analyzer = self.analyzers.get('llm')

        # Submit independent analyzers and speech analyzer in parallel
        futures = {}

        # Submit all independent analyzers
        for name, analyzer in independent_analyzers.items():
            future = self.executor.submit(self._run_analyzer, name, analyzer, audio, None)
            futures[future] = name

        # Submit speech analyzer in parallel (if available)
        speech_future = None
        if speech_analyzer is not None:
            speech_future = self.executor.submit(
                self._run_analyzer, 'speech', speech_analyzer, audio, None
            )
            futures[speech_future] = 'speech'

        # Collect results from independent analyzers and speech
        speech_data = None
        for future in as_completed(futures):
            name = futures[future]
            try:
                result, used_fallback = future.result()
                results[name] = (result, used_fallback)
                self.logger.debug(f"{name} complete (fallback: {used_fallback})")

                # Capture speech data for LLM enhancement
                if name == 'speech' and result is not None:
                    speech_data = result
                    self.logger.debug(
                        f"Speech analyzer completed: contains_speech={speech_data.get('contains_speech')}, "
                        f"transcription_length={len(speech_data.get('transcription', ''))}"
                    )
            except Exception as e:
                self.logger.error(f"{name} failed: {e}")
                results[name] = (None, False)

        # Run LLM analyzer last with speech data (if available)
        if llm_analyzer is not None:
            try:
                llm_result, llm_fallback = self._run_analyzer(
                    'llm', llm_analyzer, audio, speech_data
                )
                results['llm'] = (llm_result, llm_fallback)
                self.logger.debug(f"llm complete (fallback: {llm_fallback})")
            except Exception as e:
                self.logger.error(f"llm failed: {e}")
                results['llm'] = (None, False)

        return results

    def _run_analyzer(
        self,
        name: str,
        analyzer: Any,
        audio: AudioSample,
        speech_data: Optional[dict] = None
    ) -> Tuple[Any, bool]:
        """
        Run single analyzer with error handling.

        Args:
            name: Analyzer name
            analyzer: Analyzer instance
            audio: Audio sample
            speech_data: Optional speech recognition results (for LLM analyzer)

        Returns:
            Tuple: (result, used_fallback)
        """
        try:
            # Check if analyzer is MultiStrategyAnalyzer
            if hasattr(analyzer, 'analyze') and hasattr(analyzer, 'fallback'):
                # Returns (result, used_fallback)
                return analyzer.analyze(audio)
            elif name == 'llm' and speech_data is not None:
                # LLM analyzer with speech data enhancement
                if hasattr(analyzer, '_analyze_impl'):
                    result = analyzer._analyze_impl(audio, speech_data=speech_data)
                    return (result, False)
                else:
                    result = analyzer.analyze(audio)
                    return (result, False)
            else:
                # Regular analyzer, no fallback tracking
                result = analyzer.analyze(audio)
                return (result, False)

        except Exception as e:
            self.logger.error(f"{name} analyzer failed: {e}")
            raise

    def _create_analysis_result(
        self,
        audio: AudioSample,
        analysis_results: Dict[str, Tuple[Any, bool]],
        processing_time: float
    ) -> AnalysisResult:
        """
        Create AnalysisResult from individual analyzer results.

        Args:
            audio: Original audio sample
            analysis_results: Dict of (result, used_fallback) tuples
            processing_time: Total processing time

        Returns:
            AnalysisResult: Aggregated result
        """
        # Extract results and fallback flags
        source_result, source_fallback = analysis_results.get('source', (None, False))
        musical_result, musical_fallback = analysis_results.get('musical', (None, False))
        percussive_result, percussive_fallback = analysis_results.get('percussive', (None, False))
        rhythmic_result, rhythmic_fallback = analysis_results.get('rhythmic', (None, False))
        llm_result, llm_fallback = analysis_results.get('llm', (None, False))
        speech_result, speech_fallback = analysis_results.get('speech', (None, False))

        # Build analyzer versions dict
        analyzer_versions = {}
        for name, analyzer in self.analyzers.items():
            if hasattr(analyzer, 'version'):
                analyzer_versions[name] = analyzer.version
            else:
                analyzer_versions[name] = "1.0.0"

        # Build used_fallback dict
        used_fallback = {
            'source': source_fallback,
            'musical': musical_fallback,
            'percussive': percussive_fallback,
            'rhythmic': rhythmic_fallback,
            'llm': llm_fallback
        }

        return AnalysisResult(
            audio_sample_hash=audio.file_hash,
            timestamp=datetime.utcnow(),
            processing_time=processing_time,
            quality_metadata=audio.quality_info,
            source_classification=source_result,
            musical_analysis=musical_result,
            percussive_analysis=percussive_result,
            rhythmic_analysis=rhythmic_result,
            llm_analysis=llm_result,
            analyzer_versions=analyzer_versions,
            used_fallback=used_fallback
        )

    def run_feature(
        self,
        feature_id: str,
        results: Union[AnalysisResult, List[AnalysisResult]],
        **kwargs: Any,
    ) -> FeatureResult:
        """
        Run a downstream LLM feature on analysis results.

        Args:
            feature_id: ID of the feature to run (e.g. "production_notes").
            results: One or more AnalysisResult objects from analyze()/analyze_batch().
            **kwargs: Feature-specific parameters.

        Returns:
            FeatureResult wrapping the feature output.

        Raises:
            RuntimeError: If no feature_manager is configured.
            KeyError: If feature_id is not registered.
            FeatureDisabledError: If feature is toggled off.
            EntitlementError: If user is not entitled.
        """
        if self.feature_manager is None:
            raise RuntimeError(
                "No feature_manager configured. Pass feature_config to create_analysis_engine()."
            )
        return self.feature_manager.execute(feature_id, results, **kwargs)

    def shutdown(self) -> None:
        """Shutdown thread pool gracefully."""
        self.logger.info("Shutting down analysis engine")
        self.executor.shutdown(wait=True)

    def __enter__(self) -> "AudioAnalysisEngine":
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Cleanup on context exit."""
        self.shutdown()


def create_analysis_engine(
    config: Dict[str, Any],
    feature_config: Optional[Dict[str, Any]] = None,
) -> AudioAnalysisEngine:
    """
    Factory function to create fully configured analysis engine.

    Args:
        config: Configuration dict
        feature_config: Optional LLM feature configuration. If provided,
            creates an LLMFeatureManager with shared client and gate.
            Falls back to config["llm_features"] if present and feature_config
            is not explicitly provided.

    Returns:
        AudioAnalysisEngine: Configured engine
    """
    # Create loader
    loader = create_audio_loader(config.get('audio', {}))

    # OpenAI client (not implemented in initial version)
    openai_client = None

    # Create analyzers (with optional OpenAI fallback)
    source_analyzer = create_source_analyzer(config, openai_client)
    musical_analyzer = create_musical_analyzer(config, openai_client)
    percussive_analyzer = create_percussive_analyzer(config, openai_client)
    rhythmic_analyzer = create_rhythmic_analyzer(config)

    # Create speech analyzer (optional - for high-accuracy speech recognition)
    speech_analyzer = None
    if create_whisper_speech_analyzer is not None:
        try:
            speech_analyzer = create_whisper_speech_analyzer(config)
        except Exception as e:
            logging.getLogger('engine').warning(f"Failed to create speech analyzer: {e}")

    # Create LLM analyzer (optional - for naming, description, speech detection)
    llm_analyzer = create_llm_analyzer(config)

    # Create cache (if enabled)
    cache = None
    cache_config = config.get('cache', {})
    if cache_config.get('enabled', True):
        cache = create_cache_manager(cache_config)

    # Get max workers
    performance_config = config.get('performance', {})
    max_workers = performance_config.get('max_workers', 4)

    # Create feature manager (optional — downstream LLM features)
    feature_manager = None
    feat_cfg = feature_config or config.get("llm_features")
    if feat_cfg:
        try:
            from src.features.client import create_llm_client
            from src.features.gate import FeatureGate, AlwaysEntitled

            llm_client = create_llm_client(config.get("llm", {}))
            gate = FeatureGate(
                config=feat_cfg,
                entitlement_provider=AlwaysEntitled(),
            )
            feature_manager = LLMFeatureManager(llm_client, gate)
            logging.getLogger('engine').info(
                "LLM feature manager initialized with %d features",
                len(feature_manager.list_features()),
            )
        except Exception as e:
            logging.getLogger('engine').warning(f"Failed to create feature manager: {e}")

    return AudioAnalysisEngine(
        loader=loader,
        source_analyzer=source_analyzer,
        musical_analyzer=musical_analyzer,
        percussive_analyzer=percussive_analyzer,
        rhythmic_analyzer=rhythmic_analyzer,
        llm_analyzer=llm_analyzer,
        speech_analyzer=speech_analyzer,
        cache=cache,
        max_workers=max_workers,
        feature_manager=feature_manager,
    )
