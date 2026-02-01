"""
Multi-strategy analyzer for the Audio Sample Analysis Application.

Implements confidence-based fallback pattern - the KEY pattern for
Plan 3 (Hybrid-OpenAI approach).
"""

import logging
from typing import Any, Callable, Dict, Generic, Optional, Tuple, TypeVar

from src.core.analyzer_base import Analyzer
from src.core.models import (
    AudioSample,
    LLMAnalysis,
    MusicalAnalysis,
    PercussiveAnalysis,
    RhythmicAnalysis,
    SourceClassification,
)

T = TypeVar('T')


class MultiStrategyAnalyzer(Generic[T]):
    """
    Analyzer that tries primary first, falls back to secondary if needed.

    Design:
    - Primary analyzer: Fast, local (YAMNet, librosa, RF)
    - Fallback analyzer: Slow, accurate (OpenAI GPT-4V)
    - Confidence threshold: Determines when to use fallback
    - Optional fallback: Can be disabled globally or per-instance

    Returns:
        Tuple[T, bool]: (result, used_fallback)
    """

    def __init__(
        self,
        name: str,
        primary: Analyzer[T],
        fallback: Optional[Analyzer[T]] = None,
        confidence_threshold: float = 0.75,
        enable_fallback: bool = True,
        confidence_extractor: Optional[Callable[[T], float]] = None
    ):
        """
        Initialize multi-strategy analyzer.

        Args:
            name: Analyzer name
            primary: Primary (fast) analyzer
            fallback: Optional fallback (slow) analyzer
            confidence_threshold: Minimum confidence to accept primary result
            enable_fallback: Global fallback toggle
            confidence_extractor: Function to extract confidence from result
        """
        self._name = name
        self.primary = primary
        self.fallback = fallback
        self.confidence_threshold = confidence_threshold
        self.enable_fallback = enable_fallback
        self._confidence_extractor = confidence_extractor or self._default_confidence_extractor
        self.logger = logging.getLogger(f"multi_strategy.{name}")

    @property
    def name(self) -> str:
        """Return analyzer name."""
        return self._name

    @property
    def version(self) -> str:
        """Version includes both primary and fallback versions."""
        if self.fallback:
            return f"{self.primary.version}+{self.fallback.version}"
        return self.primary.version

    def _default_confidence_extractor(self, result: T) -> float:
        """Default confidence extraction - looks for .confidence attribute."""
        if hasattr(result, 'confidence'):
            return result.confidence
        return 1.0  # Assume full confidence if no confidence attribute

    def analyze(self, audio: AudioSample) -> Tuple[T, bool]:
        """
        Analyze with confidence-based fallback.

        Process:
        1. Run primary analyzer
        2. Extract confidence from result
        3. If confidence >= threshold OR fallback disabled, return primary
        4. Otherwise, run fallback analyzer
        5. Return fallback result

        Args:
            audio: AudioSample to analyze

        Returns:
            Tuple[T, bool]: (result, used_fallback)
        """
        # Step 1: Run primary analyzer
        self.logger.debug(f"Running primary analyzer: {self.primary.name}")
        primary_result = self.primary.analyze(audio)

        # Step 2: Extract confidence
        confidence = self._confidence_extractor(primary_result)
        self.logger.debug(f"Primary confidence: {confidence:.3f}")

        # Step 3: Check if fallback needed
        needs_fallback = (
            confidence < self.confidence_threshold and
            self.enable_fallback and
            self.fallback is not None
        )

        if not needs_fallback:
            if confidence >= self.confidence_threshold:
                self.logger.info(
                    f"Primary result accepted (confidence: {confidence:.3f})"
                )
            elif not self.enable_fallback:
                self.logger.info("Fallback disabled, using primary result")
            else:
                self.logger.info("No fallback available, using primary result")

            return (primary_result, False)

        # Step 4: Run fallback analyzer
        self.logger.info(
            f"Primary confidence {confidence:.3f} < threshold "
            f"{self.confidence_threshold:.3f}, using fallback: {self.fallback.name}"
        )

        try:
            fallback_result = self.fallback.analyze(audio)
            fallback_confidence = self._confidence_extractor(fallback_result)

            self.logger.info(f"Fallback confidence: {fallback_confidence:.3f}")

            # Step 5: Return fallback result
            return (fallback_result, True)

        except Exception as e:
            # If fallback fails, return primary result
            self.logger.error(f"Fallback failed: {e}, using primary result")
            return (primary_result, False)

    def disable_fallback(self) -> None:
        """Disable fallback for this analyzer."""
        self.enable_fallback = False
        self.logger.info("Fallback disabled")

    def enable_fallback_if_available(self) -> None:
        """Enable fallback if available."""
        if self.fallback is not None:
            self.enable_fallback = True
            self.logger.info("Fallback enabled")
        else:
            self.logger.warning(
                "Cannot enable fallback: no fallback analyzer configured"
            )


# Factory functions

def create_source_analyzer(
    config: Dict[str, Any],
    openai_client: Optional[Any] = None
) -> MultiStrategyAnalyzer[SourceClassification]:
    """
    Create source classification analyzer with optional OpenAI fallback.

    Args:
        config: Configuration dictionary
        openai_client: Optional OpenAI client for fallback

    Returns:
        MultiStrategyAnalyzer: Configured analyzer
    """
    from src.analyzers.source.yamnet import YAMNetAnalyzer

    # Primary analyzer (always available)
    primary = YAMNetAnalyzer()

    # Fallback analyzer (optional)
    fallback = None
    analyzers_config = config.get('analyzers', {})
    source_config = analyzers_config.get('source', {})

    # Note: OpenAI fallback not implemented in initial version
    # if config.get('openai', {}).get('enabled', False) and openai_client:
    #     from src.analyzers.source.openai_source import OpenAISourceAnalyzer
    #     fallback = OpenAISourceAnalyzer(openai_client)

    return MultiStrategyAnalyzer(
        name="source_classification",
        primary=primary,
        fallback=fallback,
        confidence_threshold=source_config.get('confidence_threshold', 0.75),
        enable_fallback=source_config.get('enable_fallback', False)
    )


def create_musical_analyzer(
    config: Dict[str, Any],
    openai_client: Optional[Any] = None
) -> MultiStrategyAnalyzer[MusicalAnalysis]:
    """
    Create musical analysis analyzer.

    Args:
        config: Configuration dictionary
        openai_client: Optional OpenAI client for fallback

    Returns:
        MultiStrategyAnalyzer: Configured analyzer
    """
    from src.analyzers.musical.librosa_musical import LibrosaMusicalAnalyzer

    primary = LibrosaMusicalAnalyzer()
    fallback = None

    analyzers_config = config.get('analyzers', {})
    musical_config = analyzers_config.get('musical', {})

    def musical_confidence_extractor(result: MusicalAnalysis) -> float:
        """Extract confidence from musical analysis."""
        if result.has_pitch:
            return result.key_confidence
        return 1.0  # No pitch = no need for fallback

    return MultiStrategyAnalyzer(
        name="musical_analysis",
        primary=primary,
        fallback=fallback,
        confidence_threshold=musical_config.get('confidence_threshold', 0.70),
        enable_fallback=musical_config.get('enable_fallback', False),
        confidence_extractor=musical_confidence_extractor
    )


def create_percussive_analyzer(
    config: Dict[str, Any],
    openai_client: Optional[Any] = None
) -> MultiStrategyAnalyzer[PercussiveAnalysis]:
    """
    Create percussion classification analyzer.

    Args:
        config: Configuration dictionary
        openai_client: Optional OpenAI client for fallback

    Returns:
        MultiStrategyAnalyzer: Configured analyzer
    """
    from src.analyzers.percussive.random_forest import RandomForestPercussiveAnalyzer

    primary = RandomForestPercussiveAnalyzer()
    fallback = None

    analyzers_config = config.get('analyzers', {})
    percussive_config = analyzers_config.get('percussive', {})

    return MultiStrategyAnalyzer(
        name="percussive_analysis",
        primary=primary,
        fallback=fallback,
        confidence_threshold=percussive_config.get('confidence_threshold', 0.75),
        enable_fallback=percussive_config.get('enable_fallback', False)
    )


def create_rhythmic_analyzer(
    config: Dict[str, Any]
) -> "LibrosaRhythmicAnalyzer":
    """
    Create rhythmic analysis analyzer.

    Note: No fallback for rhythmic analysis (librosa is sufficient).

    Args:
        config: Configuration dictionary

    Returns:
        LibrosaRhythmicAnalyzer: Configured analyzer
    """
    from src.analyzers.rhythmic.librosa_rhythmic import LibrosaRhythmicAnalyzer

    return LibrosaRhythmicAnalyzer()


def create_llm_analyzer(
    config: Dict[str, Any]
) -> Optional["LLMAnalyzer"]:
    """
    Create LLM-based audio analyzer for naming, description, and speech detection.

    Args:
        config: Configuration dictionary

    Returns:
        LLMAnalyzer or None if LLM is disabled/not configured
    """
    llm_config = config.get('llm', {})

    if not llm_config.get('enabled', False):
        return None

    from src.analyzers.llm.llm_analyzer import LLMAnalyzer

    return LLMAnalyzer(
        provider=llm_config.get('provider', 'togetherai'),
        model=llm_config.get('model'),
        api_key=llm_config.get('api_key'),
        temperature=llm_config.get('temperature', 0.3),
        max_tokens=llm_config.get('max_tokens', 1000)
    )
