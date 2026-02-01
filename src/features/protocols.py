"""
Protocol and base class for LLM features.

Mirrors the Analyzer(Protocol) / BaseAnalyzer pattern from analyzer_base.py.
LLMFeature is the structural subtyping protocol; BaseLLMFeature provides
the Template Method implementation with gate checking, timing, and logging.
"""

import logging
import time
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, List, Optional, Protocol, TypeVar, Union, runtime_checkable

from src.core.models import AnalysisResult
from src.features.models import CostEstimate, FeatureResult

T = TypeVar("T")


@runtime_checkable
class LLMFeature(Protocol):
    """
    Protocol that all 10 LLM features implement.

    Uses structural subtyping — a class is compatible if it has these
    methods/properties, without explicit inheritance.
    """

    @property
    def feature_id(self) -> str:
        """Unique identifier used for config toggles and entitlement."""
        ...

    @property
    def display_name(self) -> str:
        """Human-readable name for UI display."""
        ...

    @property
    def feature_type(self) -> str:
        """One of: 'single', 'batch', 'single+batch'."""
        ...

    @property
    def version(self) -> str:
        """Feature version for tracking."""
        ...

    def execute(
        self,
        results: Union[AnalysisResult, List[AnalysisResult]],
        **kwargs: Any,
    ) -> FeatureResult:
        """Execute the feature on analysis results."""
        ...

    def estimate_cost(
        self,
        results: Union[AnalysisResult, List[AnalysisResult]],
    ) -> CostEstimate:
        """Estimate cost/time before execution."""
        ...


class BaseLLMFeature(Generic[T]):
    """
    Template Method base class for LLM features.

    Mirrors BaseAnalyzer: execute() is the template method that handles
    gate checking, input validation, timing, and error wrapping.
    Subclasses implement _execute_impl() and _validate_input().

    All features receive the shared LLMClient and FeatureGate via DI.
    """

    def __init__(
        self,
        feature_id: str,
        display_name: str,
        feature_type: str,
        version: str,
        client: Any,  # LLMClient — Any to avoid circular import
        gate: Any,  # FeatureGate — Any to avoid circular import
    ):
        self._feature_id = feature_id
        self._display_name = display_name
        self._feature_type = feature_type
        self._version = version
        self._client = client
        self._gate = gate
        self.logger = logging.getLogger(f"feature.{feature_id}")

    @property
    def feature_id(self) -> str:
        return self._feature_id

    @property
    def display_name(self) -> str:
        return self._display_name

    @property
    def feature_type(self) -> str:
        return self._feature_type

    @property
    def version(self) -> str:
        return self._version

    def execute(
        self,
        results: Union[AnalysisResult, List[AnalysisResult]],
        **kwargs: Any,
    ) -> FeatureResult:
        """
        Template method: gate check → validate → impl → wrap result.

        Args:
            results: Single AnalysisResult or list for batch features.
            **kwargs: Feature-specific parameters.

        Returns:
            FeatureResult wrapping the feature-specific output.

        Raises:
            FeatureDisabledError: If feature is toggled off.
            EntitlementError: If user is not entitled.
            AnalysisError: If execution fails.
        """
        self._gate.check(self._feature_id)
        self._validate_input(results)

        start = time.time()
        self.logger.debug(f"Executing feature '{self._feature_id}'")

        try:
            data = self._execute_impl(results, **kwargs)
        except Exception as e:
            self.logger.error(f"Feature '{self._feature_id}' failed: {e}")
            raise

        elapsed = time.time() - start
        self.logger.info(f"Feature '{self._feature_id}' completed in {elapsed:.3f}s")

        return FeatureResult(
            feature_id=self._feature_id,
            data=data,
            processing_time=elapsed,
            model_used=self._client.model_id,
        )

    def estimate_cost(
        self,
        results: Union[AnalysisResult, List[AnalysisResult]],
    ) -> CostEstimate:
        """
        Estimate cost before execution. Override for accurate estimates.

        Default implementation returns a zero estimate.
        """
        count = len(results) if isinstance(results, list) else 1
        return CostEstimate(
            feature_id=self._feature_id,
            estimated_tokens=0,
            estimated_cost_usd=0.0,
            estimated_time_seconds=0.0,
            sample_count=count,
        )

    @abstractmethod
    def _execute_impl(
        self,
        results: Union[AnalysisResult, List[AnalysisResult]],
        **kwargs: Any,
    ) -> T:
        """Subclasses implement the actual feature logic here."""
        raise NotImplementedError

    def _validate_input(
        self,
        results: Union[AnalysisResult, List[AnalysisResult]],
    ) -> None:
        """
        Validate input matches feature_type expectations.

        Raises ValueError if a batch feature receives a single result
        or a single feature receives a list.
        """
        is_list = isinstance(results, list)

        if self._feature_type == "single" and is_list:
            raise ValueError(
                f"Feature '{self._feature_id}' expects a single AnalysisResult, "
                f"got a list of {len(results)}."
            )
        if self._feature_type == "batch" and not is_list:
            raise ValueError(
                f"Feature '{self._feature_id}' expects a list of AnalysisResult, "
                f"got a single result."
            )
