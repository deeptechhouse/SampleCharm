"""
LLM Feature Manager — orchestrates all 10 LLM-powered features.

Sits downstream of the core AudioAnalysisEngine, consuming AnalysisResult
objects and routing them to the appropriate feature implementations.
"""

import logging
from typing import Any, Dict, List, Optional, Union

from src.core.models import AnalysisResult
from src.features.client import LLMClient
from src.features.gate import FeatureGate
from src.features.models import CostEstimate, FeatureResult
from src.features.protocols import BaseLLMFeature


class LLMFeatureManager:
    """
    Orchestrates execution of LLM-powered features.

    Responsibilities:
        - Registers all available features with shared client and gate
        - Routes execute() calls to the correct feature
        - Provides cost estimation before execution
        - Lists features with availability status for UI

    Usage:
        manager = LLMFeatureManager(client, gate)
        result = manager.execute("production_notes", analysis_result)
        features = manager.list_features()
    """

    def __init__(self, client: LLMClient, gate: FeatureGate):
        self._client = client
        self._gate = gate
        self._features: Dict[str, BaseLLMFeature] = {}
        self.logger = logging.getLogger("features.manager")
        self._register_features()

    def _register_features(self) -> None:
        """Instantiate all features with shared client and gate."""
        from src.features.implementations import ALL_FEATURE_CLASSES

        for cls in ALL_FEATURE_CLASSES:
            feature = cls(client=self._client, gate=self._gate)
            self._features[feature.feature_id] = feature
            self.logger.debug(f"Registered feature: {feature.feature_id}")

        self.logger.info(f"Registered {len(self._features)} features")

    def execute(
        self,
        feature_id: str,
        results: Union[AnalysisResult, List[AnalysisResult]],
        **kwargs: Any,
    ) -> FeatureResult:
        """
        Execute a single feature.

        Args:
            feature_id: Which feature to run.
            results: Analysis results to process.
            **kwargs: Feature-specific parameters.

        Returns:
            FeatureResult wrapping the feature output.

        Raises:
            KeyError: If feature_id is not registered.
            FeatureDisabledError: If feature is toggled off.
            EntitlementError: If user is not entitled.
        """
        feature = self._get_feature(feature_id)
        return feature.execute(results, **kwargs)

    def estimate(
        self,
        feature_id: str,
        results: Union[AnalysisResult, List[AnalysisResult]],
    ) -> CostEstimate:
        """
        Get cost/time estimate before executing a feature.

        Args:
            feature_id: Which feature to estimate.
            results: Analysis results that would be processed.

        Returns:
            CostEstimate with token count, cost, and time estimates.
        """
        feature = self._get_feature(feature_id)
        return feature.estimate_cost(results)

    def estimate_all(
        self,
        results: Union[AnalysisResult, List[AnalysisResult]],
        feature_ids: Optional[List[str]] = None,
    ) -> List[CostEstimate]:
        """
        Estimate costs for multiple features at once.

        Args:
            results: Analysis results to estimate for.
            feature_ids: Specific features to estimate. If None,
                         estimates all available features.

        Returns:
            List of CostEstimate objects.
        """
        ids = feature_ids or list(self._features.keys())
        estimates = []
        for fid in ids:
            if self._gate.is_available(fid):
                estimates.append(self.estimate(fid, results))
        return estimates

    def execute_available_single(
        self,
        result: AnalysisResult,
        **kwargs: Any,
    ) -> Dict[str, FeatureResult]:
        """
        Run all available single-file features on one result.

        Args:
            result: Single AnalysisResult.
            **kwargs: Passed to each feature.

        Returns:
            Dict mapping feature_id to FeatureResult for successes.
        """
        outputs: Dict[str, FeatureResult] = {}
        for fid, feature in self._features.items():
            if feature.feature_type in ("single", "single+batch"):
                if self._gate.is_available(fid):
                    try:
                        outputs[fid] = feature.execute(result, **kwargs)
                    except Exception as e:
                        self.logger.warning(
                            f"Feature '{fid}' failed on single result: {e}"
                        )
        return outputs

    def execute_available_batch(
        self,
        results: List[AnalysisResult],
        **kwargs: Any,
    ) -> Dict[str, FeatureResult]:
        """
        Run all available batch features on a list of results.

        Args:
            results: List of AnalysisResult objects.
            **kwargs: Passed to each feature.

        Returns:
            Dict mapping feature_id to FeatureResult for successes.
        """
        outputs: Dict[str, FeatureResult] = {}
        for fid, feature in self._features.items():
            if feature.feature_type in ("batch", "single+batch"):
                if self._gate.is_available(fid):
                    try:
                        outputs[fid] = feature.execute(results, **kwargs)
                    except Exception as e:
                        self.logger.warning(
                            f"Feature '{fid}' failed on batch: {e}"
                        )
        return outputs

    def list_features(self) -> List[Dict[str, Any]]:
        """
        List all features with availability status.

        Returns:
            List of dicts with id, name, type, version, available, enabled.
        """
        return [
            {
                "id": f.feature_id,
                "name": f.display_name,
                "type": f.feature_type,
                "version": f.version,
                "available": self._gate.is_available(f.feature_id),
                "enabled": self._gate.is_enabled(f.feature_id),
            }
            for f in self._features.values()
        ]

    def get_feature(self, feature_id: str) -> Optional[BaseLLMFeature]:
        """Get a feature by ID, or None if not registered."""
        return self._features.get(feature_id)

    def _get_feature(self, feature_id: str) -> BaseLLMFeature:
        """Get a feature by ID, raising KeyError if not found."""
        if feature_id not in self._features:
            available = ", ".join(sorted(self._features.keys()))
            raise KeyError(
                f"Unknown feature '{feature_id}'. Available: {available}"
            )
        return self._features[feature_id]
