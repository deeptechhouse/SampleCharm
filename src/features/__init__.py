"""
LLM-powered feature expansion for SampleCharm.

This module provides 10 toggleable, paywall-ready LLM features that consume
AnalysisResult objects produced by the core analysis pipeline.

Architecture:
    Core Pipeline → AnalysisResult → LLMFeatureManager → Feature Implementations
                                          ↑
                                     FeatureGate (toggle + entitlement check)
"""

from src.features.gate import (
    EntitlementProvider,
    AlwaysEntitled,
    FeatureGate,
    FeatureDisabledError,
    EntitlementError,
)
from src.features.client import LLMClient
from src.features.protocols import LLMFeature, BaseLLMFeature
from src.features.models import FeatureResult, CostEstimate
from src.features.manager import LLMFeatureManager
from src.features.estimator import AnalysisTimeEstimator

__all__ = [
    "EntitlementProvider",
    "AlwaysEntitled",
    "FeatureGate",
    "FeatureDisabledError",
    "EntitlementError",
    "LLMClient",
    "LLMFeature",
    "BaseLLMFeature",
    "FeatureResult",
    "CostEstimate",
    "LLMFeatureManager",
    "AnalysisTimeEstimator",
]
