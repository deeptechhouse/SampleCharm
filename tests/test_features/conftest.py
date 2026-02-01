"""Shared fixtures for LLM feature tests."""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from src.core.models import AnalysisResult
from src.features.client import LLMClient
from src.features.gate import AlwaysEntitled, FeatureGate


# ---------------------------------------------------------------------------
# Feature config with all 10 features enabled
# ---------------------------------------------------------------------------

ALL_ENABLED_CONFIG = {
    "user_id": "test-user",
    "sample_pack_curator": {"enabled": True},
    "natural_language_search": {"enabled": True},
    "daw_suggestions": {"enabled": True},
    "batch_rename": {"enabled": True},
    "production_notes": {"enabled": True},
    "speech_deep_analyzer": {"enabled": True},
    "similar_sample_finder": {"enabled": True},
    "sample_chain": {"enabled": True},
    "marketplace_description": {"enabled": True},
    "anomaly_reporter": {"enabled": True},
}


# ---------------------------------------------------------------------------
# Mock LLM Client
# ---------------------------------------------------------------------------


class MockLLMClient:
    """Mock LLMClient that returns canned responses without API calls."""

    def __init__(self, response: str = '{"result": "mock"}'):
        self.provider = "mock"
        self.model = "mock-model"
        self._response = response
        self.call_count = 0

    @property
    def model_id(self) -> str:
        return "mock/mock-model"

    @property
    def cost_per_1k_tokens(self) -> float:
        return 0.0

    def chat(self, system_prompt, user_prompt, temperature=None, max_tokens=None):
        self.call_count += 1
        return self._response

    def estimate_tokens(self, text):
        return max(1, len(text) // 4)

    def estimate_cost(self, input_text, output_tokens=500):
        return 0.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_client():
    """MockLLMClient instance."""
    return MockLLMClient()


@pytest.fixture
def all_enabled_gate():
    """FeatureGate with all features enabled and AlwaysEntitled."""
    return FeatureGate(config=ALL_ENABLED_CONFIG, entitlement=AlwaysEntitled())


@pytest.fixture
def mock_analysis_result():
    """A minimal AnalysisResult for testing features."""
    return AnalysisResult(
        audio_sample_hash="abc123hash",
        timestamp=datetime(2026, 1, 1, 0, 0, 0),
        processing_time=0.5,
        quality_metadata={"sample_rate": 44100, "channels": 2},
        source_classification=None,
        musical_analysis=None,
        percussive_analysis=None,
        rhythmic_analysis=None,
        llm_analysis=None,
    )


@pytest.fixture
def mock_analysis_results(mock_analysis_result):
    """A list of 3 AnalysisResult objects for batch feature testing."""
    results = []
    for i, h in enumerate(["hash_a", "hash_b", "hash_c"]):
        results.append(
            AnalysisResult(
                audio_sample_hash=h,
                timestamp=datetime(2026, 1, 1, 0, 0, i),
                processing_time=0.3 + i * 0.1,
                quality_metadata={"sample_rate": 44100, "channels": 2},
                source_classification=None,
                musical_analysis=None,
                percussive_analysis=None,
                rhythmic_analysis=None,
                llm_analysis=None,
            )
        )
    return results
