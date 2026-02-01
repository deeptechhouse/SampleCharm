"""Tests for LLMFeatureManager."""

import pytest

from src.features.gate import FeatureDisabledError, FeatureGate, AlwaysEntitled
from src.features.manager import LLMFeatureManager
from src.features.models import FeatureResult, CostEstimate


class TestLLMFeatureManager:
    def test_registers_all_10_features(self, mock_client, all_enabled_gate):
        manager = LLMFeatureManager(mock_client, all_enabled_gate)
        features = manager.list_features()
        assert len(features) == 10

    def test_list_features_contains_expected_ids(self, mock_client, all_enabled_gate):
        manager = LLMFeatureManager(mock_client, all_enabled_gate)
        ids = {f["id"] for f in manager.list_features()}
        expected = {
            "sample_pack_curator",
            "natural_language_search",
            "daw_suggestions",
            "batch_rename",
            "production_notes",
            "speech_deep_analyzer",
            "similar_sample_finder",
            "sample_chain",
            "marketplace_description",
            "anomaly_reporter",
        }
        assert ids == expected

    def test_list_features_all_available(self, mock_client, all_enabled_gate):
        manager = LLMFeatureManager(mock_client, all_enabled_gate)
        for f in manager.list_features():
            assert f["available"] is True
            assert f["enabled"] is True

    def test_execute_single_feature(
        self, mock_client, all_enabled_gate, mock_analysis_result
    ):
        manager = LLMFeatureManager(mock_client, all_enabled_gate)
        result = manager.execute("production_notes", mock_analysis_result)
        assert isinstance(result, FeatureResult)
        assert result.feature_id == "production_notes"

    def test_execute_batch_feature(
        self, mock_client, all_enabled_gate, mock_analysis_results
    ):
        manager = LLMFeatureManager(mock_client, all_enabled_gate)
        result = manager.execute("sample_pack_curator", mock_analysis_results)
        assert isinstance(result, FeatureResult)
        assert result.feature_id == "sample_pack_curator"

    def test_execute_unknown_feature_raises(
        self, mock_client, all_enabled_gate, mock_analysis_result
    ):
        manager = LLMFeatureManager(mock_client, all_enabled_gate)
        with pytest.raises(KeyError, match="nonexistent"):
            manager.execute("nonexistent", mock_analysis_result)

    def test_execute_disabled_feature_raises(
        self, mock_client, mock_analysis_result
    ):
        config = {"production_notes": {"enabled": False}}
        gate = FeatureGate(config=config)
        manager = LLMFeatureManager(mock_client, gate)
        with pytest.raises(FeatureDisabledError):
            manager.execute("production_notes", mock_analysis_result)

    def test_estimate_returns_cost_estimate(
        self, mock_client, all_enabled_gate, mock_analysis_results
    ):
        manager = LLMFeatureManager(mock_client, all_enabled_gate)
        estimate = manager.estimate("sample_pack_curator", mock_analysis_results)
        assert isinstance(estimate, CostEstimate)
        assert estimate.feature_id == "sample_pack_curator"

    def test_estimate_all_returns_list(
        self, mock_client, all_enabled_gate, mock_analysis_results
    ):
        manager = LLMFeatureManager(mock_client, all_enabled_gate)
        estimates = manager.estimate_all(mock_analysis_results)
        assert len(estimates) == 10
        assert all(isinstance(e, CostEstimate) for e in estimates)

    def test_get_feature_returns_feature(self, mock_client, all_enabled_gate):
        manager = LLMFeatureManager(mock_client, all_enabled_gate)
        feature = manager.get_feature("production_notes")
        assert feature is not None
        assert feature.feature_id == "production_notes"

    def test_get_feature_returns_none_for_unknown(self, mock_client, all_enabled_gate):
        manager = LLMFeatureManager(mock_client, all_enabled_gate)
        assert manager.get_feature("nonexistent") is None

    def test_execute_available_single(
        self, mock_client, all_enabled_gate, mock_analysis_result
    ):
        manager = LLMFeatureManager(mock_client, all_enabled_gate)
        outputs = manager.execute_available_single(mock_analysis_result)
        # Should include single and single+batch features
        assert "production_notes" in outputs
        assert "speech_deep_analyzer" in outputs

    def test_execute_available_batch(
        self, mock_client, all_enabled_gate, mock_analysis_results
    ):
        manager = LLMFeatureManager(mock_client, all_enabled_gate)
        outputs = manager.execute_available_batch(mock_analysis_results)
        # Should include batch and single+batch features
        assert "sample_pack_curator" in outputs
        assert "anomaly_reporter" in outputs
