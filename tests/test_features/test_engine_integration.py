"""Tests for AudioAnalysisEngine integration with the LLM feature system (Phase B8).

Validates that the engine correctly stores, delegates to, and guards the
optional feature_manager introduced in Phase B4.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.core.engine import AudioAnalysisEngine
from src.features.gate import AlwaysEntitled, FeatureGate
from src.features.manager import LLMFeatureManager
from src.features.models import FeatureResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine(**overrides):
    """Create an AudioAnalysisEngine with all-mock dependencies.

    Any keyword in *overrides* replaces the corresponding constructor arg.
    Returns a tuple (engine, kwargs) so tests can inspect what was passed.
    """
    defaults = dict(
        loader=MagicMock(),
        source_analyzer=MagicMock(),
        musical_analyzer=MagicMock(),
        percussive_analyzer=MagicMock(),
        rhythmic_analyzer=MagicMock(),
        feature_manager=None,
    )
    defaults.update(overrides)
    engine = AudioAnalysisEngine(**defaults)
    return engine


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCreateEngineWithoutFeatures:
    """Engine created without llm_features config has feature_manager = None."""

    def test_feature_manager_is_none(self):
        engine = _make_engine()
        try:
            assert engine.feature_manager is None
        finally:
            engine.shutdown()

    def test_no_feature_manager_key_by_default(self):
        """feature_manager defaults to None when omitted entirely."""
        engine = AudioAnalysisEngine(
            loader=MagicMock(),
            source_analyzer=MagicMock(),
            musical_analyzer=MagicMock(),
            percussive_analyzer=MagicMock(),
            rhythmic_analyzer=MagicMock(),
        )
        try:
            assert engine.feature_manager is None
        finally:
            engine.shutdown()


class TestCreateEngineWithFeatures:
    """Engine created with a feature_manager stores it correctly."""

    def test_engine_stores_feature_manager(self):
        fm = MagicMock(spec=LLMFeatureManager)
        engine = _make_engine(feature_manager=fm)
        try:
            assert engine.feature_manager is fm
        finally:
            engine.shutdown()

    def test_engine_stores_feature_manager_is_not_none(self):
        fm = MagicMock(spec=LLMFeatureManager)
        engine = _make_engine(feature_manager=fm)
        try:
            assert engine.feature_manager is not None
        finally:
            engine.shutdown()


class TestRunFeatureWithoutManager:
    """Calling engine.run_feature() when no feature_manager raises RuntimeError."""

    def test_raises_runtime_error(self):
        engine = _make_engine(feature_manager=None)
        try:
            with pytest.raises(RuntimeError, match="No feature_manager configured"):
                engine.run_feature("production_notes", MagicMock())
        finally:
            engine.shutdown()

    def test_error_message_mentions_create_analysis_engine(self):
        engine = _make_engine(feature_manager=None)
        try:
            with pytest.raises(RuntimeError, match="create_analysis_engine"):
                engine.run_feature("any_feature", MagicMock())
        finally:
            engine.shutdown()


class TestRunFeatureDelegatesToManager:
    """run_feature() calls feature_manager.execute() with correct args."""

    def test_delegates_feature_id_and_results(self):
        fm = MagicMock(spec=LLMFeatureManager)
        fm.execute.return_value = MagicMock(spec=FeatureResult)
        engine = _make_engine(feature_manager=fm)
        try:
            mock_result = MagicMock()
            engine.run_feature("production_notes", mock_result)
            fm.execute.assert_called_once_with("production_notes", mock_result)
        finally:
            engine.shutdown()

    def test_delegates_kwargs(self):
        fm = MagicMock(spec=LLMFeatureManager)
        fm.execute.return_value = MagicMock(spec=FeatureResult)
        engine = _make_engine(feature_manager=fm)
        try:
            mock_result = MagicMock()
            engine.run_feature("production_notes", mock_result, foo="bar", baz=42)
            fm.execute.assert_called_once_with(
                "production_notes", mock_result, foo="bar", baz=42
            )
        finally:
            engine.shutdown()

    def test_returns_feature_result(self):
        expected = MagicMock(spec=FeatureResult)
        fm = MagicMock(spec=LLMFeatureManager)
        fm.execute.return_value = expected
        engine = _make_engine(feature_manager=fm)
        try:
            result = engine.run_feature("production_notes", MagicMock())
            assert result is expected
        finally:
            engine.shutdown()

    def test_propagates_key_error(self):
        """If manager raises KeyError for unknown feature, it propagates."""
        fm = MagicMock(spec=LLMFeatureManager)
        fm.execute.side_effect = KeyError("Unknown feature 'nope'")
        engine = _make_engine(feature_manager=fm)
        try:
            with pytest.raises(KeyError):
                engine.run_feature("nope", MagicMock())
        finally:
            engine.shutdown()

    def test_delegates_list_of_results(self):
        fm = MagicMock(spec=LLMFeatureManager)
        fm.execute.return_value = MagicMock(spec=FeatureResult)
        engine = _make_engine(feature_manager=fm)
        try:
            results_list = [MagicMock(), MagicMock()]
            engine.run_feature("sample_chain", results_list)
            fm.execute.assert_called_once_with("sample_chain", results_list)
        finally:
            engine.shutdown()


class TestEngineFeatureManagerSharedGate:
    """feature_manager gate reflects config toggles."""

    def test_gate_reflects_enabled_toggle(self):
        config = {
            "production_notes": {"enabled": True},
            "batch_rename": {"enabled": False},
        }
        gate = FeatureGate(config=config, entitlement=AlwaysEntitled())

        assert gate.is_enabled("production_notes") is True
        assert gate.is_enabled("batch_rename") is False

    def test_gate_runtime_toggle(self):
        config = {
            "production_notes": {"enabled": False},
        }
        gate = FeatureGate(config=config, entitlement=AlwaysEntitled())

        assert gate.is_enabled("production_notes") is False
        gate.set_enabled("production_notes", True)
        assert gate.is_enabled("production_notes") is True

    def test_gate_availability_includes_entitlement(self):
        config = {
            "production_notes": {"enabled": True},
        }
        gate = FeatureGate(config=config, entitlement=AlwaysEntitled())

        assert gate.is_available("production_notes") is True

    def test_gate_disabled_feature_not_available(self):
        config = {
            "production_notes": {"enabled": False},
        }
        gate = FeatureGate(config=config, entitlement=AlwaysEntitled())

        assert gate.is_available("production_notes") is False

    def test_gate_unknown_feature_not_enabled(self):
        config = {}
        gate = FeatureGate(config=config, entitlement=AlwaysEntitled())

        assert gate.is_enabled("nonexistent_feature") is False
        assert gate.is_available("nonexistent_feature") is False
