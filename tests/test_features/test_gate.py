"""Tests for FeatureGate, EntitlementProvider, and error classes."""

import pytest

from src.features.gate import (
    AlwaysEntitled,
    EntitlementError,
    FeatureDisabledError,
    FeatureGate,
)


class TestFeatureDisabledError:
    def test_message_contains_feature_id(self):
        err = FeatureDisabledError("production_notes")
        assert "production_notes" in str(err)
        assert err.feature_id == "production_notes"


class TestEntitlementError:
    def test_message_contains_ids(self):
        err = EntitlementError("batch_rename", user_id="user42")
        assert "batch_rename" in str(err)
        assert "user42" in str(err)
        assert err.feature_id == "batch_rename"
        assert err.user_id == "user42"

    def test_default_user_id(self):
        err = EntitlementError("batch_rename")
        assert err.user_id == "anonymous"


class TestAlwaysEntitled:
    def test_always_returns_true(self):
        provider = AlwaysEntitled()
        assert provider.is_entitled("any_user", "any_feature") is True


class NeverEntitled:
    """Test stub: denies all entitlements."""

    def is_entitled(self, user_id: str, feature_id: str) -> bool:
        return False


class TestFeatureGate:
    def test_check_passes_when_enabled_and_entitled(self, all_enabled_gate):
        # Should not raise
        all_enabled_gate.check("production_notes")

    def test_check_raises_when_disabled(self):
        config = {"production_notes": {"enabled": False}}
        gate = FeatureGate(config=config)
        with pytest.raises(FeatureDisabledError):
            gate.check("production_notes")

    def test_check_raises_when_missing_from_config(self):
        gate = FeatureGate(config={})
        with pytest.raises(FeatureDisabledError):
            gate.check("production_notes")

    def test_check_raises_when_not_entitled(self):
        config = {"production_notes": {"enabled": True}}
        gate = FeatureGate(config=config, entitlement=NeverEntitled())
        with pytest.raises(EntitlementError):
            gate.check("production_notes")

    def test_is_available_true(self, all_enabled_gate):
        assert all_enabled_gate.is_available("production_notes") is True

    def test_is_available_false_when_disabled(self):
        config = {"production_notes": {"enabled": False}}
        gate = FeatureGate(config=config)
        assert gate.is_available("production_notes") is False

    def test_is_enabled_ignores_entitlement(self):
        config = {"production_notes": {"enabled": True}}
        gate = FeatureGate(config=config, entitlement=NeverEntitled())
        assert gate.is_enabled("production_notes") is True

    def test_list_available(self, all_enabled_gate):
        available = all_enabled_gate.list_available()
        assert "production_notes" in available
        assert "batch_rename" in available

    def test_set_enabled_toggles_feature(self, all_enabled_gate):
        all_enabled_gate.set_enabled("production_notes", False)
        assert all_enabled_gate.is_enabled("production_notes") is False

        all_enabled_gate.set_enabled("production_notes", True)
        assert all_enabled_gate.is_enabled("production_notes") is True

    def test_set_enabled_creates_config_entry(self):
        gate = FeatureGate(config={})
        gate.set_enabled("new_feature", True)
        assert gate.is_enabled("new_feature") is True

    def test_user_id_from_config(self):
        gate = FeatureGate(config={"user_id": "test42"})
        assert gate.user_id == "test42"

    def test_user_id_defaults_to_anonymous(self):
        gate = FeatureGate(config={})
        assert gate.user_id == "anonymous"
