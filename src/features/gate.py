"""
Feature gating with toggle and entitlement support.

Provides a single checkpoint for all feature access control:
1. Config-based toggle (enabled/disabled per feature)
2. Entitlement check (paywall-ready stub)

The EntitlementProvider protocol is designed for future payment integration.
Swap AlwaysEntitled for a Stripe/payment provider when ready.
"""

import logging
from typing import Dict, List, Optional, Protocol, runtime_checkable

from src.utils.errors import EntitlementError, FeatureDisabledError


@runtime_checkable
class EntitlementProvider(Protocol):
    """
    Protocol for entitlement checking.

    Implement this interface for payment provider integration.
    The default AlwaysEntitled stub grants access to all features.

    Future implementations:
        - StripeEntitled: Check Stripe subscription status
        - APIKeyEntitled: Check API key tier/quota
        - LicenseEntitled: Check local license file
    """

    def is_entitled(self, user_id: str, feature_id: str) -> bool:
        """Check if user has access to a feature."""
        ...


class AlwaysEntitled:
    """Default stub: all users are entitled to all features."""

    def is_entitled(self, user_id: str, feature_id: str) -> bool:
        return True


class FeatureGate:
    """
    Single checkpoint for feature access control.

    Checks two conditions before allowing feature execution:
    1. Is the feature enabled in config? (user toggle)
    2. Is the user entitled? (payment/subscription check)

    Usage:
        gate = FeatureGate(config, entitlement_provider)
        gate.check("production_notes")  # Raises if blocked
        if gate.is_available("production_notes"):  # Non-throwing check
            ...
    """

    def __init__(
        self,
        config: Dict,
        entitlement: Optional[EntitlementProvider] = None,
    ):
        """
        Initialize feature gate.

        Args:
            config: The 'llm_features' section from config.yaml.
                    Each feature_id key maps to {"enabled": bool, ...}.
            entitlement: Provider for entitlement checks. Defaults to
                         AlwaysEntitled (grants access to everything).
        """
        self._config = config
        self._entitlement = entitlement or AlwaysEntitled()
        self._user_id = config.get("user_id", "anonymous")
        self.logger = logging.getLogger("features.gate")

    @property
    def user_id(self) -> str:
        return self._user_id

    def check(self, feature_id: str) -> None:
        """
        Verify feature access. Raises on failure.

        Args:
            feature_id: The feature to check.

        Raises:
            FeatureDisabledError: If feature is toggled off.
            EntitlementError: If user is not entitled.
        """
        feature_config = self._config.get(feature_id, {})
        if not feature_config.get("enabled", False):
            self.logger.debug(f"Feature '{feature_id}' is disabled")
            raise FeatureDisabledError(feature_id)

        if not self._entitlement.is_entitled(self._user_id, feature_id):
            self.logger.debug(
                f"User '{self._user_id}' not entitled to '{feature_id}'"
            )
            raise EntitlementError(feature_id, self._user_id)

    def is_available(self, feature_id: str) -> bool:
        """Non-throwing availability check for UI display."""
        try:
            self.check(feature_id)
            return True
        except (FeatureDisabledError, EntitlementError):
            return False

    def is_enabled(self, feature_id: str) -> bool:
        """Check only the config toggle (ignores entitlement)."""
        feature_config = self._config.get(feature_id, {})
        return feature_config.get("enabled", False)

    def list_available(self) -> List[str]:
        """Return feature_ids that are both enabled and entitled."""
        available = []
        for key, val in self._config.items():
            if isinstance(val, dict) and val.get("enabled", False):
                if self._entitlement.is_entitled(self._user_id, key):
                    available.append(key)
        return available

    def set_enabled(self, feature_id: str, enabled: bool) -> None:
        """Runtime toggle for GUI integration."""
        if feature_id not in self._config:
            self._config[feature_id] = {}
        self._config[feature_id]["enabled"] = enabled
        self.logger.info(
            f"Feature '{feature_id}' {'enabled' if enabled else 'disabled'}"
        )
