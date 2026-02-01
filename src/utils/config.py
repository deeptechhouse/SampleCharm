"""
Configuration management for the Audio Sample Analysis Application.

Loads and validates configuration from YAML files with environment
variable interpolation support.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from src.utils.errors import ConfigurationError


class ConfigManager:
    """
    Manages application configuration loaded from YAML files.

    Features:
    - YAML configuration loading
    - Environment variable interpolation (${VAR_NAME})
    - Nested key access with dot notation
    - Default value support
    - Configuration validation
    """

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration manager.

        Args:
            config_dict: Optional pre-loaded configuration dictionary
        """
        self._config: Dict[str, Any] = config_dict or {}
        self._env_pattern = re.compile(r'\$\{([^}]+)\}')

    @classmethod
    def from_file(cls, file_path: Path) -> "ConfigManager":
        """
        Create ConfigManager from YAML file.

        Args:
            file_path: Path to YAML configuration file

        Returns:
            ConfigManager: Initialized with file contents

        Raises:
            ConfigurationError: If file cannot be loaded or parsed
        """
        if not file_path.exists():
            raise ConfigurationError(
                f"Configuration file not found: {file_path}",
                config_key=str(file_path)
            )

        try:
            with open(file_path, 'r') as f:
                config_dict = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigurationError(
                f"Failed to parse YAML configuration: {e}",
                config_key=str(file_path)
            )

        manager = cls(config_dict)
        manager._interpolate_env_vars()
        return manager

    def _interpolate_env_vars(self) -> None:
        """
        Replace ${ENV_VAR} patterns with environment variable values.

        Recursively processes all string values in the configuration.
        """
        self._config = self._interpolate_dict(self._config)

    def _interpolate_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively interpolate environment variables in a dictionary."""
        result = {}
        for key, value in d.items():
            if isinstance(value, dict):
                result[key] = self._interpolate_dict(value)
            elif isinstance(value, list):
                result[key] = self._interpolate_list(value)
            elif isinstance(value, str):
                result[key] = self._interpolate_string(value)
            else:
                result[key] = value
        return result

    def _interpolate_list(self, lst: list) -> list:
        """Recursively interpolate environment variables in a list."""
        result = []
        for item in lst:
            if isinstance(item, dict):
                result.append(self._interpolate_dict(item))
            elif isinstance(item, list):
                result.append(self._interpolate_list(item))
            elif isinstance(item, str):
                result.append(self._interpolate_string(item))
            else:
                result.append(item)
        return result

    def _interpolate_string(self, s: str) -> str:
        """Replace ${ENV_VAR} with environment variable value."""
        def replace(match: re.Match) -> str:
            var_name = match.group(1)
            value = os.environ.get(var_name)
            if value is None:
                return match.group(0)  # Keep original if not found
            return value

        return self._env_pattern.sub(replace, s)

    def get(
        self,
        key: str,
        default: Any = None,
        required: bool = False
    ) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key: Configuration key (supports dot notation: "audio.max_file_size")
            default: Default value if key not found
            required: If True, raise error when key not found

        Returns:
            Configuration value or default

        Raises:
            ConfigurationError: If required key is not found

        Example:
            config.get("audio.max_file_size", default=52428800)
            config.get("openai.api_key", required=True)
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                if required:
                    raise ConfigurationError(
                        f"Required configuration key not found: {key}",
                        config_key=key
                    )
                return default

        return value

    def get_section(self, key: str) -> Dict[str, Any]:
        """
        Get an entire configuration section as a dictionary.

        Args:
            key: Section key (e.g., "audio", "analyzers")

        Returns:
            Dictionary of section values (empty dict if not found)
        """
        value = self.get(key, default={})
        if not isinstance(value, dict):
            return {}
        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value using dot notation.

        Args:
            key: Configuration key (supports dot notation)
            value: Value to set

        Example:
            config.set("audio.max_file_size", 100000000)
        """
        keys = key.split('.')
        current = self._config

        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        current[keys[-1]] = value

    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self._config.copy()

    def validate(self, schema: Dict[str, Any]) -> None:
        """
        Validate configuration against a schema.

        Args:
            schema: Dictionary defining required keys and their types

        Raises:
            ConfigurationError: If validation fails

        Schema format:
            {
                "audio.max_file_size": {"type": int, "required": True},
                "openai.enabled": {"type": bool, "default": False}
            }
        """
        for key, rules in schema.items():
            value = self.get(key)
            required = rules.get("required", False)
            expected_type = rules.get("type")

            if value is None:
                if required:
                    raise ConfigurationError(
                        f"Required configuration missing: {key}",
                        config_key=key
                    )
                continue

            if expected_type and not isinstance(value, expected_type):
                raise ConfigurationError(
                    f"Invalid type for {key}: expected {expected_type.__name__}, "
                    f"got {type(value).__name__}",
                    config_key=key
                )


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from file or return defaults.

    Args:
        config_path: Optional path to config file.
                    If None, tries "config/config.yaml"

    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    if config_path is None:
        # Try default locations
        default_paths = [
            Path("config/config.yaml"),
            Path("config.yaml"),
            Path(__file__).parent.parent.parent / "config" / "config.yaml",
        ]

        for path in default_paths:
            if path.exists():
                config_path = str(path)
                break

    if config_path and Path(config_path).exists():
        manager = ConfigManager.from_file(Path(config_path))
        return manager.to_dict()

    # Return default configuration
    return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """Return default configuration values."""
    return {
        "audio": {
            "supported_formats": [".wav", ".aiff", ".aif", ".mp3", ".flac"],
            "max_file_size": 52428800,  # 50MB
            "max_duration": 30.0,
            "target_sample_rate": 22050,
        },
        "analyzers": {
            "source": {
                "primary": "yamnet",
                "enable_fallback": False,
                "confidence_threshold": 0.75,
            },
            "musical": {
                "primary": "librosa",
                "enable_fallback": False,
                "confidence_threshold": 0.70,
            },
            "percussive": {
                "primary": "random_forest",
                "enable_fallback": False,
                "confidence_threshold": 0.75,
            },
            "rhythmic": {
                "primary": "librosa",
                "enable_fallback": False,
            },
        },
        "openai": {
            "enabled": False,
        },
        "cache": {
            "enabled": True,
            "backend": "memory",
            "max_size": 1000,
            "ttl": 3600,
        },
        "logging": {
            "level": "INFO",
            "format": "json",
        },
        "performance": {
            "max_workers": 4,
        },
    }
