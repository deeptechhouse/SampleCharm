"""
Configuration management for the Audio Sample Analysis Application.

=============================================================================
ANNOTATED VERSION - Extensive comments for educational purposes
=============================================================================

This module loads and validates configuration from YAML files with
environment variable interpolation support.

OVERVIEW FOR JUNIOR DEVELOPERS:
-------------------------------
Configuration management is crucial for any production application.
It allows you to:
- Change behavior without modifying code
- Use different settings for dev/test/production
- Keep secrets (API keys) out of source code
- Make deployment flexible

WHY YAML?
- Human-readable (unlike JSON, allows comments)
- Supports complex nested structures
- Well-supported in Python
- Standard format in DevOps (Kubernetes, Docker Compose)

YAML vs JSON vs .env:
- YAML: Complex configurations, nested structures
- JSON: Data interchange, APIs
- .env: Simple key-value secrets

ENVIRONMENT VARIABLE INTERPOLATION:
In config, you might write:
    api_key: ${OPENAI_API_KEY}

This gets replaced with the actual environment variable value.
This keeps secrets out of config files (which may be in git).

CONFIGURATION HIERARCHY (typical):
1. Default values (hardcoded)
2. Config file (config.yaml)
3. Environment variables (override)
4. Command-line arguments (highest priority)

We implement layers 1-3 in this module.
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os        # For environment variable access
import re        # For regex pattern matching (env var interpolation)
from pathlib import Path   # Modern path handling
from typing import Any, Dict, Optional  # Type hints

import yaml      # YAML parsing library (install: pip install pyyaml)

from src.utils.errors import ConfigurationError


# =============================================================================
# CONFIG MANAGER CLASS
# =============================================================================

class ConfigManager:
    """
    Manages application configuration loaded from YAML files.

    Features:
    - YAML configuration loading
    - Environment variable interpolation (${VAR_NAME})
    - Nested key access with dot notation
    - Default value support
    - Configuration validation

    DESIGN PATTERN: Configuration Object
    Instead of scattered global variables or constants, we centralize
    all configuration in one object. This makes it:
    - Easy to pass around (dependency injection)
    - Easy to test (mock configuration)
    - Easy to reload (just create new instance)

    EXAMPLE USAGE:
        # Load from file
        config = ConfigManager.from_file(Path("config/config.yaml"))

        # Access values
        max_size = config.get("audio.max_file_size", default=50000000)
        api_key = config.get("openai.api_key", required=True)

        # Get entire section
        audio_config = config.get_section("audio")
    """

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration manager.

        Args:
            config_dict: Optional pre-loaded configuration dictionary.
                        If None, starts with empty config.

        IMPLEMENTATION NOTE:
        We use a private attribute (_config) to store the data.
        The underscore convention indicates "internal use only".
        External code should use get()/set() methods.
        """
        # Store configuration in internal dictionary
        self._config: Dict[str, Any] = config_dict or {}

        # Compile regex pattern for environment variable substitution
        # Pattern: ${VARIABLE_NAME}
        # - \$ matches literal $
        # - \{ matches literal {
        # - ([^}]+) captures everything until }
        # - \} matches literal }
        self._env_pattern = re.compile(r'\$\{([^}]+)\}')

    @classmethod
    def from_file(cls, file_path: Path) -> "ConfigManager":
        """
        Create ConfigManager from YAML file.

        This is a "factory method" - an alternative constructor that creates
        instances with special initialization logic.

        Args:
            file_path: Path to YAML configuration file

        Returns:
            ConfigManager: Initialized with file contents

        Raises:
            ConfigurationError: If file cannot be loaded or parsed

        WHY CLASSMETHOD?
        Regular __init__ can only return the instance being created.
        A classmethod can do preprocessing and call __init__ with
        the processed data.

        EXAMPLE:
            config = ConfigManager.from_file(Path("config.yaml"))

        HOW YAML LOADING WORKS:
        yaml.safe_load() parses YAML into Python objects:
        - YAML maps {} -> Python dict
        - YAML lists [] -> Python list
        - YAML strings -> Python str
        - YAML numbers -> Python int/float
        - YAML booleans -> Python bool
        - YAML null -> Python None
        """
        # Check file exists
        if not file_path.exists():
            raise ConfigurationError(
                f"Configuration file not found: {file_path}",
                config_key=str(file_path)
            )

        try:
            # Open and parse YAML file
            with open(file_path, 'r') as f:
                # safe_load() is IMPORTANT - it prevents code execution
                # from malicious YAML (yaml.load() can execute arbitrary code!)
                config_dict = yaml.safe_load(f) or {}
                # If file is empty, safe_load returns None, so we default to {}

        except yaml.YAMLError as e:
            # YAML parsing failed (syntax error in file)
            raise ConfigurationError(
                f"Failed to parse YAML configuration: {e}",
                config_key=str(file_path)
            )

        # Create instance with loaded config
        manager = cls(config_dict)

        # Process environment variable substitutions
        manager._interpolate_env_vars()

        return manager

    def _interpolate_env_vars(self) -> None:
        """
        Replace ${ENV_VAR} patterns with environment variable values.

        Recursively processes all string values in the configuration.

        EXAMPLE:
        Before interpolation:
            openai:
              api_key: ${OPENAI_API_KEY}
              model: gpt-4

        After interpolation (if OPENAI_API_KEY="sk-abc123"):
            openai:
              api_key: sk-abc123
              model: gpt-4

        WHY DO THIS?
        Keeping secrets in environment variables is a security best practice:
        - Config files might be committed to git
        - Environment variables are set at deployment time
        - Different environments can have different secrets
        """
        self._config = self._interpolate_dict(self._config)

    def _interpolate_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively interpolate environment variables in a dictionary.

        Args:
            d: Dictionary to process

        Returns:
            New dictionary with interpolated values

        RECURSION EXPLAINED:
        Configuration is nested (dicts within dicts), so we need to
        process each level. For each value:
        - If dict: recursively process it
        - If list: process each item
        - If string: look for ${VAR} patterns
        - Otherwise: leave as-is
        """
        result = {}
        for key, value in d.items():
            if isinstance(value, dict):
                # Nested dictionary - recurse
                result[key] = self._interpolate_dict(value)
            elif isinstance(value, list):
                # List - process each item
                result[key] = self._interpolate_list(value)
            elif isinstance(value, str):
                # String - check for env vars
                result[key] = self._interpolate_string(value)
            else:
                # Numbers, bools, None - keep as-is
                result[key] = value
        return result

    def _interpolate_list(self, lst: list) -> list:
        """
        Recursively interpolate environment variables in a list.

        Lists can contain strings, dicts, or other lists, so we
        need to handle all cases.
        """
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
        """
        Replace ${ENV_VAR} with environment variable value.

        Args:
            s: String potentially containing ${VAR} patterns

        Returns:
            String with patterns replaced by env var values

        HOW REGEX SUBSTITUTION WORKS:
        self._env_pattern.sub(replace_func, string)
        - Finds all matches of the pattern
        - For each match, calls replace_func
        - Replaces the match with the function's return value

        EXAMPLE:
        Input: "Bearer ${API_TOKEN}"
        Pattern matches: "${API_TOKEN}"
        match.group(1) = "API_TOKEN"
        os.environ.get("API_TOKEN") = "abc123"
        Output: "Bearer abc123"
        """
        def replace(match: re.Match) -> str:
            # match.group(0) is the full match: "${VAR_NAME}"
            # match.group(1) is the captured group: "VAR_NAME"
            var_name = match.group(1)

            # Get environment variable value
            value = os.environ.get(var_name)

            if value is None:
                # Variable not set - keep original pattern
                # This allows later processing or shows it's unconfigured
                return match.group(0)

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

        DOT NOTATION EXPLAINED:
        Instead of: config["audio"]["max_file_size"]
        You can use: config.get("audio.max_file_size")

        This is:
        - More concise
        - Handles missing intermediate keys gracefully
        - Works well with deeply nested configs

        EXAMPLE:
            # Get with default
            size = config.get("audio.max_file_size", default=52428800)

            # Get required value (raises if missing)
            api_key = config.get("openai.api_key", required=True)

            # Nested access
            threshold = config.get("analyzers.source.confidence_threshold")
        """
        # Split "audio.max_file_size" into ["audio", "max_file_size"]
        keys = key.split('.')

        # Start with the full config dict
        value = self._config

        # Traverse the nested structure
        for k in keys:
            if isinstance(value, dict) and k in value:
                # Key exists, go deeper
                value = value[k]
            else:
                # Key not found at this level
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

        WHEN TO USE:
        When you need to pass all related config to a component:

            audio_config = config.get_section("audio")
            loader = AudioLoader(**audio_config)

        vs getting each value individually:

            loader = AudioLoader(
                max_file_size=config.get("audio.max_file_size"),
                max_duration=config.get("audio.max_duration"),
                ...
            )
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

        WHEN TO USE:
        - Override config values at runtime
        - Set computed values
        - Testing (inject test configurations)

        EXAMPLE:
            config.set("audio.max_file_size", 100000000)
            config.set("openai.enabled", False)

        NOTE: This modifies the in-memory config, not the file.
        """
        keys = key.split('.')
        current = self._config

        # Navigate/create nested dicts up to the last key
        for k in keys[:-1]:
            if k not in current:
                # Create missing intermediate dict
                current[k] = {}
            current = current[k]

        # Set the final key
        current[keys[-1]] = value

    def to_dict(self) -> Dict[str, Any]:
        """
        Return configuration as dictionary.

        Returns:
            Copy of the configuration dictionary

        WHY RETURN A COPY?
        To prevent external code from modifying our internal state.
        This is called "defensive copying".
        """
        return self._config.copy()

    def validate(self, schema: Dict[str, Any]) -> None:
        """
        Validate configuration against a schema.

        Args:
            schema: Dictionary defining required keys and their types

        Raises:
            ConfigurationError: If validation fails

        SCHEMA FORMAT:
        {
            "audio.max_file_size": {"type": int, "required": True},
            "openai.enabled": {"type": bool, "default": False},
            "openai.api_key": {"type": str, "required": False}
        }

        VALIDATION RULES:
        - "required": True means key must exist
        - "type": specifies expected Python type

        WHY VALIDATE?
        Catch configuration errors at startup rather than runtime.
        It's much better to fail immediately with "missing config X"
        than to fail hours later in the middle of processing.

        EXAMPLE:
            schema = {
                "audio.max_file_size": {"type": int, "required": True},
                "openai.api_key": {"type": str, "required": False},
            }
            config.validate(schema)  # Raises if invalid
        """
        for key, rules in schema.items():
            value = self.get(key)
            required = rules.get("required", False)
            expected_type = rules.get("type")

            # Check required
            if value is None:
                if required:
                    raise ConfigurationError(
                        f"Required configuration missing: {key}",
                        config_key=key
                    )
                continue  # Skip type check for missing optional

            # Check type
            if expected_type and not isinstance(value, expected_type):
                raise ConfigurationError(
                    f"Invalid type for {key}: expected {expected_type.__name__}, "
                    f"got {type(value).__name__}",
                    config_key=key
                )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from file or return defaults.

    This is the main entry point for loading configuration.
    It handles finding the config file and falling back to defaults.

    Args:
        config_path: Optional path to config file.
                    If None, tries standard locations.

    Returns:
        Dict[str, Any]: Configuration dictionary

    SEARCH ORDER (when config_path is None):
    1. config/config.yaml (relative to CWD)
    2. config.yaml (in CWD)
    3. Default to hardcoded values

    EXAMPLE USAGE:
        # Load from default location
        config = load_config()

        # Load from specific file
        config = load_config("my_config.yaml")

        # Use configuration
        engine = create_analysis_engine(config)
    """
    if config_path is None:
        # Try default locations in order
        default_paths = [
            Path("config/config.yaml"),   # Standard location
            Path("config.yaml"),          # Simple location
            # Also try relative to this module
            Path(__file__).parent.parent.parent / "config" / "config.yaml",
        ]

        for path in default_paths:
            if path.exists():
                config_path = str(path)
                break

    if config_path and Path(config_path).exists():
        # File found - load and return
        manager = ConfigManager.from_file(Path(config_path))
        return manager.to_dict()

    # No config file found - return defaults
    return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """
    Return default configuration values.

    This provides sensible defaults when no config file is found.
    These values should work for basic local development.

    DESIGN DECISION: Explicit Defaults
    Rather than having defaults scattered throughout the code,
    we centralize them here. This makes it easy to:
    - See all available options
    - Understand expected values
    - Generate documentation

    IMPORTANT VALUES EXPLAINED:

    target_sample_rate: 22050 Hz
    - Nyquist frequency is 11 kHz, sufficient for most analysis
    - Lower than CD quality (44.1k) to reduce processing time
    - YAMNet uses 16k, so this is a good intermediate

    max_file_size: 52428800 (50 MB)
    - Prevents memory issues with very large files
    - 50 MB is enough for most audio samples

    max_duration: 30.0 seconds
    - Long enough for meaningful analysis
    - Short enough for quick processing

    confidence_threshold: 0.75
    - Balance between using fallback too often (expensive)
      and accepting low-confidence results
    - Can be adjusted based on accuracy requirements
    """
    return {
        # Audio processing settings
        "audio": {
            # Supported audio formats
            "supported_formats": [".wav", ".aiff", ".aif", ".mp3", ".flac"],
            # Maximum file size in bytes (50 MB)
            "max_file_size": 52428800,
            # Maximum audio duration in seconds
            "max_duration": 30.0,
            # Sample rate for analysis (Hz)
            "target_sample_rate": 22050,
        },

        # Analyzer configurations
        "analyzers": {
            # Source classification (what made the sound?)
            "source": {
                "primary": "yamnet",           # Primary analyzer
                "enable_fallback": False,      # OpenAI fallback
                "confidence_threshold": 0.75,  # When to use fallback
            },
            # Musical analysis (pitch, key, tonality)
            "musical": {
                "primary": "librosa",
                "enable_fallback": False,
                "confidence_threshold": 0.70,  # Slightly lower for music
            },
            # Percussion classification (drum type)
            "percussive": {
                "primary": "random_forest",
                "enable_fallback": False,
                "confidence_threshold": 0.75,
            },
            # Rhythmic analysis (tempo, beats)
            "rhythmic": {
                "primary": "librosa",
                "enable_fallback": False,  # Not needed for tempo
            },
        },

        # OpenAI/LLM configuration (disabled by default)
        "openai": {
            "enabled": False,  # Must be explicitly enabled
        },

        # Cache configuration
        "cache": {
            "enabled": True,       # Caching improves performance
            "backend": "memory",   # In-memory cache (vs Redis)
            "max_size": 1000,      # Maximum cached items
            "ttl": 3600,           # Time to live (1 hour)
        },

        # Logging configuration
        "logging": {
            "level": "INFO",   # Default log level
            "format": "json",  # JSON for structured logging
        },

        # Performance settings
        "performance": {
            "max_workers": 4,  # Parallel analyzer workers
        },
    }


# =============================================================================
# USAGE EXAMPLES (for educational purposes)
# =============================================================================

"""
EXAMPLE 1: Basic usage

    from src.utils.config import load_config

    # Load configuration (tries default locations)
    config = load_config()

    # Access values
    max_size = config["audio"]["max_file_size"]
    threshold = config["analyzers"]["source"]["confidence_threshold"]

EXAMPLE 2: Using ConfigManager directly

    from src.utils.config import ConfigManager
    from pathlib import Path

    # Load from specific file
    config = ConfigManager.from_file(Path("my_config.yaml"))

    # Use dot notation
    max_size = config.get("audio.max_file_size", default=50000000)

    # Get required value (raises if missing)
    api_key = config.get("openai.api_key", required=True)

    # Get entire section
    audio_config = config.get_section("audio")

EXAMPLE 3: Environment variable interpolation

    # config.yaml:
    # openai:
    #   api_key: ${OPENAI_API_KEY}
    #   enabled: true

    # Set environment variable:
    # export OPENAI_API_KEY=sk-abc123

    config = load_config("config.yaml")
    print(config["openai"]["api_key"])  # "sk-abc123"

EXAMPLE 4: Configuration validation

    from src.utils.config import ConfigManager

    config = ConfigManager.from_file(Path("config.yaml"))

    schema = {
        "audio.max_file_size": {"type": int, "required": True},
        "audio.supported_formats": {"type": list, "required": True},
        "openai.api_key": {"type": str, "required": False},
    }

    config.validate(schema)  # Raises ConfigurationError if invalid

EXAMPLE 5: Runtime configuration modification

    config = ConfigManager.from_file(Path("config.yaml"))

    # Override for testing
    config.set("openai.enabled", False)
    config.set("cache.backend", "memory")

    # Export modified config
    modified = config.to_dict()
"""
