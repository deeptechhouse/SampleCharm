"""
Shared LLM client for all features.

Extracted from src/analyzers/llm/llm_analyzer.py to avoid duplicate client
creation across the existing LLMAnalyzer and all 10 new features.

Supports TogetherAI and OpenAI providers with thread-safe lazy initialization.
"""

import logging
import os
import threading
from typing import Dict, List, Optional

from src.utils.errors import ModelLoadError


# Default models per provider
DEFAULT_MODELS: Dict[str, str] = {
    "togetherai": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "openai": "gpt-4o-mini",
}

# Base URLs per provider
PROVIDER_BASE_URLS: Dict[str, Optional[str]] = {
    "togetherai": "https://api.together.xyz/v1",
    "openai": None,  # OpenAI SDK uses default
}

# Approximate cost per 1K tokens (input + output blended) in USD
TOKEN_COST_PER_1K: Dict[str, float] = {
    "togetherai": 0.0025,
    "openai": 0.0060,
}


class LLMClient:
    """
    Shared, thread-safe, lazy-initialized OpenAI-compatible LLM client.

    Used by both the existing LLMAnalyzer and all new LLM features.
    Wraps provider-specific details behind a clean chat() interface.

    The client is created lazily on first use and reused across all calls.
    Thread-safe via double-checked locking (same pattern as AudioSample.features).
    """

    def __init__(
        self,
        provider: str = "togetherai",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1000,
    ):
        self.provider = provider.lower()
        self.model = model or DEFAULT_MODELS.get(
            self.provider, DEFAULT_MODELS["togetherai"]
        )
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens
        self._api_key = self._resolve_api_key(api_key)
        self._client = None
        self._lock = threading.Lock()
        self.logger = logging.getLogger("features.client")

    @property
    def model_id(self) -> str:
        """Provider/model identifier string."""
        return f"{self.provider}/{self.model}"

    @property
    def cost_per_1k_tokens(self) -> float:
        """Approximate blended cost per 1K tokens for this provider."""
        return TOKEN_COST_PER_1K.get(self.provider, 0.005)

    @property
    def client(self):
        """Thread-safe lazy-initialized OpenAI client."""
        if self._client is None:
            with self._lock:
                if self._client is None:
                    self._client = self._create_client()
        return self._client

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Send a chat completion request.

        Args:
            system_prompt: System role message.
            user_prompt: User role message.
            temperature: Override default temperature.
            max_tokens: Override default max_tokens.

        Returns:
            The assistant's response text.

        Raises:
            AnalysisError: If the API call fails.
        """
        from src.utils.errors import AnalysisError

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature if temperature is not None else self.default_temperature,
                max_tokens=max_tokens or self.default_max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise AnalysisError(
                f"LLM API call failed: {e}",
                analyzer_name="llm_client",
                original_error=e,
            ) from e

    def estimate_tokens(self, text: str) -> int:
        """
        Rough token count estimate (4 chars per token heuristic).

        Args:
            text: Input text to estimate.

        Returns:
            Approximate token count.
        """
        return max(1, len(text) // 4)

    def estimate_cost(self, input_text: str, output_tokens: int = 500) -> float:
        """
        Estimate cost in USD for a single call.

        Args:
            input_text: The combined prompt text.
            output_tokens: Expected output token count.

        Returns:
            Estimated cost in USD.
        """
        input_tokens = self.estimate_tokens(input_text)
        total_tokens = input_tokens + output_tokens
        return (total_tokens / 1000) * self.cost_per_1k_tokens

    def _resolve_api_key(self, api_key: Optional[str]) -> Optional[str]:
        """Resolve API key from parameter, template, or environment."""
        if api_key:
            if api_key.startswith("${") and api_key.endswith("}"):
                var_name = api_key[2:-1]
                return os.environ.get(var_name)
            return api_key

        if self.provider == "togetherai":
            return os.environ.get("TOGETHER_API_KEY") or os.environ.get(
                "TOGETHERAI_API_KEY"
            )
        elif self.provider == "openai":
            return os.environ.get("OPENAI_API_KEY")
        return None

    def _create_client(self):
        """Create the OpenAI-compatible client for the configured provider."""
        if not self._api_key:
            env_vars = (
                "TOGETHER_API_KEY or TOGETHERAI_API_KEY"
                if self.provider == "togetherai"
                else "OPENAI_API_KEY"
            )
            raise ModelLoadError(
                f"No API key found for {self.provider}. Set {env_vars} environment variable.",
                model_name=self.provider,
            )

        try:
            from openai import OpenAI
        except ImportError:
            raise ModelLoadError(
                "openai package required. Install with: pip install openai",
                model_name=self.provider,
            )

        base_url = PROVIDER_BASE_URLS.get(self.provider)
        if base_url:
            return OpenAI(api_key=self._api_key, base_url=base_url)
        return OpenAI(api_key=self._api_key)


def create_llm_client(config: Dict) -> LLMClient:
    """
    Factory function to create LLMClient from config dict.

    Args:
        config: The 'llm' section from config.yaml.

    Returns:
        Configured LLMClient instance.
    """
    return LLMClient(
        provider=config.get("provider", "togetherai"),
        model=config.get("model"),
        api_key=config.get("api_key"),
        temperature=config.get("temperature", 0.3),
        max_tokens=config.get("max_tokens", 1000),
    )
