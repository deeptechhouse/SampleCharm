"""Tests for LLMClient."""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.features.client import LLMClient, DEFAULT_MODELS, TOKEN_COST_PER_1K, create_llm_client


class TestLLMClientInit:
    def test_default_provider(self):
        client = LLMClient(api_key="test-key")
        assert client.provider == "togetherai"
        assert client.model == DEFAULT_MODELS["togetherai"]

    def test_openai_provider(self):
        client = LLMClient(provider="openai", api_key="test-key")
        assert client.provider == "openai"
        assert client.model == DEFAULT_MODELS["openai"]

    def test_custom_model(self):
        client = LLMClient(model="custom/model", api_key="test-key")
        assert client.model == "custom/model"

    def test_model_id_property(self):
        client = LLMClient(api_key="test-key")
        assert client.model_id == f"togetherai/{DEFAULT_MODELS['togetherai']}"

    def test_cost_per_1k_tokens(self):
        client = LLMClient(api_key="test-key")
        assert client.cost_per_1k_tokens == TOKEN_COST_PER_1K["togetherai"]


class TestLLMClientTokenEstimation:
    def test_estimate_tokens(self):
        client = LLMClient(api_key="test-key")
        assert client.estimate_tokens("hello world") >= 1

    def test_estimate_tokens_minimum_one(self):
        client = LLMClient(api_key="test-key")
        assert client.estimate_tokens("") == 1

    def test_estimate_cost(self):
        client = LLMClient(api_key="test-key")
        cost = client.estimate_cost("a" * 4000, output_tokens=500)
        assert cost > 0.0


class TestLLMClientAPIKeyResolution:
    def test_direct_api_key(self):
        client = LLMClient(api_key="direct-key")
        assert client._api_key == "direct-key"

    def test_template_api_key(self):
        with patch.dict(os.environ, {"MY_KEY": "resolved-key"}):
            client = LLMClient(api_key="${MY_KEY}")
            assert client._api_key == "resolved-key"

    def test_env_togetherai_key(self):
        with patch.dict(os.environ, {"TOGETHER_API_KEY": "env-key"}, clear=False):
            client = LLMClient(provider="togetherai")
            assert client._api_key == "env-key"

    def test_env_openai_key(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "oai-key"}, clear=False):
            client = LLMClient(provider="openai")
            assert client._api_key == "oai-key"


class TestCreateLLMClient:
    def test_factory_creates_client(self):
        config = {"provider": "openai", "api_key": "factory-key", "temperature": 0.5}
        client = create_llm_client(config)
        assert isinstance(client, LLMClient)
        assert client.provider == "openai"
        assert client.default_temperature == 0.5
