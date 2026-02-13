"""Shared fixtures and helpers for end-to-end tests."""

import sys
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from allos import Agent, AgentConfig
from allos.providers.base import ProviderResponse, ToolCall
from allos.providers.metadata import MetadataBuilder
from tests.conftest import PROVIDER_MODELS


def _reset_ollama_global_state() -> None:
    """Clear module-level Ollama caches used across provider instances."""
    try:
        from allos.providers.ollama import _MODEL_WARMUP_TRACKER, _OLLAMA_CLIENT_POOL

        _OLLAMA_CLIENT_POOL.clear()
        _MODEL_WARMUP_TRACKER.clear()
    except Exception:
        # If optional Ollama dependency is unavailable in this environment,
        # there is no state to reset for these tests.
        pass


@pytest.fixture(autouse=True)
def clear_ollama_global_state_between_e2e_tests():
    """Prevent cross-test leakage of pooled Ollama clients/warm-up state."""
    _reset_ollama_global_state()
    yield
    _reset_ollama_global_state()


@contextmanager
def mock_provider_environment(provider_name: str):
    """
    Context manager to setup mocks for a specific provider.

    Consolidates provider initialization mocking logic used across all e2e tests.

    Args:
        provider_name: Provider to mock (openai, anthropic, ollama, google)

    Yields:
        Mock object for the provider client (or None for providers that don't need it)
    """
    model_name = PROVIDER_MODELS.get(provider_name, "default")

    if provider_name == "google":
        if sys.version_info < (3, 10):
            yield None
            return
        with patch("allos.providers.google.genai") as mock_genai:
            setup_google_mocks(model_name, mock_genai)
            yield mock_genai
    elif provider_name == "ollama":
        _reset_ollama_global_state()
        with patch("allos.providers.ollama.Client") as mock_ollama_client:
            setup_ollama_mocks(mock_ollama_client)
            yield mock_ollama_client
        _reset_ollama_global_state()
    else:
        # OpenAI, Anthropic, chat_completions - mock the client
        client_path = get_client_patch_path(provider_name)
        with patch(client_path) as mock_client:
            yield mock_client


def get_client_patch_path(provider_name: str) -> str:
    """Get the full patch path for mocking the provider's client."""
    mapping = {
        "openai": "allos.providers.openai.openai.OpenAI",
        "anthropic": "allos.providers.anthropic.anthropic.Anthropic",
        "ollama": "allos.providers.ollama.Client",
        "google": "allos.providers.google.genai.Client",
        "chat_completions": "allos.providers.chat_completions.openai.OpenAI",
    }
    return mapping.get(provider_name, "allos.providers.base.BaseProvider")


def setup_google_mocks(model_name: str, mock_genai):
    """Setup required mocks for Google provider initialization."""
    if sys.version_info < (3, 10):
        return None

    from google.genai import types

    mock_client = MagicMock()
    mock_genai.Client.return_value = mock_client
    mock_client.models.list.return_value = [
        types.Model(name=f"models/{model_name}", input_token_limit=1048576)
    ]
    return mock_client


def setup_ollama_mocks(mock_ollama_client):
    """Setup required mocks for Ollama provider initialization."""
    from ollama._types import ListResponse, ShowResponse

    mock_client_instance = MagicMock()
    mock_ollama_client.return_value = mock_client_instance

    mock_client_instance.list.return_value = ListResponse(
        models=[
            ListResponse.Model(model=PROVIDER_MODELS.get("ollama", "qwen2.5:7b")),
            ListResponse.Model(model="llama3.1:latest"),
            ListResponse.Model(model="mistral:latest"),
        ]
    )

    mock_show_response = MagicMock(spec=ShowResponse)
    mock_show_response.capabilities = {"tools": True}
    mock_show_response.modelinfo = {"num_ctx": 32768}
    mock_client_instance.show.return_value = mock_show_response

    return mock_client_instance


def create_test_metadata_with_builder(
    provider_name: str,
    model_id: str,
    input_tokens: int,
    output_tokens: int,
    provider_specific_fields: Optional[Dict[str, Any]] = None,
):
    """
    Create realistic test metadata using MetadataBuilder.

    Consolidates metadata creation logic used across all e2e tests.

    Args:
        provider_name: Provider name (openai, anthropic, ollama, google)
        model_id: Model identifier
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        provider_specific_fields: Optional dict for provider-specific metadata
                                  e.g., {"openai": {"system_fingerprint": "fp_123"}}

    Returns:
        Metadata object built using MetadataBuilder
    """
    # Create synthetic response object with fields MetadataBuilder expects
    synthetic_response = {
        "id": f"resp_{provider_name}_{int(time.time())}",
        "model": model_id,
        "status": "success",
        "usage": type(
            "Usage",
            (object,),
            {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        )(),
        "choices": [
            type(
                "Choice",
                (object,),
                {
                    "finish_reason": "stop",
                    "message": type("Message", (object,), {"content": "test"})(),
                },
            )()
        ],
    }

    # Build metadata
    start_time = time.time() - 1.0  # Simulate 1 second ago
    builder = MetadataBuilder(
        provider_name=provider_name,
        request_kwargs={"model": model_id},
        start_time=start_time,
    )

    builder_instance = builder.with_response_obj(
        type("obj", (object,), synthetic_response)()
    )

    # Add provider-specific fields if provided
    if provider_specific_fields:
        builder_instance = builder_instance.with_provider_specific(
            **provider_specific_fields
        )

    return builder_instance.build()


def get_provider_specific_fields(
    provider_name: str, variant: str = "default"
) -> Optional[Dict[str, Any]]:
    """
    Generate provider-specific metadata fields for testing.

    Centralizes provider-specific field generation to reduce complexity
    and duplication across tests.

    Args:
        provider_name: Provider name (openai, anthropic, ollama, google)
        variant: Variant identifier for different test scenarios

    Returns:
        Dictionary suitable for provider_specific_fields parameter,
        or None if provider has no specific fields
    """
    if provider_name == "openai":
        return {"openai": {"system_fingerprint": f"fp_{variant}"}}
    elif provider_name == "ollama":
        return {
            "ollama": {
                "warm_up": True,
                "warm_up_duration_seconds": 2.5 if variant == "default" else 3.2,
            }
        }
    elif provider_name == "google":
        return {
            "google": {
                "used_thought_signatures": variant != "no_thought",
                "vertexai": False,
            }
        }
    return None


def create_mock_response_simple(
    provider_name: str,
    model_id: str,
    input_tokens: int,
    output_tokens: int,
    content: Optional[str],
    tool_calls: Optional[List[ToolCall]] = None,
    provider_specific_fields: Optional[Dict[str, Any]] = None,
) -> ProviderResponse:
    """
    Create a simple mock ProviderResponse for testing.

    Args:
        provider_name: Provider name
        model_id: Model identifier
        input_tokens: Input token count
        output_tokens: Output token count
        content: Response content (None for tool calls)
        tool_calls: List of tool calls (empty list if none)
        provider_specific_fields: Optional provider-specific metadata

    Returns:
        ProviderResponse with populated metadata
    """
    metadata = create_test_metadata_with_builder(
        provider_name=provider_name,
        model_id=model_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        provider_specific_fields=provider_specific_fields,
    )

    return ProviderResponse(
        content=content,
        tool_calls=tool_calls or [],
        metadata=metadata,
    )


class MultiTurnResponseMocker:
    """
    Helper class to manage multi-turn mock responses.

    Reduces test complexity by encapsulating turn counting and conditional logic.
    """

    def __init__(self, provider_name: str, model_id: str, turns_config: List[Dict]):
        """
        Initialize multi-turn mocker.

        Args:
            provider_name: Provider name
            model_id: Model identifier
            turns_config: List of dicts, each containing:
                - input_tokens: int
                - output_tokens: int
                - content: Optional[str]
                - tool_calls: Optional[List[ToolCall]]
                - provider_specific: Optional[Dict] (only for first turn typically)
        """
        self.provider_name = provider_name
        self.model_id = model_id
        self.turns_config = turns_config
        self.call_count = 0

    def __call__(self, messages, **kwargs) -> ProviderResponse:
        """Generate response for current turn."""
        if self.call_count >= len(self.turns_config):
            raise ValueError(f"Unexpected call #{self.call_count + 1}")

        turn_config = self.turns_config[self.call_count]
        self.call_count += 1

        return create_mock_response_simple(
            provider_name=self.provider_name,
            model_id=self.model_id,
            input_tokens=turn_config["input_tokens"],
            output_tokens=turn_config["output_tokens"],
            content=turn_config.get("content"),
            tool_calls=turn_config.get("tool_calls", []),
            provider_specific_fields=turn_config.get("provider_specific"),
        )


def create_agent_with_mock(
    provider_name: str,
    tool_names: List[str],
    mock_response_fn,
) -> Agent:
    """
    Create an agent with mocked provider chat method.

    Args:
        provider_name: Provider name
        tool_names: List of tool names to enable
        mock_response_fn: Callable to use as chat mock (or side_effect list)

    Returns:
        Configured Agent instance with mocked provider
    """
    agent = Agent(
        AgentConfig(
            provider_name=provider_name,
            model=PROVIDER_MODELS.get(provider_name, "default"),
            tool_names=tool_names,
            auto_approve=True,
        )
    )
    agent.provider.chat = MagicMock(side_effect=mock_response_fn)  # type: ignore[method-assign]
    return agent
