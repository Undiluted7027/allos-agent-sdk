# tests/integration/providers/test_ollama_real.py

import pytest

from allos.providers import Message, MessageRole, ProviderRegistry
from allos.tools.base import BaseTool, ToolParameter

pytestmark = pytest.mark.requires_ollama


class GetWeatherTool(BaseTool):
    """A dummy tool for testing function calling."""

    name = "get_current_weather"
    description = "Get the current weather in a given location"
    parameters = [
        ToolParameter(
            name="location",
            type="string",
            description="The city and state, e.g. San Francisco, CA",
            required=True,
        )
    ]

    def execute(self, **kwargs):
        # In a real test, we just need the definition, not the execution.
        pass


def _assert_ollama_metadata(response) -> None:
    """Assert stable metadata shape for real Ollama calls."""
    metadata = response.metadata
    assert metadata is not None
    assert metadata.model.provider == "ollama"
    assert metadata.model.model_id != ""
    assert metadata.usage.input_tokens >= 0
    assert metadata.usage.output_tokens >= 0
    assert metadata.usage.total_tokens == (
        metadata.usage.input_tokens + metadata.usage.output_tokens
    )
    assert metadata.latency.total_duration_ms >= 0


@pytest.mark.integration
def test_ollama_provider_simple_chat_integration(default_ollama_model):
    """
    Performs a real API call to local Ollama client (set OLLAMA_HOST
    env var for custom). This test is skipped unless --run-integration
    is provided and Ollama is installed.
    """
    provider = ProviderRegistry.get_provider("ollama", model=default_ollama_model)
    messages = [
        Message(
            role=MessageRole.SYSTEM,
            content="You are a test assistant. Your goal is to be concise.",
        ),
        Message(
            role=MessageRole.USER, content="In one word, what is the color of the sky?"
        ),
    ]

    response = provider.chat(messages, temperature=0)
    assert response.content is not None
    assert "blue" in response.content.lower()
    _assert_ollama_metadata(response)


@pytest.mark.integration
def test_ollama_provider_tool_calling_integration(default_ollama_model):
    """
    Performs a real API call to local Ollama client (set OLLAMA_HOST
    env var for custom) to test tool calling. This test is skipped unless
    --run-integration is provided and Ollama is installed.
    """
    provider = ProviderRegistry.get_provider("ollama", model=default_ollama_model)
    messages = [
        Message(role=MessageRole.USER, content="What is the weather like in Boston?"),
    ]
    tools = [GetWeatherTool()]

    response = provider.chat(messages, tools=tools)
    assert (
        response.content is None or response.content == ""
    ), "Expected no direct text content when a tool is called"
    assert len(response.tool_calls) > 0, "Expected the model to request a tool call"

    tool_call = response.tool_calls[0]
    assert tool_call.name == "get_current_weather"
    assert "location" in tool_call.arguments
    assert "boston" in tool_call.arguments["location"].lower()
    _assert_ollama_metadata(response)
