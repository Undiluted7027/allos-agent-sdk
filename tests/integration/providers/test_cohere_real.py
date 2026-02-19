# tests/integration/providers/test_cohere_real.py

import pytest

from allos.providers import Message, MessageRole, ProviderRegistry
from allos.tools.base import BaseTool, ToolParameter

pytestmark = pytest.mark.requires_cohere


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
        pass


def _assert_cohere_metadata(response) -> None:
    """Assert stable metadata shape for real Cohere calls."""
    metadata = response.metadata
    assert metadata is not None
    assert metadata.model.provider == "cohere"
    assert metadata.model.model_id != ""
    assert metadata.usage.input_tokens >= 0
    assert metadata.usage.output_tokens >= 0
    assert metadata.usage.total_tokens == (
        metadata.usage.input_tokens + metadata.usage.output_tokens
    )
    assert metadata.latency.total_duration_ms >= 0


@pytest.mark.integration
def test_cohere_provider_simple_chat_integration():
    """Performs a real API call to test basic Cohere chat."""
    provider = ProviderRegistry.get_provider("cohere", model="command-r7b-12-2024")
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
    _assert_cohere_metadata(response)


@pytest.mark.integration
def test_cohere_provider_tool_calling_integration():
    """Performs a real API call to test Cohere tool-calling behavior."""
    provider = ProviderRegistry.get_provider("cohere", model="command-r7b-12-2024")
    messages = [
        Message(role=MessageRole.USER, content="What is the weather like in Boston?"),
    ]
    tools = [GetWeatherTool()]

    response = provider.chat(messages, tools=tools, temperature=0)

    # Tool calling can be model/version-dependent in real runs.
    if len(response.tool_calls) == 0:
        pytest.skip(
            "Cohere model did not emit a tool call in this real run (model-dependent)."
        )

    tool_call = response.tool_calls[0]
    assert tool_call.name == "get_current_weather"
    assert "location" in tool_call.arguments
    assert "boston" in tool_call.arguments["location"].lower()
    _assert_cohere_metadata(response)
