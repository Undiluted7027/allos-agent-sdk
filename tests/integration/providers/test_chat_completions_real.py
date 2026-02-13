# tests/integration/providers/test_chat_completions_real.py

import pytest

from allos.providers import Message, MessageRole, ProviderRegistry
from allos.tools.base import BaseTool, ToolParameter

# Reuse the OpenAI marker since this provider requires an OpenAI-compatible key.
# In this specific test, we hit OpenAI's servers to verify the contract.
pytestmark = pytest.mark.requires_openai


class GetWeatherTool(BaseTool):
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


def _assert_chat_completions_metadata(response) -> None:
    """Assert stable metadata shape for real chat_completions calls."""
    metadata = response.metadata
    assert metadata is not None
    assert metadata.model.provider == "chat_completions"
    assert metadata.model.model_id != ""
    assert metadata.usage.input_tokens >= 0
    assert metadata.usage.output_tokens >= 0
    assert metadata.usage.total_tokens == (
        metadata.usage.input_tokens + metadata.usage.output_tokens
    )
    assert metadata.latency.total_duration_ms >= 0


@pytest.mark.integration
def test_chat_completions_real_simple_chat():
    """Test basic chat functionality against real OpenAI endpoint via the adapter."""
    # Note: We use a cheap model for integration tests
    provider = ProviderRegistry.get_provider("chat_completions", model="gpt-3.5-turbo")

    messages = [
        Message(
            role=MessageRole.USER, content="What is 2+2? Answer with just the number."
        )
    ]

    response = provider.chat(messages, temperature=0)
    assert response.content and "4" in response.content
    _assert_chat_completions_metadata(response)


@pytest.mark.integration
def test_chat_completions_real_tool_calling():
    """Test tool calling against real OpenAI endpoint via the adapter."""
    provider = ProviderRegistry.get_provider("chat_completions", model="gpt-3.5-turbo")

    messages = [
        Message(role=MessageRole.USER, content="What is the weather in London?")
    ]
    tools = [GetWeatherTool()]

    response = provider.chat(messages, tools=tools)

    assert len(response.tool_calls) > 0
    assert response.tool_calls[0].name == "get_current_weather"
    assert "London" in response.tool_calls[0].arguments.get("location", "")
    _assert_chat_completions_metadata(response)
