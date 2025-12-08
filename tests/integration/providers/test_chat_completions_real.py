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
