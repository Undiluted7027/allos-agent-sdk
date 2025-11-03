# tests/integration/test_openai_real.py


import pytest

from allos.providers import Message, MessageRole, ProviderRegistry
from allos.tools.base import BaseTool, ToolParameter


# --- Test Case 2: Tool Definition ---
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


@pytest.mark.integration
def test_openai_provider_simple_chat_integration():
    """
    Performs a real API call to test simple chat completion.
    This test is skipped unless --run-integration is provided and OPENAI_API_KEY is set.
    """
    provider = ProviderRegistry.get_provider("openai", model="gpt-4o")
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
    assert response.metadata["messages"]["processed"] == 1


@pytest.mark.integration
def test_openai_provider_tool_calling_integration():
    """
    Performs a real API call to test tool calling.
    This test is skipped unless --run-integration is provided and OPENAI_API_KEY is set.
    """
    provider = ProviderRegistry.get_provider("openai", model="gpt-4o")
    messages = [
        Message(role=MessageRole.USER, content="What is the weather like in Boston?"),
    ]
    tools = [GetWeatherTool()]

    response = provider.chat(messages, tools=tools)

    assert (
        response.content is None
    ), "Expected no direct text content when a tool is called"
    assert len(response.tool_calls) > 0, "Expected the model to request a tool call"

    tool_call = response.tool_calls[0]
    assert tool_call.name == "get_current_weather"
    assert "location" in tool_call.arguments
    assert "boston" in tool_call.arguments["location"].lower()
    assert response.metadata["tool_calls"]["processed"] == 1
