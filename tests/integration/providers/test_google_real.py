# tests/integration/test_google_real.py

import sys

import pytest

if sys.version_info < (3, 10):
    pytest.skip(
        "Google provider requires Python 3.10+ (google-auth>=2.48.0 dependency). "
        "All Google provider tests skipped on Python 3.9.",
        allow_module_level=True,
    )

from allos.providers import Message, MessageRole, ProviderRegistry
from allos.tools.base import BaseTool, ToolParameter


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
@pytest.mark.requires_gemini
def test_google_gemini_provider_simple_chat_integration():
    """
    Performs a real API call to test simple chat completion.
    This test is skipped unless --run-integration is provided and GEMINI_API_KEY is set.
    """
    provider = ProviderRegistry.get_provider("google", model="gemini-2.5-flash-lite")
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


@pytest.mark.integration
@pytest.mark.requires_vertexai
def test_google_vertexai_provider_simple_chat_integration():
    """
    Performs a real API call to test simple chat completion.
    This test is skipped unless --run-integration is provided and VertexAI is authenticated.
    """
    provider = ProviderRegistry.get_provider(
        "google",
        model="gemini-2.5-flash-lite",
        vertexai=True,
    )
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


@pytest.mark.integration
@pytest.mark.requires_gemini
def test_google_gemini_provider_tool_calling_integration():
    """Peforms a real API call to test tool calling.
    This test is skipped unless --run-integration is provided and GEMINI_API_KEY is set.
    """
    provider = ProviderRegistry.get_provider("google", model="gemini-2.5-flash-lite")
    messages = [
        Message(
            role=MessageRole.USER, content="What is the weather like in Boston, MA?"
        ),
    ]
    tools = [GetWeatherTool()]

    response = provider.chat(messages, tools=tools)
    assert response.content is not None or len(response.tool_calls) > 0, (
        "Expected either text part content or a tool/function call"
    )
    assert len(response.tool_calls) > 0, "Expected the model to request a tool call"

    tool_call = response.tool_calls[0]
    assert tool_call.name == "get_current_weather"
    assert "location" in tool_call.arguments
    assert "boston" in tool_call.arguments["location"].lower()


@pytest.mark.integration
@pytest.mark.requires_vertexai
def test_google_vertex_provider_tool_calling_integration():
    """Performs a real API call to test tool calling.
    This test is skipped unless --run-integration is provided and Vertex AI is authenticated."""
    provider = ProviderRegistry.get_provider(
        "google",
        vertexai=True,
        model="gemini-2.5-flash-lite",
    )

    messages = [
        Message(
            role=MessageRole.USER, content="What is the weather like in Boston, MA?"
        ),
    ]
    tools = [GetWeatherTool()]

    response = provider.chat(messages, tools=tools)
    assert response.content is not None or len(response.tool_calls) > 0, (
        "Expected either text part content or a tool/function call"
    )
    assert len(response.tool_calls) > 0, "Expected the model to request a tool call"

    tool_call = response.tool_calls[0]
    assert tool_call.name == "get_current_weather"
    assert "location" in tool_call.arguments
    assert "boston" in tool_call.arguments["location"].lower()
