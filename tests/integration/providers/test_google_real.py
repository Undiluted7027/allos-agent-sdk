# tests/integration/test_google_real.py

import pytest

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
def test_google_provider_simple_chat_integration():
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
@pytest.mark.requires_vertex
def test_google_vertexai_provider_simple_chat_integration():
    """
    Performs a real API call to test simple chat completion.
    This test is skipped unless --run-integration is provided and VertexAI is authenticated.
    """
    provider = ProviderRegistry.get_provider(
        "google", model="gemini-2.5-flash-lite", vertexai=True
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
