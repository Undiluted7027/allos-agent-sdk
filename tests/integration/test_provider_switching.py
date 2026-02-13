# tests/integration/test_provider_switching.py


import pytest

from allos.providers import Message, MessageRole, ProviderRegistry
from allos.tools.base import BaseTool, ToolParameter
from tests.conftest import get_available_provider_params


# --- Tool Definition (Shared between tests) ---
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


# --- Test Data (Shared between tests) ---
SIMPLE_CHAT_MESSAGES = [
    Message(
        role=MessageRole.SYSTEM,
        content="You are a test assistant. Your goal is to be concise.",
    ),
    Message(
        role=MessageRole.USER, content="In one word, what is the color of the sky?"
    ),
]

TOOL_CALL_MESSAGES = [
    Message(role=MessageRole.USER, content="What is the weather like in Boston, MA?"),
]

# --- Pytest Parametrization ---
# tuples of (provider_name, model_name)

providers_to_test = get_available_provider_params()

# --- The Tests ---


def _provider_matches(expected: str, actual: str) -> bool:
    if expected == "chat_completions" and actual == "openai":
        return True
    return expected == actual


def _assert_basic_metadata(response, provider_name: str) -> None:
    metadata = response.metadata
    assert metadata is not None
    assert _provider_matches(provider_name, metadata.model.provider)
    assert metadata.model.model_id != ""
    assert metadata.usage.input_tokens >= 0
    assert metadata.usage.output_tokens >= 0
    assert metadata.usage.total_tokens == (
        metadata.usage.input_tokens + metadata.usage.output_tokens
    )
    assert metadata.latency.total_duration_ms >= 0


@pytest.mark.integration
@pytest.mark.parametrize("provider_name, model", providers_to_test)
def test_provider_switching_simple_chat(provider_name, model):
    """
    Tests simple chat across multiple providers to ensure a consistent interface.
    """
    print(f"\n--- Testing Simple Chat on Provider: {provider_name.upper()} ---")

    provider = ProviderRegistry.get_provider(provider_name, model=model)
    response = provider.chat(SIMPLE_CHAT_MESSAGES, temperature=0)

    # Assert that we get a valid response
    assert response.content != "None"
    assert response.content and "blue" in response.content.lower()
    _assert_basic_metadata(response, provider_name)
    print(f"[{provider_name.upper()}] Response: {response.content}")


@pytest.mark.integration
@pytest.mark.parametrize("provider_name, model", providers_to_test)
def test_provider_switching_tool_calling(provider_name, model):
    """
    Tests tool calling across multiple providers to ensure a consistent interface.
    """
    print(f"\n--- Testing Tool Calling on Provider: {provider_name.upper()} ---")

    tools = [GetWeatherTool()]

    provider = ProviderRegistry.get_provider(provider_name, model=model)
    response = provider.chat(TOOL_CALL_MESSAGES, tools=tools)

    # Assert that a valid tool call was requested
    assert len(response.tool_calls) > 0, (
        f"Expected {provider_name} to request a tool call"
    )

    tool_call = response.tool_calls[0]
    assert tool_call.name == "get_current_weather"
    assert "location" in tool_call.arguments
    assert "boston" in tool_call.arguments["location"].lower()
    _assert_basic_metadata(response, provider_name)
    print(
        f"[{provider_name.upper()}] Tool Call: {tool_call.name}({tool_call.arguments})"
    )
