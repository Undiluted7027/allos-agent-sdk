# tests/unit/test_anthropic_provider.py

from unittest.mock import MagicMock, patch

import anthropic
import pytest

from allos.providers.anthropic import AnthropicProvider
from allos.providers.base import Message, MessageRole, ToolCall
from allos.tools.base import BaseTool, ToolParameter
from allos.utils.errors import ProviderError


class DummyTool(BaseTool):
    name = "get_weather"
    description = "Gets the weather"
    parameters = [
        ToolParameter(
            name="location",
            type="string",
            description="location of city to get weather from",
            required=True,
        )
    ]

    def execute(self, **kwargs):
        pass


@patch("allos.providers.anthropic.anthropic.Anthropic")
def test_provider_initialization(MockAnthropic):
    """Test that the Anthropic client is initialized correctly."""
    provider = AnthropicProvider(model="claude-3-5-sonnet-20240620", api_key="test_key")
    MockAnthropic.assert_called_once_with(api_key="test_key")
    assert isinstance(provider, AnthropicProvider)


@patch("allos.providers.anthropic.anthropic.Anthropic")
def test_provider_initialization_failure(MockAnthropic):
    """Test that a failure during Anthropic client initialization is handled."""
    MockAnthropic.side_effect = Exception("Invalid configuration passed to client")
    with pytest.raises(ProviderError) as excinfo:
        AnthropicProvider(model="claude-3-haiku-20240307", api_key="test_key")

    assert "Failed to initialize Anthropic client" in str(excinfo.value)
    assert "Invalid configuration" in str(excinfo.value)


@patch("allos.providers.anthropic.anthropic.Anthropic")
def test_chat_sends_correctly_formatted_messages(MockAnthropic):
    """Test that the chat method correctly converts messages and calls the API."""
    mock_client = MockAnthropic.return_value
    mock_client.messages.create.return_value = (
        MagicMock()
    )  # Return value doesn't matter here

    provider = AnthropicProvider(model="claude-3-5-sonnet-20240620")
    messages = [
        Message(role=MessageRole.SYSTEM, content="System prompt"),
        Message(role=MessageRole.USER, content="Weather in Boston?"),
        Message(
            role=MessageRole.ASSISTANT,
            content="Thinking...",
            tool_calls=[ToolCall("tool_123", "get_weather", {"location": "Boston"})],
        ),
        Message(
            role=MessageRole.TOOL,
            content='{"temperature": "72F"}',
            tool_call_id="tool_123",
        ),
    ]

    provider.chat(messages)

    mock_client.messages.create.assert_called_once()
    call_kwargs = mock_client.messages.create.call_args.kwargs

    # Assert system prompt is handled correctly
    assert call_kwargs["system"] == "System prompt"

    # Assert message history is converted correctly
    expected_messages = [
        {"role": "user", "content": "Weather in Boston?"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Thinking..."},
                {
                    "type": "tool_use",
                    "id": "tool_123",
                    "name": "get_weather",
                    "input": {"location": "Boston"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tool_123",
                    "content": '{"temperature": "72F"}',
                }
            ],
        },
    ]
    assert call_kwargs["messages"] == expected_messages


@patch("allos.providers.anthropic.anthropic.Anthropic")
def test_chat_sends_correctly_formatted_tools(MockAnthropic):
    """Test conversion of Allos tools to Anthropic format via the chat method."""
    mock_client = MockAnthropic.return_value
    mock_client.messages.create.return_value = MagicMock()

    provider = AnthropicProvider(model="claude-3-5-sonnet-20240620")
    tools = [DummyTool()]
    messages = [Message(role=MessageRole.USER, content="Hi")]

    provider.chat(messages, tools=tools)

    mock_client.messages.create.assert_called_once()
    call_kwargs = mock_client.messages.create.call_args.kwargs

    assert "tools" in call_kwargs
    anthropic_tools = call_kwargs["tools"]
    assert len(anthropic_tools) == 1
    tool_def = anthropic_tools[0]
    assert tool_def["name"] == "get_weather"
    assert "input_schema" in tool_def
    assert "location" in tool_def["input_schema"]["properties"]


@patch("allos.providers.anthropic.anthropic.Anthropic")
def test_chat_handles_response_with_no_output(MockAnthropic):
    """
    Test that a response with no 'output' field is handled gracefully.
    """
    mock_client = MockAnthropic.return_value
    mock_api_response = MagicMock()

    # Setup the mock response to have no 'content' attribute, or it could be None
    mock_api_response.content = None
    mock_client.messages.create.return_value = mock_api_response

    provider = AnthropicProvider(model="claude-3-5-sonnet-20240620")

    # The chat should complete successfully without raising an error
    response = provider.chat([Message(role=MessageRole.USER, content="...")])

    # --- ASSERTIONS ---
    # 1. Assert that the response is empty and valid.
    assert response.content is None
    assert len(response.tool_calls) == 0

    # 2. Assert that the metadata correctly reflects that nothing was processed.
    assert response.metadata.usage.total_tokens == 2


@patch("allos.providers.anthropic.anthropic.Anthropic")
def test_chat_parses_simple_response(MockAnthropic):
    """Test parsing a simple text response from Anthropic via the chat method."""
    mock_client = MockAnthropic.return_value
    mock_response = MagicMock()
    mock_response.stop_reason = "end_turn"
    mock_response.content = [MagicMock(type="text", text="Hello there")]
    mock_client.messages.create.return_value = mock_response

    provider = AnthropicProvider(model="claude-3-5-sonnet-20240620")
    response = provider.chat([Message(role=MessageRole.USER, content="Hi")])

    assert response.content == "Hello there"
    assert not response.tool_calls

    assert response.metadata.usage.total_tokens > 0


@patch("allos.providers.anthropic.anthropic.Anthropic")
def test_chat_parses_tool_use_response(MockAnthropic):
    """Test parsing a tool use response from Anthropic via the chat method."""
    mock_client = MockAnthropic.return_value
    mock_response = MagicMock()
    mock_response.stop_reason = "tool_use"
    mock_response.content = [
        MagicMock(type="text", text="Thinking..."),
        MagicMock(
            type="tool_use",
            id="tool_abc",
            name="get_weather",
            input={"location": "SF"},
        ),
    ]
    mock_client.messages.create.return_value = mock_response

    provider = AnthropicProvider(model="claude-3-5-sonnet-20240620")
    response = provider.chat([Message(role=MessageRole.USER, content="...")])

    assert response.content == "Thinking..."
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].id == "tool_abc"
    assert response.tool_calls[0].arguments == {"location": "SF"}

    assert response.metadata.usage.total_tokens > 0


@patch("allos.providers.anthropic.anthropic.Anthropic")
def test_chat_handles_empty_and_skipped_blocks(MockAnthropic):
    """Test parsing a response with empty or invalid blocks."""
    mock_client = MockAnthropic.return_value
    mock_response = MagicMock()
    mock_response.stop_reason = "end_turn"

    # An invalid text block and an invalid tool_use block
    invalid_text_block = MagicMock(type="text", text=None)
    invalid_tool_block = MagicMock(type="tool_use", id=None, name="some_tool", input={})

    mock_response.content = [invalid_text_block, invalid_tool_block]
    mock_client.messages.create.return_value = mock_response

    provider = AnthropicProvider(model="claude-3-5-sonnet-20240620")
    response = provider.chat([Message(role=MessageRole.USER, content="...")])

    assert response.content is None
    assert len(response.tool_calls) == 0

    assert response.metadata.usage.total_tokens >= 0


@pytest.mark.parametrize(
    "anthropic_error, expected_error_message_part",
    [
        (
            anthropic.RateLimitError("Rate limit", response=MagicMock(), body={}),
            "Rate limit exceeded",
        ),
        (
            anthropic.AuthenticationError("Auth error", response=MagicMock(), body={}),
            "Authentication error",
        ),
        (anthropic.APIConnectionError(request=MagicMock()), "Connection error"),
        (
            anthropic.BadRequestError(
                "Bad request",
                response=MagicMock(),
                body={"error": {"message": "invalid param"}},
            ),
            "invalid param",
        ),
        (
            anthropic.InternalServerError(
                "Server error occurred",
                response=MagicMock(status_code=500),
                body={"error": {"message": "The server had an error"}},
            ),
            "Anthropic API error (500): Server error occurred",
        ),
    ],
)
@patch("allos.providers.anthropic.anthropic.Anthropic")
def test_chat_error_handling(
    MockAnthropic, anthropic_error, expected_error_message_part
):
    """Test that Anthropic API errors are caught and wrapped in ProviderError."""
    mock_client = MockAnthropic.return_value
    mock_client.messages.create.side_effect = anthropic_error

    provider = AnthropicProvider(model="claude-3-5-sonnet-20240620")
    with pytest.raises(ProviderError) as excinfo:
        provider.chat([Message(role=MessageRole.USER, content="Hi")])

    assert expected_error_message_part in str(excinfo.value)


@pytest.mark.parametrize(
    "model_name, expected_window",
    [
        ("claude-3-opus-20240229", 200000),
        ("claude-3-sonnet-20240229", 200000),
        ("claude-3-haiku-20240307", 200000),
        ("unknown-model", 4096),  # Test fallback
    ],
)
@patch("allos.providers.anthropic.anthropic.Anthropic")
def test_get_context_window(MockAnthropic, model_name, expected_window):
    """Test that get_context_window returns correct size for known and unknown models."""
    # This test doesn't need to mock the Anthropic client
    provider = AnthropicProvider(model=model_name)
    assert provider.get_context_window() == expected_window
