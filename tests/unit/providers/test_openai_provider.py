# tests/unit/providers/test_openai_provider.py

from typing import Any
from unittest.mock import MagicMock, patch

import openai
import pytest

from allos.providers.base import Message, MessageRole, ToolCall
from allos.providers.openai import OpenAIProvider
from allos.tools.base import BaseTool, ToolParameter
from allos.utils.errors import ProviderError


class DummyTool(BaseTool):
    name = "get_weather"
    description = "Gets the weather"
    parameters = [
        ToolParameter(
            name="location", type="string", description="The city", required=True
        )
    ]

    def execute(self, **kwargs: Any):
        return {"weather": "sunny"}


# We patch the client at the location where it is imported
@patch("allos.providers.openai.openai.OpenAI")
def test_provider_initialization(MockOpenAI):
    """Test that the OpenAI client is initialized correctly."""
    provider = OpenAIProvider(model="gpt-4o", api_key="test_key")
    MockOpenAI.assert_called_once_with(api_key="test_key")
    assert isinstance(provider, OpenAIProvider)


@patch("allos.providers.openai.openai.OpenAI")
def test_provider_initialization_failure(MockOpenAI):
    """Test that a failure during OpenAI client initialization is handled."""
    MockOpenAI.side_effect = Exception("Invalid configuration passed to client")
    with pytest.raises(ProviderError) as excinfo:
        OpenAIProvider(model="gpt-4o", api_key="test_key")

    assert "Failed to initialize OpenAI client" in str(excinfo.value)
    assert "Invalid configuration" in str(excinfo.value)


@patch("allos.providers.openai.openai.OpenAI")
def test_chat_sends_correctly_formatted_messages(MockOpenAI):
    """
    Test that the chat method correctly converts messages and calls the API
    with the right instructions and input.
    """
    mock_client = MockOpenAI.return_value
    mock_client.responses.create.return_value = (
        MagicMock()
    )  # We only care about the call args

    provider = OpenAIProvider(model="gpt-4o")
    messages = [
        Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
        Message(role=MessageRole.USER, content="Hello"),
        Message(
            role=MessageRole.TOOL, content='{"status": "ok"}', tool_call_id="call_123"
        ),
    ]

    provider.chat(messages)

    mock_client.responses.create.assert_called_once()
    # Inspect the keyword arguments passed to the mocked create method
    call_kwargs = mock_client.responses.create.call_args.kwargs

    assert call_kwargs["instructions"] == "You are a helpful assistant."
    assert call_kwargs["input"] == [
        {"role": "user", "content": "Hello"},
        {
            "type": "function_call_output",
            "id": "fco_call_123",
            "call_id": "call_123",
            "status": "completed",
            "output": '{"status": "ok"}',
        },
    ]


@patch("allos.providers.openai.openai.OpenAI")
def test_chat_sends_correctly_formatted_tools(MockOpenAI):
    """
    Test that the chat method correctly converts tools and calls the API
    with the right tools definition.
    """
    mock_client = MockOpenAI.return_value
    mock_client.responses.create.return_value = MagicMock()

    provider = OpenAIProvider(model="gpt-4o")
    tools = [DummyTool()]
    messages = [Message(role=MessageRole.USER, content="Weather?")]

    provider.chat(messages, tools=tools)

    mock_client.responses.create.assert_called_once()
    call_kwargs = mock_client.responses.create.call_args.kwargs

    assert "tools" in call_kwargs
    openai_tools = call_kwargs["tools"]
    assert len(openai_tools) == 1
    tool_def = openai_tools[0]
    assert tool_def["type"] == "function"
    assert tool_def["name"] == "get_weather"
    assert "location" in tool_def["parameters"]["properties"]


@patch("allos.providers.openai.openai.OpenAI")
def test_chat_parses_simple_response(MockOpenAI):
    """Test that a simple text-based chat response is parsed correctly."""
    mock_client = MockOpenAI.return_value
    mock_response = MagicMock()
    mock_response.output = [
        MagicMock(
            type="message",
            content=[MagicMock(type="output_text", text="Hello, world!")],
        )
    ]
    mock_client.responses.create.return_value = mock_response

    provider = OpenAIProvider(model="gpt-4o")
    response = provider.chat([Message(role=MessageRole.USER, content="Hi")])

    assert response.content == "Hello, world!"
    assert not response.tool_calls


@patch("allos.providers.openai.openai.OpenAI")
def test_chat_parses_tool_call_response(MockOpenAI):
    """Test that a response requesting a tool call is parsed correctly."""
    mock_client = MockOpenAI.return_value
    mock_response = MagicMock()

    mock_tool_call_item = MagicMock()
    mock_tool_call_item.type = "function_call"
    mock_tool_call_item.id = "fc_abc"  # The item's own ID
    mock_tool_call_item.call_id = "call_abc"  # The correlation ID
    mock_tool_call_item.name = "get_weather"  # This now sets the attribute correctly
    mock_tool_call_item.arguments = '{"location": "Boston"}'

    mock_response.output = [mock_tool_call_item]
    mock_client.responses.create.return_value = mock_response

    provider = OpenAIProvider(model="gpt-4o")
    response = provider.chat([Message(role=MessageRole.USER, content="Weather?")])

    assert response.content is None
    assert len(response.tool_calls) == 1
    tool_call = response.tool_calls[0]
    assert tool_call.id == "call_abc"
    assert tool_call.name == "get_weather"
    assert tool_call.arguments == {"location": "Boston"}


@patch("allos.providers.openai.openai.OpenAI")
def test_chat_raises_error_on_malformed_tool_call_json(MockOpenAI):
    """
    Test that a ProviderError is raised when tool call arguments are malformed JSON.
    """
    mock_client = MockOpenAI.return_value
    mock_api_response = MagicMock()

    # Setup the mock to match the OpenAI v1+ client response structure
    mock_tool_call_item = MagicMock()
    mock_tool_call_item.type = "function_call"
    mock_tool_call_item.id = "call_123"
    mock_tool_call_item.name = "some_tool"
    mock_tool_call_item.arguments = (
        '{"key": "value", "unterminated string}'  # Malformed JSON
    )

    mock_api_response.output = [mock_tool_call_item]
    mock_client.responses.create.return_value = mock_api_response

    provider = OpenAIProvider(model="gpt-4o")

    # Assert that a ProviderError is raised
    with pytest.raises(ProviderError) as excinfo:
        provider.chat([Message(role=MessageRole.USER, content="...")])

    # Assert that the error message is correct
    assert "Failed to decode tool call arguments" in str(excinfo.value)
    assert "some_tool" in str(excinfo.value)


@patch("allos.providers.openai.openai.OpenAI")
def test_chat_handles_missing_tool_call_arguments(MockOpenAI):
    """
    Test that missing tool call arguments are gracefully handled as an empty dict.
    """
    mock_client = MockOpenAI.return_value
    mock_api_response = MagicMock()

    # Setup the mock with arguments set to None
    mock_tool_call_item = MagicMock()
    mock_tool_call_item.type = "function_call"
    mock_tool_call_item.id = "fc_456"
    mock_tool_call_item.call_id = "call_456"
    mock_tool_call_item.name = "tool_with_no_args"
    mock_tool_call_item.arguments = None  # Arguments are missing

    mock_api_response.output = [mock_tool_call_item]
    mock_client.responses.create.return_value = mock_api_response

    provider = OpenAIProvider(model="gpt-4o")

    # The chat should complete successfully without raising an error
    response = provider.chat([Message(role=MessageRole.USER, content="...")])

    # Assert that a tool call was correctly parsed
    assert response.content is None
    assert len(response.tool_calls) == 1
    tool_call = response.tool_calls[0]

    assert tool_call.id == "call_456"
    assert tool_call.name == "tool_with_no_args"

    # CRITICAL ASSERTION: The arguments should be an empty dictionary
    assert tool_call.arguments == {}


@pytest.mark.parametrize(
    "openai_error, expected_error_message_part",
    [
        (
            openai.RateLimitError(
                "Rate limit exceeded",
                response=MagicMock(text="limit error details"),
                body=None,
            ),
            "limit error details",
        ),
        (
            openai.AuthenticationError(
                "Invalid API key",
                response=MagicMock(text="auth error details"),
                body=None,
            ),
            "auth error details",
        ),
        (openai.APIConnectionError(request=MagicMock()), "Connection error"),
        (
            openai.BadRequestError(
                "Bad request", response=MagicMock(text="bad request details"), body=None
            ),
            "bad request details",
        ),
        (  # Test the generic fallback
            openai.APIError(
                message="A generic error occurred", request=MagicMock(), body=None
            ),
            "A generic error occurred",
        ),
    ],
)
@patch("allos.providers.openai.openai.OpenAI")
def test_chat_error_handling(MockOpenAI, openai_error, expected_error_message_part):
    """Test that OpenAI API errors are caught and wrapped in ProviderError."""
    mock_client = MockOpenAI.return_value
    mock_client.responses.create.side_effect = openai_error

    provider = OpenAIProvider(model="gpt-4o")
    with pytest.raises(ProviderError) as excinfo:
        provider.chat([Message(role=MessageRole.USER, content="Hi")])

    assert expected_error_message_part in str(excinfo.value)


@pytest.mark.parametrize(
    "model_name, expected_window",
    [
        ("gpt-4o", 128000),
        ("gpt-4", 8192),
        ("gpt-3.5-turbo", 16385),
        ("unknown-model", 4096),  # Test fallback
    ],
)
@patch("allos.providers.openai.openai.OpenAI")
def test_get_context_window(MockOpenAI, model_name, expected_window):
    """Test that get_context_window returns correct size for known and unknown models."""
    # This test doesn't need to mock the OpenAI client
    provider = OpenAIProvider(model=model_name)
    assert provider.get_context_window() == expected_window


@patch("allos.providers.openai.openai.OpenAI")
def test_chat_skips_tool_call_with_missing_id_and_logs_warning(
    MockOpenAI, configured_caplog
):  # Use your configured_caplog
    """
    Test that a tool call with a missing ID is skipped, a warning is logged,
    and valid tool calls are still processed.
    """
    mock_client = MockOpenAI.return_value
    mock_api_response = MagicMock()

    # A valid tool call
    valid_tool_call = MagicMock(spec=["type", "id", "name", "arguments"])
    valid_tool_call.type = "function_call"
    valid_tool_call.id = "fc_valid_123"
    valid_tool_call.call_id = "call_valid_123"
    valid_tool_call.name = "get_weather"
    valid_tool_call.arguments = '{"location": "Boston"}'

    # An invalid tool call
    invalid_tool_call = MagicMock(spec=["type", "id", "name", "arguments"])
    invalid_tool_call.type = "function_call"
    invalid_tool_call.id = "fc_invalid_456"
    invalid_tool_call.call_id = None  # Explicitly missing
    invalid_tool_call.name = "get_forecast"
    invalid_tool_call.arguments = "{}"

    mock_api_response.output = [invalid_tool_call, valid_tool_call]
    mock_client.responses.create.return_value = mock_api_response

    provider = OpenAIProvider(model="gpt-4o")
    response = provider.chat([Message(role=MessageRole.USER, content="...")])

    # Assertions
    assert len(response.tool_calls) == 1
    processed_call = response.tool_calls[0]
    assert processed_call.id == "call_valid_123"
    assert "Skipping tool call due to missing call_id" in configured_caplog.text
    assert "get_forecast" in configured_caplog.text


@patch("allos.providers.openai.openai.OpenAI")
def test_chat_handles_response_with_no_output(MockOpenAI):
    """
    Test that a response with no 'output' field is handled gracefully.
    """
    mock_client = MockOpenAI.return_value
    mock_api_response = MagicMock()

    # Setup the mock response to have no 'output' attribute, or it could be None
    mock_api_response.output = None
    mock_client.responses.create.return_value = mock_api_response

    provider = OpenAIProvider(model="gpt-4o")

    # The chat should complete successfully without raising an error
    response = provider.chat([Message(role=MessageRole.USER, content="...")])

    # --- ASSERTIONS ---
    # 1. Assert that the response is empty and valid.
    assert response.content is None
    assert len(response.tool_calls) == 0

    # 2. Assert that the metadata correctly reflects that nothing was processed.
    assert response.metadata["overall"]["total"] == 0
    assert response.metadata["overall"]["processed"] == 0
    assert response.metadata["overall"]["skipped"] == 0


@patch("allos.providers.openai.openai.OpenAI")
def test_chat_skips_message_with_no_content(MockOpenAI):
    """
    Test that a message item with no 'content' is skipped, but others are processed.
    """
    mock_client = MockOpenAI.return_value
    mock_api_response = MagicMock()

    # --- Setup the mock response items ---
    # 1. An invalid message item with no content
    invalid_message = MagicMock(spec=["type", "content"])
    invalid_message.type = "message"
    invalid_message.content = None  # The condition to be tested

    # 2. A valid message item with content
    valid_message = MagicMock(spec=["type", "content"])
    valid_message.type = "message"
    # Create a mock for the content part as well
    content_part = MagicMock(spec=["type", "text"])
    content_part.type = "output_text"
    content_part.text = "This is a valid message."
    valid_message.content = [content_part]

    # The API response contains both items
    mock_api_response.output = [invalid_message, valid_message]
    mock_client.responses.create.return_value = mock_api_response

    provider = OpenAIProvider(model="gpt-4o")

    # The chat should complete successfully
    response = provider.chat([Message(role=MessageRole.USER, content="...")])

    # --- ASSERTIONS ---
    # 1. Assert that only the text from the valid message was included.
    assert response.content == "This is a valid message."
    assert len(response.tool_calls) == 0

    # 2. Assert the metadata reflects one processed and one skipped message.
    assert response.metadata["messages"]["total"] == 2
    assert response.metadata["messages"]["processed"] == 1
    assert response.metadata["messages"]["skipped"] == 1
    assert response.metadata["overall"]["total"] == 2
    assert response.metadata["overall"]["processed"] == 1
    assert response.metadata["overall"]["skipped"] == 1


@patch("allos.providers.openai.openai.OpenAI")
def test_assistant_message_with_content_and_tool_call_is_split(MockOpenAI):
    """
    Tests that a single assistant message with both text content and a tool call
    is correctly split into two separate items for the OpenAI `input` array.
    """
    mock_client = MockOpenAI.return_value
    mock_client.responses.create.return_value = MagicMock()

    provider = OpenAIProvider(model="gpt-4o")

    # This is the specific message type we need to test
    messages = [
        Message(
            role=MessageRole.ASSISTANT,
            content="Thinking about calling a tool...",
            tool_calls=[
                ToolCall(id="call_123", name="search", arguments={"query": "allos sdk"})
            ],
        )
    ]

    provider.chat(messages)

    call_kwargs = mock_client.responses.create.call_args.kwargs
    actual_input = call_kwargs.get("input", [])

    # The single Message should be expanded into two items
    assert len(actual_input) == 2

    # The first item should be the assistant's text content
    assert actual_input[0] == {
        "role": "assistant",
        "content": "Thinking about calling a tool...",
    }

    # The second item should be the function_call
    assert actual_input[1] == {
        "type": "function_call",
        "id": "fc_call_123",
        "call_id": "call_123",
        "name": "search",
        "arguments": '{"query": "allos sdk"}',
    }
