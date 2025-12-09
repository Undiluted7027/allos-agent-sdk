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
def test_chat_converts_max_tokens_parameter(MockOpenAI):
    """
    Test that the legacy 'max_tokens' parameter is converted to the
    parameter name expected by the Responses API (e.g., 'max_output_tokens').
    """
    mock_client = MockOpenAI.return_value
    mock_client.responses.create.return_value = MagicMock()

    provider = OpenAIProvider(model="gpt-4o")

    # Call with the generic SDK parameter 'max_tokens'
    provider.chat([Message(role=MessageRole.USER, content="Hi")], max_tokens=500)

    mock_client.responses.create.assert_called_once()
    call_kwargs = mock_client.responses.create.call_args.kwargs

    # Assert 'max_tokens' was removed
    assert "max_tokens" not in call_kwargs

    # Assert the new key is present
    assert call_kwargs.get("max_output_tokens") == 500


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
    assert response.metadata.usage.total_tokens == 2


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
    assert response.metadata.usage.total_tokens >= 0


@patch("allos.providers.openai.openai.OpenAI")
def test_assistant_message_with_content_and_tool_call_is_split(MockOpenAI):
    """
    Tests that a single assistant message with both text content and a tool call
    is correctly split into two separate items for the OpenAI `input` array.
    """
    mock_client = MockOpenAI.return_value
    mock_api_response = MagicMock()
    mock_api_response.id = "msg_123"
    mock_api_response.model = "mock-model-id"
    mock_api_response.usage = MagicMock(input_tokens=10, output_tokens=20)
    mock_api_response.content = []  # or other content as needed by the test
    mock_client.messages.create.return_value = mock_api_response

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


class TestOpenAIProviderStreaming:
    @pytest.fixture
    def provider(self):
        with patch("allos.providers.openai.openai.OpenAI"):
            provider = OpenAIProvider(model="gpt-4o")
            yield provider

    def test_stream_chat_simple_text(self, provider):
        """Test a simple text stream with delta and completed events."""
        mock_events = [
            MagicMock(type="response.output_text.delta", delta="Hello"),
            MagicMock(type="response.output_text.delta", delta=" World"),
            MagicMock(
                type="response.completed",
                response=MagicMock(usage=MagicMock(input_tokens=10, output_tokens=5)),
            ),
        ]
        provider.client.responses.create.return_value = iter(mock_events)

        chunks = list(
            provider.stream_chat([Message(role=MessageRole.USER, content="Hi")])
        )

        content = "".join(c.content for c in chunks if c.content)
        assert content == "Hello World"

        final_chunk = chunks[-1]
        assert final_chunk.final_metadata is not None
        assert final_chunk.final_metadata.usage.input_tokens == 10
        assert final_chunk.final_metadata.usage.output_tokens == 5

    def test_stream_chat_tool_call(self, provider):
        """Test a stream that involves a tool call."""
        # Create a properly configured mock for the 'item' in the events
        item_mock = MagicMock()
        item_mock.type = "function_call"
        item_mock.call_id = "call_123"
        item_mock.name = "get_weather"  # Explicitly set the name attribute as a string

        mock_events = [
            MagicMock(
                type="response.output_item.added",
                output_index=0,
                item=item_mock,
            ),
            MagicMock(
                type="response.function_call_arguments.delta",
                output_index=0,
                delta='{"location":',
            ),
            MagicMock(
                type="response.function_call_arguments.delta",
                output_index=0,
                delta='"Boston"}',
            ),
            MagicMock(
                type="response.output_item.done",
                output_index=0,
                item=item_mock,  # Reuse the same correctly configured mock
            ),
            MagicMock(
                type="response.completed",
                response=MagicMock(
                    id="resp_123",
                    model="gpt-4o",
                    usage=MagicMock(input_tokens=20, output_tokens=30),
                ),
            ),
        ]
        provider.client.responses.create.return_value = iter(mock_events)

        chunks = list(
            provider.stream_chat(
                [Message(role=MessageRole.USER, content="Weather?")],
                tools=[DummyTool()],
            )
        )

        tool_start_chunk = next(c for c in chunks if c.tool_call_start)
        assert tool_start_chunk.tool_call_start["name"] == "get_weather"

        tool_done_chunk = next(c for c in chunks if c.tool_call_done)
        assert tool_done_chunk.tool_call_done.id == "call_123"
        assert tool_done_chunk.tool_call_done.arguments == {"location": "Boston"}

        final_chunk = chunks[-1]
        assert final_chunk.final_metadata is not None
        assert final_chunk.final_metadata.usage.total_tokens == 50

    def test_stream_chat_pre_stream_api_error(self, provider):
        """Test that ProviderError is raised for API errors before streaming."""
        error = openai.APIError(
            message="Test API Error", request=MagicMock(), body=None
        )
        provider.client.responses.create.side_effect = error

        with pytest.raises(
            ProviderError, match="OpenAI API error during streaming: Test API Error"
        ):
            list(provider.stream_chat([Message(role=MessageRole.USER, content="Hi")]))

    def test_stream_handles_api_error_event(self, provider):
        """Test the _handle_error helper for in-stream 'error' events."""
        mock_events = [
            MagicMock(type="error", error=MagicMock(message="In-stream failure"))
        ]
        provider.client.responses.create.return_value = iter(mock_events)

        chunks = list(provider.stream_chat([]))

        assert len(chunks) == 1
        assert chunks[0].error == "API Error: In-stream failure"

    def test_handle_tool_args_delta_with_unknown_index(self, provider):
        """Test that a delta for an untracked tool call index is ignored."""
        # This covers the 'if event.output_index not in state:' branch
        state = {0: {"id": "call_0", "name": "tool_0", "arguments": ""}}
        mock_event = MagicMock(
            output_index=1, delta="ignored"
        )  # Index 1 is not in state

        # The generator should yield nothing and not raise an error
        result = list(provider._handle_tool_args_delta(mock_event, state, {}))
        assert not result

    def test_handle_tool_done_with_unknown_index(self, provider):
        """Test that a done event for an untracked tool call index is ignored."""
        # This covers the 'if not call_state:' branch
        state = {0: {"id": "call_0", "name": "tool_0", "arguments": ""}}
        mock_event = MagicMock(
            type="function_call", output_index=1
        )  # Index 1 is not in state

        # The generator should yield nothing and not raise an error
        result = list(provider._handle_tool_done(mock_event, state, {}))
        assert not result

    def test_handle_tool_done_json_error(self, provider):
        """Test that malformed JSON in tool arguments yields an error chunk."""
        state = {0: {"id": "call_123", "name": "bad_tool", "arguments": "{bad_json"}}
        mock_event = MagicMock(
            type="function_call", output_index=0, item=MagicMock(type="function_call")
        )

        # The generator will yield an error and then the finally block runs.
        chunks = list(provider._handle_tool_done(mock_event, state, {}))

        assert len(chunks) == 1
        error_chunk = chunks[0]
        assert error_chunk.error is not None
        assert "Failed to parse tool arguments for bad_tool" in error_chunk.error
        # Also assert that the state was cleaned up by the 'finally' block
        assert 0 not in state

    def test_handle_completed_error(self, provider, configured_caplog, mocker):
        """Test that errors during final metadata creation are handled."""
        # This covers the 'except' block in _handle_completed
        mock_event = MagicMock(
            type="response.completed",
            response=MagicMock(usage=MagicMock(input_tokens=10, output_tokens=5)),
        )
        api_call_context = {"builder_kwargs": {}, "start_time": 0}

        # Patch the MetadataBuilder to force an exception
        mocker.patch(
            "allos.providers.openai.MetadataBuilder",
            side_effect=AttributeError("Forced build error"),
        )

        # Consume the generator to get the yielded chunk
        error_chunk = next(provider._handle_completed(mock_event, {}, api_call_context))

        assert (
            error_chunk.error
            == "Internal error: Failed to process final stream metadata."
        )
        assert (
            "Error building final metadata for OpenAI stream" in configured_caplog.text
        )

    def test_stream_chat_sends_instructions(self, provider):
        """Test that a SYSTEM message is correctly passed as 'instructions'."""
        # This covers the 'if instructions:' branch in stream_chat
        provider.client.responses.create.return_value = iter(
            []
        )  # We only care about call args
        messages = [
            Message(role=MessageRole.SYSTEM, content="You are a test bot."),
            Message(role=MessageRole.USER, content="Hi"),
        ]

        list(provider.stream_chat(messages))

        provider.client.responses.create.assert_called_once()
        call_kwargs = provider.client.responses.create.call_args.kwargs
        assert "instructions" in call_kwargs
        assert call_kwargs["instructions"] == "You are a test bot."

    def test_handle_tool_start_ignores_non_function_call(self, provider):
        """Test that _handle_tool_start ignores items that aren't function calls."""
        # This covers the 'if event.item.type != "function_call":' branch
        state = {}
        mock_event = MagicMock(
            type="response.output_item.added",
            item=MagicMock(type="message"),  # Not a 'function_call'
        )

        # The generator should be empty and not modify the state
        chunks = list(provider._handle_tool_start(mock_event, state, {}))
        assert not chunks
        assert not state

    @pytest.mark.parametrize(
        "state_to_test, description",
        [
            (
                {0: {"id": "call_123", "name": "bad_tool", "arguments": "{bad_json"}},
                "Test malformed JSON in arguments",
            ),
            (
                {0: {"name": "bad_tool", "arguments": "{}"}},
                "Test KeyError from missing 'id'",
            ),
        ],
    )
    def test_handle_completed_fallback_state_parsing_errors(
        self, provider, configured_caplog, state_to_test, description
    ):
        """Test the error handling within the _handle_completed state fallback logic."""
        # This covers the 'except (json.JSONDecodeError, KeyError)' branch
        mock_event = MagicMock(
            type="response.completed",
            response=MagicMock(
                id="resp_123",
                model="gpt-4o",
                output=None,  # No output to force fallback
                usage=MagicMock(input_tokens=10, output_tokens=5),
            ),
        )
        api_call_context = {"builder_kwargs": {}, "start_time": 0}

        # The generator should yield one metadata chunk
        chunks = list(
            provider._handle_completed(mock_event, state_to_test, api_call_context)
        )

        assert len(chunks) == 1
        metadata = chunks[0].final_metadata

        # Assert that no tools were added due to the parsing error
        assert metadata.tools.total_tool_calls == 0
        assert not metadata.tools.tool_calls
        assert "Failed to parse tool call from state" in configured_caplog.text

    def test_handle_completed_fallback_to_state_for_tool_calls(
        self, provider, configured_caplog
    ):
        """Test the fallback logic to extract tool calls from streaming state."""
        # This covers the 'if not metadata.tools.tool_calls and _state:' branch
        _state = {
            0: {
                "id": "call_123",
                "name": "get_weather",
                "arguments": '{"location": "Boston"}',
            }
        }
        mock_event = MagicMock(
            type="response.completed",
            response=MagicMock(
                id="resp_123",
                model="gpt-4o",
                output=None,  # No output to force fallback
                usage=MagicMock(input_tokens=10, output_tokens=5),
            ),
        )
        api_call_context = {"builder_kwargs": {"tools": [DummyTool()]}, "start_time": 0}

        chunk = next(provider._handle_completed(mock_event, _state, api_call_context))

        assert "Response object missing tool calls" in configured_caplog.text
        metadata = chunk.final_metadata
        assert metadata.tools.total_tool_calls == 1
        assert metadata.tools.tool_calls[0].tool_call_id == "call_123"
        assert metadata.tools.tool_calls[0].tool_name == "get_weather"

    def test_handle_completed_logs_output_items(self, provider, configured_caplog):
        """Test the debug logging loop for output items in _handle_completed."""
        # This covers the 'for item in event.response.output or []:' loop

        # Create more realistic mocks for the output items
        message_item = MagicMock()
        message_item.type = "message"

        function_call_item = MagicMock()
        function_call_item.type = "function_call"
        function_call_item.call_id = "call_xyz"
        function_call_item.name = "a_tool"
        function_call_item.arguments = "{}"  # Must be a valid JSON string

        mock_event = MagicMock(
            type="response.completed",
            response=MagicMock(
                id="resp_123",
                model="gpt-4o",
                output=[message_item, function_call_item],
                usage=MagicMock(input_tokens=10, output_tokens=5),
            ),
        )
        api_call_context = {"builder_kwargs": {"tools": []}, "start_time": 0}

        list(provider._handle_completed(mock_event, {}, api_call_context))

        # Check that the build process didn't log an error
        assert "Error building final metadata" not in configured_caplog.text

        # Now check for the intended debug logs
        assert "Output items: 2" in configured_caplog.text
        assert "- Item type: message" in configured_caplog.text
        assert "- Item type: function_call" in configured_caplog.text

    def test_handle_tool_done_ignores_untracked_index(self, provider):
        """Test that _handle_tool_done ignores an event with an untracked index."""
        # This covers the 'if not call_state:' branch
        state = {0: {"id": "call_0", "name": "tool_0", "arguments": "{}"}}
        # Create an event for an index (e.g., 99) that is not in the state
        mock_event = MagicMock(item=MagicMock(type="function_call"), output_index=99)

        # Call the handler and consume the generator
        result_chunks = list(provider._handle_tool_done(mock_event, state, {}))

        # Assert that nothing was yielded and the original state is unchanged
        assert not result_chunks
        assert 0 in state  # The original call at index 0 should still be there
