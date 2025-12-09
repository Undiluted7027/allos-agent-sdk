# tests/unit/test_anthropic_provider.py

import time
from unittest.mock import MagicMock, patch

import anthropic
import pytest

from allos.providers.anthropic import AnthropicProvider
from allos.providers.base import Message, MessageRole, ProviderChunk, ToolCall
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


class TestAnthropicProviderStreaming:
    @pytest.fixture
    def provider(self):
        with patch("allos.providers.anthropic.anthropic.Anthropic"):
            # The provider will now be created with the mocked client inside it
            provider = AnthropicProvider(model="claude-3-haiku-20240307")
            yield provider

    def test_build_api_kwargs_with_system_prompt_and_tools(self, provider):
        """Test the _build_api_kwargs helper with all options."""
        messages = [
            Message(role=MessageRole.SYSTEM, content="System prompt"),
            Message(role=MessageRole.USER, content="User prompt"),
        ]
        tools = [DummyTool()]

        api_kwargs, system_prompt = provider._build_api_kwargs(
            messages, tools, {"temperature": 0.5}
        )

        assert system_prompt == "System prompt"
        assert api_kwargs["system"] == "System prompt"
        assert api_kwargs["temperature"] == 0.5
        assert len(api_kwargs["tools"]) == 1
        assert api_kwargs["tools"][0]["name"] == "get_weather"

    def test_stream_chat_simple_text(self, provider):
        """Test a simple text stream with start, delta, and stop events."""
        mock_text_stream_events = [
            MagicMock(
                type="message_start",
                message=MagicMock(
                    id="msg_1", model="claude-test", usage=MagicMock(input_tokens=10)
                ),
            ),
            MagicMock(
                type="content_block_delta",
                delta=MagicMock(type="text_delta", text="Hello"),
            ),
            MagicMock(
                type="content_block_delta",
                delta=MagicMock(type="text_delta", text=" World"),
            ),
            MagicMock(type="message_delta", usage=MagicMock(output_tokens=5)),
            MagicMock(type="message_stop"),
        ]
        provider.client.messages.stream.return_value.__enter__.return_value = (
            mock_text_stream_events
        )

        chunks = list(
            provider.stream_chat([Message(role=MessageRole.USER, content="Hi")])
        )

        # Get text content from all chunks
        content = "".join(c.content for c in chunks if c.content)
        assert content == "Hello World"

        # The last chunk should have final metadata
        final_chunk = chunks[-1]
        assert isinstance(final_chunk, ProviderChunk)
        assert final_chunk.final_metadata is not None
        assert final_chunk.final_metadata.usage.input_tokens == 10
        assert final_chunk.final_metadata.usage.output_tokens == 5

    def test_stream_chat_tool_call(self, provider):
        """Test a stream that involves a tool call."""
        # Create a proper mock for content_block
        content_block_mock = MagicMock()
        content_block_mock.type = "tool_use"
        content_block_mock.id = "tool_123"
        content_block_mock.name = "get_weather"  # Set as string, not MagicMock

        mock_tool_stream_events = [
            MagicMock(
                type="message_start",
                message=MagicMock(
                    id="msg_2", model="claude-test", usage=MagicMock(input_tokens=25)
                ),
            ),
            MagicMock(
                type="content_block_start",
                index=0,
                content_block=content_block_mock,
            ),
            MagicMock(
                type="content_block_delta",
                index=0,
                delta=MagicMock(type="input_json_delta", partial_json='{"location":'),
            ),
            MagicMock(
                type="content_block_delta",
                index=0,
                delta=MagicMock(type="input_json_delta", partial_json='"Boston"}'),
            ),
            MagicMock(type="content_block_stop", index=0),
            MagicMock(type="message_delta", usage=MagicMock(output_tokens=30)),
            MagicMock(type="message_stop"),
        ]
        provider.client.messages.stream.return_value.__enter__.return_value = (
            mock_tool_stream_events
        )

        chunks = list(
            provider.stream_chat([Message(role=MessageRole.USER, content="Weather?")])
        )

        tool_done_chunks = [c for c in chunks if c.tool_call_done]
        assert len(tool_done_chunks) == 1
        tool_call = tool_done_chunks[0].tool_call_done
        assert tool_call.id == "tool_123"
        assert tool_call.name == "get_weather"
        assert tool_call.arguments == {"location": "Boston"}

        final_chunk = chunks[-1]
        assert final_chunk.final_metadata.usage.input_tokens == 25
        assert final_chunk.final_metadata.usage.output_tokens == 30

    def test_stream_chat_api_error_before_stream(self, provider):
        """Test that a ProviderError is raised for API errors before streaming starts."""
        error = anthropic.APIError(
            message="Test API Error", request=MagicMock(), body=None
        )

        # Make the stream() call itself raise the error
        provider.client.messages.stream.side_effect = error

        with pytest.raises(
            ProviderError, match="Anthropic API streaming error: Test API Error"
        ):
            # Consume the generator to trigger the code
            list(provider.stream_chat([Message(role=MessageRole.USER, content="Hi")]))

    def test_stream_chat_handles_tool_argument_json_error(self, provider):
        """Test that malformed JSON in tool arguments yields an error chunk."""
        mock_bad_json_events = [
            MagicMock(
                type="message_start",
                message=MagicMock(
                    id="msg_3", model="claude-test", usage=MagicMock(input_tokens=1)
                ),
            ),
            MagicMock(
                type="content_block_start",
                index=0,
                content_block=MagicMock(
                    type="tool_use", id="tool_456", name="bad_tool"
                ),
            ),
            MagicMock(
                type="content_block_delta",
                index=0,
                delta=MagicMock(type="input_json_delta", partial_json='{"bad"'),
            ),
            MagicMock(type="content_block_stop", index=0),
            MagicMock(type="message_stop"),
        ]
        provider.client.messages.stream.return_value.__enter__.return_value = (
            mock_bad_json_events
        )

        chunks = list(
            provider.stream_chat([Message(role=MessageRole.USER, content="...")])
        )

        error_chunk = next((c for c in chunks if c.error), None)
        assert error_chunk is not None
        assert "Failed to parse tool arguments" in error_chunk.error

    def test_finalize_stream_handles_error(self, provider, configured_caplog):
        """Test that an error during metadata finalization returns an error chunk."""
        # Create a state that will cause the metadata builder to fail
        bad_state = {
            "in_progress_tool_calls": {},
            "final_usage": {},  # Missing required token counts
            "response_id": "",
            "model_id": "",
        }

        # Patch MetadataBuilder to raise an error
        with patch("allos.providers.anthropic.MetadataBuilder") as MockBuilder:
            MockBuilder.return_value.with_response_obj.side_effect = ValueError(
                "Forced error"
            )

            # Call the internal method directly
            error_chunk = provider._finalize_stream(bad_state, {}, time.time())

        assert isinstance(error_chunk, ProviderChunk)
        assert error_chunk.error is not None
        assert (
            "Internal error: Failed to process final stream metadata"
            in error_chunk.error
        )

        # Check that the error was logged
        assert (
            "Error building final metadata for Anthropic stream"
            in configured_caplog.text
        )

    def test_handle_block_stop_no_call_in_state(self, provider):
        """
        Test that _handle_block_stop gracefully returns None if the event index
        is not in the tracking state. This covers the 'if not call:' branch.
        """
        # 1. Create a state, but keep 'in_progress_tool_calls' empty
        state = provider._init_stream_state()

        # 2. Create a mock event with an index that doesn't exist in the state
        mock_event = MagicMock(index=99)

        # 3. Call the method
        result = provider._handle_block_stop(mock_event, state)

        # 4. Assert that the method returned None and did not raise an error
        assert result is None
