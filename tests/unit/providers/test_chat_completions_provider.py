# tests/unit/providers/test_chat_completions_provider.py

from unittest.mock import MagicMock, patch

import openai
import pytest

from allos.providers.base import Message, MessageRole, ToolCall
from allos.providers.chat_completions import ChatCompletionsProvider
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

    def execute(self, **kwargs):
        return {"weather": "sunny"}


@patch("allos.providers.chat_completions.openai.OpenAI")
def test_provider_initialization(MockOpenAI):
    """Test initialization with various parameters."""
    # 1. Standard initialization
    provider = ChatCompletionsProvider(model="gpt-4", api_key="test_key")
    MockOpenAI.assert_called_with(api_key="test_key", base_url=None)
    assert provider.model == "gpt-4"

    # 2. Initialization with custom base_url (e.g., Together AI)
    provider_custom = ChatCompletionsProvider(
        model="meta-llama/Llama-3-70b",
        api_key="xyz",
        base_url="https://api.together.xyz/v1",
    )
    MockOpenAI.assert_called_with(api_key="xyz", base_url="https://api.together.xyz/v1")
    assert (
        provider_custom.base_url == "https://api.together.xyz/v1"
    )  # pyright: ignore[reportAttributeAccessIssue]


@patch("allos.providers.chat_completions.openai.OpenAI")
def test_provider_initialization_error(MockOpenAI):
    """Test exception handling during initialization."""
    MockOpenAI.side_effect = Exception("Init failed")
    with pytest.raises(ProviderError) as excinfo:
        ChatCompletionsProvider(model="gpt-4")
    assert "Failed to initialize ChatCompletions client" in str(excinfo.value)


@patch("allos.providers.chat_completions.openai.OpenAI")
def test_chat_sends_correctly_formatted_messages(MockOpenAI):
    """Test message conversion for the Chat Completions API format."""
    mock_client = MockOpenAI.return_value
    mock_client.chat.completions.create.return_value = MagicMock()

    provider = ChatCompletionsProvider(model="gpt-4")
    messages = [
        Message(role=MessageRole.SYSTEM, content="System prompt"),
        Message(role=MessageRole.USER, content="User prompt"),
        Message(
            role=MessageRole.ASSISTANT,
            content=None,
            tool_calls=[ToolCall("call_1", "my_tool", {"arg": 1})],
        ),
        Message(
            role=MessageRole.TOOL,
            content='{"result": "ok"}',
            tool_call_id="call_1",
        ),
    ]

    provider.chat(messages)

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    sent_messages = call_kwargs["messages"]

    # Verify structure matches Chat Completions API requirements
    assert sent_messages[0] == {"role": "system", "content": "System prompt"}
    assert sent_messages[1] == {"role": "user", "content": "User prompt"}

    # Verify assistant tool call format
    assert sent_messages[2]["role"] == "assistant"
    assert "tool_calls" in sent_messages[2]
    assert sent_messages[2]["tool_calls"][0]["id"] == "call_1"
    assert sent_messages[2]["tool_calls"][0]["function"]["name"] == "my_tool"

    # Verify tool result format (critical difference from Responses API)
    assert sent_messages[3] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": '{"result": "ok"}',
    }


@patch("allos.providers.chat_completions.openai.OpenAI")
def test_chat_sends_correctly_converted_assistant_messages(MockOpenAI):
    """Test message conversion, including assistant messages with content."""
    mock_client = MockOpenAI.return_value
    mock_client.chat.completions.create.return_value = MagicMock()

    provider = ChatCompletionsProvider(model="gpt-4")
    messages = [
        # This covers the `if msg.content` branch for assistant messages
        Message(role=MessageRole.ASSISTANT, content="I am thinking."),
    ]

    provider.chat(messages)

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    sent_messages = call_kwargs["messages"]

    assert sent_messages[0] == {"role": "assistant", "content": "I am thinking."}


@patch("allos.providers.chat_completions.openai.OpenAI")
def test_chat_parses_tool_call_malformed_json(MockOpenAI, configured_caplog):
    """Test parsing a tool call with invalid JSON arguments."""
    mock_client = MockOpenAI.return_value
    mock_response = MagicMock()
    mock_choice = MagicMock()

    # Setup tool call with bad JSON
    mock_tool_call = MagicMock()
    mock_tool_call.type = "function"
    mock_tool_call.function.name = "bad_tool"
    mock_tool_call.function.arguments = "{bad_json"

    mock_choice.message.tool_calls = [mock_tool_call]
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response

    provider = ChatCompletionsProvider(model="gpt-4")
    provider.chat([Message(role=MessageRole.USER, content="Hi")])

    # Verify the warning was logged (hitting the except JSONDecodeError block)
    assert "Failed to decode arguments for tool 'bad_tool'" in configured_caplog.text


@patch("allos.providers.chat_completions.openai.OpenAI")
def test_chat_sends_correctly_formatted_tools(MockOpenAI):
    """Test tool schema conversion (must use 'function' wrapper)."""
    mock_client = MockOpenAI.return_value
    mock_client.chat.completions.create.return_value = MagicMock()

    provider = ChatCompletionsProvider(model="gpt-4")
    tools = [DummyTool()]

    provider.chat([Message(role=MessageRole.USER, content="Hi")], tools=tools)

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    sent_tools = call_kwargs["tools"]

    assert len(sent_tools) == 1
    # Chat Completions uses external tagging: { type: function, function: { ... } }
    assert sent_tools[0]["type"] == "function"
    assert "function" in sent_tools[0]
    assert sent_tools[0]["function"]["name"] == "get_weather"
    assert "parameters" in sent_tools[0]["function"]


@patch("allos.providers.chat_completions.openai.OpenAI")
def test_chat_skips_unsupported_tool_type(MockOpenAI, configured_caplog):
    """Test that non-function tool calls are skipped."""
    mock_client = MockOpenAI.return_value
    mock_response = MagicMock()
    mock_choice = MagicMock()

    # Setup a tool call with type='custom' (or anything not 'function')
    mock_tool_call = MagicMock()
    mock_tool_call.type = "custom_tool_type"

    mock_choice.message.tool_calls = [mock_tool_call]
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response

    provider = ChatCompletionsProvider(model="gpt-4")
    provider.chat([Message(role=MessageRole.USER, content="Hi")])

    # Verify the debug log (hitting the else block)
    assert "Skipping unsupported tool type: custom_tool_type" in configured_caplog.text


@patch("allos.providers.chat_completions.openai.OpenAI")
def test_chat_error_handling_detailed(MockOpenAI):
    """Test exception mapping with detailed body parsing."""
    mock_client = MockOpenAI.return_value

    # 1. BadRequestError with body
    mock_client.chat.completions.create.side_effect = openai.BadRequestError(
        "Standard message",
        response=MagicMock(),
        body={"message": "Detailed API error message"},
    )

    provider = ChatCompletionsProvider(model="gpt-4")

    with pytest.raises(ProviderError) as excinfo:
        provider.chat([Message(role=MessageRole.USER, content="Hi")])

    # Verify we extracted the detailed message
    assert "Detailed API error message" in str(excinfo.value)

    # 2. APIConnectionError
    mock_client.chat.completions.create.side_effect = openai.APIConnectionError(
        request=MagicMock()
    )
    with pytest.raises(ProviderError) as excinfo:
        provider.chat([Message(role=MessageRole.USER, content="Hi")])
    assert "Connection error" in str(excinfo.value)

    # 3. APIError
    mock_client.chat.completions.create.side_effect = openai.APIError(
        "Generic API failure", request=MagicMock(), body=None
    )
    with pytest.raises(ProviderError) as excinfo:
        provider.chat([Message(role=MessageRole.USER, content="Hi")])
    assert "API error: Generic API failure" in str(excinfo.value)


@patch("allos.providers.chat_completions.openai.OpenAI")
def test_chat_parses_simple_response(MockOpenAI):
    """Test parsing a text response."""
    mock_client = MockOpenAI.return_value
    mock_response = MagicMock()
    mock_choice = MagicMock()

    # Setup choice message
    mock_choice.message.content = "Hello world"
    mock_choice.message.tool_calls = None
    mock_response.choices = [mock_choice]

    mock_client.chat.completions.create.return_value = mock_response

    provider = ChatCompletionsProvider(model="gpt-4")
    response = provider.chat([Message(role=MessageRole.USER, content="Hi")])

    assert response.content == "Hello world"
    assert len(response.tool_calls) == 0


@patch("allos.providers.chat_completions.openai.OpenAI")
def test_chat_parses_tool_call_response(MockOpenAI):
    """Test parsing a response containing tool calls."""
    mock_client = MockOpenAI.return_value
    mock_response = MagicMock()
    mock_choice = MagicMock()

    # Setup tool call
    mock_tool_call = MagicMock()
    mock_tool_call.id = "call_123"
    mock_tool_call.type = "function"
    mock_tool_call.function.name = "get_weather"
    mock_tool_call.function.arguments = '{"location": "Paris"}'

    mock_choice.message.content = None
    mock_choice.message.tool_calls = [mock_tool_call]
    mock_response.choices = [mock_choice]

    mock_client.chat.completions.create.return_value = mock_response

    provider = ChatCompletionsProvider(model="gpt-4")
    response = provider.chat([Message(role=MessageRole.USER, content="Hi")])

    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].id == "call_123"
    assert response.tool_calls[0].name == "get_weather"
    assert response.tool_calls[0].arguments == {"location": "Paris"}


@patch("allos.providers.chat_completions.openai.OpenAI")
def test_chat_error_handling(MockOpenAI):
    """Test exception mapping."""
    mock_client = MockOpenAI.return_value
    # Simulate a specific API error
    mock_client.chat.completions.create.side_effect = openai.BadRequestError(
        "Context length exceeded", response=MagicMock(), body=None
    )

    provider = ChatCompletionsProvider(model="gpt-4")

    with pytest.raises(ProviderError) as excinfo:
        provider.chat([Message(role=MessageRole.USER, content="Hi")])

    assert "Context length exceeded" in str(excinfo.value)
    assert "[chat_completions]" in str(excinfo.value)


@pytest.mark.parametrize(
    "model_name, expected_window",
    [
        ("gpt-4o", 128000),
        ("gpt-4", 8192),
        ("gpt-3.5-turbo", 16385),
        ("unknown-model", 4096),  # Test fallback
    ],
)
@patch("allos.providers.chat_completions.openai.OpenAI")
def test_get_context_window(MockOpenAI, model_name, expected_window):
    """Test that get_context_window returns correct size for known and unknown models."""
    provider = ChatCompletionsProvider(model=model_name)
    assert provider.get_context_window() == expected_window
