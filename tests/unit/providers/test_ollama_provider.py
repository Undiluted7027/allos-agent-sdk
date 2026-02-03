# tests/unit/test_ollama_provider.py

import time
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest
from ollama import RequestError, ResponseError
from ollama._types import ListResponse, ShowResponse

from allos.providers.base import Message, MessageRole, ToolCall
from allos.providers.ollama import OLLAMA_TOOL_SUPPORTED_MODELS, OllamaProvider
from allos.tools.base import BaseTool, ToolParameter
from allos.utils.errors import ProviderError

# Mock response from `ollama.Client.list()`
MOCK_MODEL_LIST: ListResponse = ListResponse(
    models=[
        ListResponse.Model(model="llama3:latest"),
        ListResponse.Model(model="mistral:latest"),
        ListResponse.Model(model="qwen3:8b"),
        ListResponse.Model(model="qwen2:latest"),
        ListResponse.Model(model="llama3.1:latest"),
    ]
)


# Helper function to create mock ShowResponse with capabilities
def create_mock_show_response(
    model_name: str, supports_tools: bool = False, context_length: Optional[int] = None
) -> ShowResponse:
    """Create a mock ShowResponse for testing."""
    mock_response = MagicMock(spec=ShowResponse)
    mock_response.capabilities = {"tools": True} if supports_tools else {}
    mock_response.modelinfo = {"num_ctx": context_length} if context_length else {}
    return mock_response


# Mock response from `ollama.Client.chat()`
MOCK_CHAT_RESPONSE = {
    "model": "llama3:latest",
    "created_at": "2023-08-04T08:52:19.811567Z",
    "message": {"role": "assistant", "content": "Hello there!"},
    "done": True,
    "total_duration": 5032962375,
    "load_duration": 263208,
    "prompt_eval_count": 26,
    "prompt_eval_duration": 325953000,
    "eval_count": 298,
    "eval_duration": 4703952000,
}

# Mock response for a chat call that returns a tool call
MOCK_TOOL_CALL_RESPONSE = {
    "model": "qwen3:8b",
    "message": {
        "role": "assistant",
        "content": "",
        "thinking": "I will be executing the `get_weather` tool.",
        "tool_name": None,
        "tool_calls": [
            {
                "function": {
                    "name": "get_weather",
                    "arguments": {"location": "Boston"},
                }
            }
        ],
    },
    "done": True,
    "prompt_eval_count": 50,
    "eval_count": 10,
}

# Mock stream for a simple text response
MOCK_TEXT_STREAM = [
    {"message": {"role": "assistant", "content": "Hel"}, "done": False},
    {"message": {"role": "assistant", "content": "lo!"}, "done": False},
    {
        "done": True,
        "model": "llama3:latest",
        "total_duration": 1000,
        "prompt_eval_count": 10,
        "eval_count": 2,
    },
]

# Mock stream for a tool call response
MOCK_TOOL_STREAM = [
    {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": {"location": "San Francisco"},
                    }
                }
            ],
        },
        "done": False,
    },
    {
        "done": True,
        "model": "qwen2:latest",
        "total_duration": 1200,
        "prompt_eval_count": 60,
        "eval_count": 15,
    },
]


class MockTool(BaseTool):
    name: str = "get_weather"
    description: str = "Gets the weather"
    parameters: list[ToolParameter] = [
        ToolParameter(
            name="location",
            type="string",
            required=True,
            description="Location to get weather of.",
        )
    ]

    def execute(self, **kwargs):
        return {}


MOCK_TOOLS = [MockTool()]


@patch("allos.providers.ollama.Client")
def test_init_success(MockClient):
    """Tests successful initialization of OllamaProvider."""
    mock_instance = MockClient.return_value
    mock_instance.list.return_value = MOCK_MODEL_LIST
    mock_instance.show.return_value = create_mock_show_response("llama3:latest")

    provider = OllamaProvider(model="llama3:latest")
    assert provider.model == "llama3:latest"
    mock_instance.list.assert_called_once()


@patch("allos.providers.ollama.Client")
def test_init_model_not_found(MockClient):
    """Tests that ProviderError is raised if the model is not available locally."""
    mock_instance = MockClient.return_value
    mock_instance.list.return_value = MOCK_MODEL_LIST
    mock_instance.show.return_value = create_mock_show_response("default")

    with pytest.raises(ProviderError) as excinfo:
        OllamaProvider(model="gemma:latest")

    assert "Model 'gemma:latest' not available locally" in str(excinfo.value)
    assert "ollama pull gemma:latest" in str(excinfo.value)


@patch("allos.providers.ollama.Client")
def test_init_connection_error(MockClient):
    """Tests that ProviderError is raised if the client cannot connect."""

    MockClient.return_value.list.side_effect = RequestError("Connection failed")

    with pytest.raises(ProviderError) as excinfo:
        OllamaProvider(model="llama3:latest")

    assert "Could not connect to Ollama server" in str(excinfo.value)


@patch("allos.providers.ollama.Client")
def test_init_response_error(MockClient):
    """Tests that a ResponseError during initialization is handled."""

    # Simulate a model not found error or similar issue from the server
    mock_instance = MockClient.return_value
    mock_instance.list.side_effect = ResponseError("Model not found", status_code=404)

    with pytest.raises(ProviderError) as excinfo:
        OllamaProvider(model="llama3:latest")

    assert "Failed to connect to Ollama server: Model not found" in str(excinfo.value)
    assert excinfo.value.provider == "ollama"


def test_chat_success():
    """Tests a successful synchronous chat call."""
    with patch("allos.providers.ollama.Client") as MockClient:
        mock_instance = MockClient.return_value
        mock_instance.list.return_value = MOCK_MODEL_LIST
        mock_instance.show.return_value = create_mock_show_response("default")
        mock_instance.chat.return_value = MOCK_CHAT_RESPONSE

        provider = OllamaProvider(model="llama3:latest")
        messages = [Message(role=MessageRole.USER, content="Hi")]
        response = provider.chat(messages)

        mock_instance.chat.assert_called_once()
        assert response.content == "Hello there!"
        assert response.metadata.usage.input_tokens == 26
        assert response.metadata.usage.output_tokens == 298


@patch("allos.providers.ollama.Client")
def test_convert_messages_formats_all_roles(MockClient):
    """Test that all message roles are correctly converted to Ollama format."""
    mock_instance = MockClient.return_value
    mock_instance.list.return_value = MOCK_MODEL_LIST
    mock_instance.show.return_value = create_mock_show_response("default")

    provider = OllamaProvider(model="llama3:latest")

    messages = [
        Message(role=MessageRole.SYSTEM, content="You are helpful."),
        Message(role=MessageRole.USER, content="Hello"),
        Message(role=MessageRole.ASSISTANT, content="Hi there!"),
        Message(
            role=MessageRole.TOOL, content='{"result": "ok"}', tool_call_id="call_123"
        ),
    ]

    ollama_messages = provider._convert_to_ollama_messages(messages)  # type: ignore

    assert ollama_messages[0] == {"role": "system", "content": "You are helpful."}
    assert ollama_messages[1] == {"role": "user", "content": "Hello"}
    assert ollama_messages[2] == {"role": "assistant", "content": "Hi there!"}
    assert ollama_messages[3] == {
        "role": "tool",
        "content": '{"result": "ok"}',
        "tool_call_id": "call_123",
    }


def test_convert_message_with_tool_result():
    """Test that tool result messages are correctly formatted."""
    tool_call_id_suffix = int(time.time() * 1000)
    messages = [
        Message(
            role=MessageRole.TOOL,
            content="Weather is sunny",
            tool_call_id=f"ollama-tool-{tool_call_id_suffix}",
        )
    ]
    ollama_messages = OllamaProvider._convert_to_ollama_messages(messages)  # type: ignore
    assert ollama_messages == [
        {
            "role": "tool",
            "content": "Weather is sunny",
            "tool_call_id": f"ollama-tool-{tool_call_id_suffix}",
        }
    ]


@patch("allos.providers.ollama.Client")
def test_assistant_message_with_content_and_tool_calls(MockClient):
    """Test that assistant messages with both content and tool calls are handled."""
    mock_instance = MockClient.return_value
    mock_instance.list.return_value = MOCK_MODEL_LIST
    mock_instance.show.return_value = create_mock_show_response("default")

    provider = OllamaProvider(model="llama3:latest")

    messages = [
        Message(
            role=MessageRole.ASSISTANT,
            content="Let me check the weather...",
            tool_calls=[
                ToolCall(
                    id="call_123", name="get_weather", arguments={"location": "NYC"}
                )
            ],
        )
    ]

    ollama_messages = provider._convert_to_ollama_messages(messages)  # type: ignore

    assert len(ollama_messages) == 1
    assert ollama_messages[0]["role"] == "assistant"
    assert ollama_messages[0]["content"] == "Let me check the weather..."
    assert "tool_calls" in ollama_messages[0]
    assert ollama_messages[0]["tool_calls"][0]["function"]["name"] == "get_weather"


@patch("allos.providers.ollama.Client")
def test_convert_messages_handles_empty_content(MockClient):
    """Test that messages with empty/None content are handled gracefully."""
    mock_instance = MockClient.return_value
    mock_instance.list.return_value = MOCK_MODEL_LIST
    mock_instance.show.return_value = create_mock_show_response("default")

    provider = OllamaProvider(model="llama3:latest")

    messages = [
        Message(role=MessageRole.USER, content=None),
        Message(role=MessageRole.ASSISTANT, content=""),
    ]

    ollama_messages = provider._convert_to_ollama_messages(messages)  # type: ignore

    assert ollama_messages[0]["content"] == ""
    assert ollama_messages[1]["content"] == ""


@patch("allos.providers.ollama.Client")
def test_chat_with_tools_supported_model(MockClient):
    """Tests that tools are passed to a supported model."""
    mock_instance = MockClient.return_value
    mock_instance.list.return_value = MOCK_MODEL_LIST
    # Mock show() to return tool support capability
    mock_instance.show.return_value = create_mock_show_response(
        "qwen3:8b", supports_tools=True, context_length=40960
    )
    mock_instance.chat.return_value = MOCK_TOOL_CALL_RESPONSE

    provider = OllamaProvider(model="qwen3:8b")

    response = provider.chat(messages=[], tools=MOCK_TOOLS)

    mock_instance.chat.assert_called_once()
    # Check that 'tools' was in the arguments passed to the client
    assert "tools" in mock_instance.chat.call_args.kwargs
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "get_weather"
    assert response.tool_calls[0].arguments == {"location": "Boston"}


@patch("allos.providers.ollama.Client")
def test_chat_with_tools_unsupported_model(MockClient, caplog):
    """Tests that a warning is logged for unsupported models and tools are ignored."""
    mock_instance = MockClient.return_value
    mock_instance.list.return_value = MOCK_MODEL_LIST
    # Mock show() to return NO tool support
    mock_instance.show.return_value = create_mock_show_response(
        "llama3:latest", supports_tools=False
    )
    mock_instance.chat.return_value = (
        MOCK_CHAT_RESPONSE  # Returns a simple text response
    )

    provider = OllamaProvider(
        model="llama3:latest"
    )  # llama3 is not in our supported set
    provider.chat(messages=[], tools=MOCK_TOOLS)

    # Assert that a warning was logged
    assert "does not support native tool calling" in caplog.text
    # Assert that 'tools' was NOT passed to the client
    assert "tools" not in mock_instance.chat.call_args.kwargs


@patch("allos.providers.ollama.Client")
def test_stream_chat_text_response(MockClient):
    """Tests a successful streaming call with a text response."""
    mock_instance = MockClient.return_value
    mock_instance.list.return_value = MOCK_MODEL_LIST
    mock_instance.show.return_value = create_mock_show_response("default")
    mock_instance.chat.return_value = iter(MOCK_TEXT_STREAM)

    provider = OllamaProvider(model="llama3:latest")
    stream = provider.stream_chat(messages=[])

    chunks = list(stream)
    assert len(chunks) == 3
    assert chunks[0].content == "Hel"
    assert chunks[1].content == "lo!"
    assert chunks[2].final_metadata is not None
    assert chunks[2].final_metadata.usage.output_tokens == 2


@patch("allos.providers.ollama.Client")
def test_stream_chat_tool_response(MockClient):
    """Tests a successful streaming call with a tool call response."""
    mock_instance = MockClient.return_value
    mock_instance.list.return_value = MOCK_MODEL_LIST
    mock_instance.show.return_value = create_mock_show_response("default")
    mock_instance.chat.return_value = iter(MOCK_TOOL_STREAM)

    OLLAMA_TOOL_SUPPORTED_MODELS.add("qwen2")
    provider = OllamaProvider(model="qwen2:latest")
    stream = provider.stream_chat(messages=[], tools=MOCK_TOOLS)

    chunks = list(stream)
    assert len(chunks) == 2
    assert chunks[0].tool_call_done is not None
    assert chunks[0].tool_call_done.name == "get_weather"
    assert chunks[0].tool_call_done.arguments == {"location": "San Francisco"}
    assert chunks[1].final_metadata is not None
    assert chunks[1].final_metadata.usage.output_tokens == 15


@patch("allos.providers.ollama.Client")
def test_convert_tools_to_ollama_format(MockClient):
    """Test that tools are correctly converted to Ollama's expected format."""
    mock_instance = MockClient.return_value
    mock_instance.list.return_value = MOCK_MODEL_LIST
    mock_instance.show.return_value = create_mock_show_response("default")

    provider = OllamaProvider(model="qwen3:8b")

    tools = [MockTool()]
    ollama_tools = provider._convert_tools_to_ollama_format(tools)  # type: ignore

    assert len(ollama_tools) == 1
    tool_def = ollama_tools[0]
    assert tool_def["function"] is not None
    assert tool_def["function"]["name"] == "get_weather"
    assert "location" in tool_def["function"]["parameters"]["properties"]


@patch("allos.providers.ollama.Client")
def test_chat_handles_response_with_no_message_field(MockClient):
    """Test that a response missing the 'message' field is handled gracefully."""
    mock_instance = MockClient.return_value
    mock_instance.list.return_value = MOCK_MODEL_LIST
    mock_instance.show.return_value = create_mock_show_response("default")
    mock_instance.chat.return_value = {
        "model": "llama3:latest",
        "done": True,
        # No 'message' field
    }

    provider = OllamaProvider(model="llama3:latest")
    response = provider.chat([Message(role=MessageRole.USER, content="Hi")])

    assert response.content is None
    assert len(response.tool_calls) == 0


@patch("allos.providers.ollama.Client")
def test_chat_handles_tool_call_with_missing_id(MockClient):
    """Test that tool calls without IDs get auto-generated IDs."""
    mock_instance = MockClient.return_value
    mock_instance.list.return_value = MOCK_MODEL_LIST
    mock_instance.show.return_value = create_mock_show_response("default")
    mock_instance.chat.return_value = {
        "model": "qwen3:8b",
        "message": {
            "role": "assistant",
            "tool_calls": [
                {
                    # No 'id' field
                    "function": {
                        "name": "get_weather",
                        "arguments": {"location": "Boston"},
                    }
                }
            ],
        },
        "done": True,
    }

    provider = OllamaProvider(model="qwen3:8b")
    response = provider.chat([])

    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].id.startswith("ollama-tool-")
    assert response.tool_calls[0].name == "get_weather"


@patch("allos.providers.ollama.Client")
def test_chat_handles_missing_usage_fields(MockClient):
    """Test that missing usage fields default to 0 in metadata."""
    mock_instance = MockClient.return_value
    mock_instance.list.return_value = MOCK_MODEL_LIST
    mock_instance.show.return_value = create_mock_show_response("default")
    mock_instance.chat.return_value = {
        "model": "llama3:latest",
        "message": {"role": "assistant", "content": "Hello"},
        "done": True,
        # Missing prompt_eval_count and eval_count
    }

    provider = OllamaProvider(model="llama3:latest")
    response = provider.chat([])

    assert response.metadata.usage.input_tokens == 0
    assert response.metadata.usage.output_tokens == 0
    assert response.metadata.usage.total_tokens == 0


@pytest.mark.parametrize(
    "ollama_error, expected_message_part",
    [
        (ResponseError("Model not found", status_code=404), "Model not found"),
        (
            ResponseError("Rate limit exceeded", status_code=429),
            "Rate limit exceeded",
        ),
        (RequestError("Connection timeout"), "Connection timeout"),
        (ResponseError("Model not loaded", status_code=500), "Model not loaded"),
        (Exception("Unexpected error"), "Unexpected error"),
    ],
)
@patch("allos.providers.ollama.Client")
def test_chat_error_handling(MockClient, ollama_error, expected_message_part):
    """Test that various Ollama errors are properly wrapped in ProviderError."""
    mock_instance = MockClient.return_value
    mock_instance.list.return_value = MOCK_MODEL_LIST
    mock_instance.show.return_value = create_mock_show_response("default")
    mock_instance.chat.side_effect = ollama_error

    provider = OllamaProvider(model="llama3:latest")

    with pytest.raises(ProviderError) as excinfo:
        provider.chat([Message(role=MessageRole.USER, content="Hi")])

    assert expected_message_part in str(excinfo.value)
    assert excinfo.value.provider == "ollama"


@patch("allos.providers.ollama.Client")
def test_chat_filters_unsupported_parameters(MockClient):
    """Test that only OLLAMA_SUPPORTED_OPTIONS are passed to the client."""
    mock_instance = MockClient.return_value
    mock_instance.list.return_value = MOCK_MODEL_LIST
    mock_instance.show.return_value = create_mock_show_response("default")
    mock_instance.chat.return_value = MOCK_CHAT_RESPONSE

    provider = OllamaProvider(model="llama3:latest")

    # Mix of supported and unsupported parameters
    provider.chat(
        [],
        temperature=0.7,  # Supported
        top_p=0.9,  # Supported
        max_tokens=100,  # NOT in OLLAMA_SUPPORTED_OPTIONS
        custom_param="value",  # NOT in OLLAMA_SUPPORTED_OPTIONS
    )

    call_kwargs = mock_instance.chat.call_args.kwargs
    options = call_kwargs.get("options", {})

    assert "temperature" in options
    assert "top_p" in options
    assert "max_tokens" not in options
    assert "custom_param" not in options


@patch("allos.providers.ollama.Client")
def test_stream_chat_handles_missing_message_field(MockClient):
    """Test streaming with chunks missing the 'message' field."""
    mock_instance = MockClient.return_value
    mock_instance.list.return_value = MOCK_MODEL_LIST
    mock_instance.show.return_value = create_mock_show_response("default")
    mock_instance.chat.return_value = iter(
        [
            {"done": False},  # No 'message' field
            {"message": {"role": "assistant", "content": "Hello"}, "done": False},
            {"done": True, "model": "llama3:latest"},
        ]
    )

    provider = OllamaProvider(model="llama3:latest")
    chunks = list(provider.stream_chat([]))

    # Should handle gracefully and still process valid chunks
    content_chunks = [c for c in chunks if c.content]
    assert len(content_chunks) == 1
    assert content_chunks[0].content == "Hello"


@patch("allos.providers.ollama.Client")
def test_stream_chat_handles_multiple_tool_calls_in_chunk(MockClient):
    """Test streaming with multiple tool calls in a single chunk."""
    mock_instance = MockClient.return_value
    mock_instance.list.return_value = MOCK_MODEL_LIST
    mock_instance.show.return_value = create_mock_show_response("default")
    mock_instance.chat.return_value = iter(
        [
            {
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {"function": {"name": "tool1", "arguments": {"a": 1}}},
                        {"function": {"name": "tool2", "arguments": {"b": 2}}},
                    ],
                },
                "done": False,
            },
            {"done": True, "model": "qwen3:8b"},
        ]
    )

    OLLAMA_TOOL_SUPPORTED_MODELS.add("qwen3")
    provider = OllamaProvider(model="qwen3:8b")

    chunks = list(provider.stream_chat([]))
    tool_chunks = [c for c in chunks if c.tool_call_done]

    assert len(tool_chunks) == 2
    assert (
        tool_chunks[0].tool_call_done and tool_chunks[0].tool_call_done.name == "tool1"
    )
    assert (
        tool_chunks[1].tool_call_done and tool_chunks[1].tool_call_done.name == "tool2"
    )


@patch("allos.providers.ollama.Client")
def test_stream_chat_with_content_and_tool_calls_same_chunk(MockClient):
    """Test streaming chunk containing both content and tool calls."""
    mock_instance = MockClient.return_value
    mock_instance.list.return_value = MOCK_MODEL_LIST
    mock_instance.show.return_value = create_mock_show_response("default")
    mock_instance.chat.return_value = iter(
        [
            {
                "message": {
                    "role": "assistant",
                    "content": "Calling tool...",
                    "tool_calls": [
                        {"function": {"name": "get_weather", "arguments": {}}}
                    ],
                },
                "done": False,
            },
            {"done": True, "model": "qwen3:8b"},
        ]
    )

    provider = OllamaProvider(model="qwen3:8b")
    chunks = list(provider.stream_chat([]))

    content_chunks = [c for c in chunks if c.content]
    tool_chunks = [c for c in chunks if c.tool_call_done]

    assert len(content_chunks) == 1
    assert len(tool_chunks) == 1


@pytest.mark.parametrize(
    "model_name, should_support",
    [
        ("llama3.1:latest", True),
        ("qwen2:7b", True),
        ("qwen3:8b", True),
        ("llama3:latest", False),
        ("mistral:latest", False),
        ("gemma:latest", False),
    ],
)
@patch("allos.providers.ollama.Client")
def test_model_supports_tools(MockClient, model_name, should_support):
    """Test that _model_supports_tools correctly identifies tool-capable models."""
    mock_instance = MockClient.return_value
    mock_instance.list.return_value = ListResponse(
        models=[ListResponse.Model(model=model_name)]
    )
    # Mock show() to return appropriate tool support
    mock_instance.show.return_value = create_mock_show_response(
        model_name, supports_tools=should_support
    )

    provider = OllamaProvider(model=model_name)

    assert provider._model_supports_tools() == should_support  # type: ignore


@patch("allos.providers.ollama.Client")
def test_get_context_window_handles_version_tags(MockClient):
    """Test context window detection with version tags like :latest, :7b, etc."""
    mock_instance = MockClient.return_value

    # Test :latest tag removal
    mock_instance.list.return_value = ListResponse(
        models=[ListResponse.Model(model="llama3.1:latest")]
    )
    # Mock show() to return context length
    mock_instance.show.return_value = create_mock_show_response(
        "llama3.1:latest", context_length=128000
    )
    provider = OllamaProvider(model="llama3.1:latest")
    assert provider.get_context_window() == 128000

    # Test specific version tags (shouldn't match)
    mock_instance.list.return_value = ListResponse(
        models=[ListResponse.Model(model="llama3:13b")]
    )
    mock_instance.show.return_value = create_mock_show_response("default")
    provider = OllamaProvider(model="llama3:13b")
    assert provider.get_context_window() == 8192


@patch("allos.providers.ollama.Client")
def test_extract_model_family_no_colon(MockClient):
    """Test that models without a colon are returned as-is."""
    mock_instance = MockClient.return_value
    mock_instance.list.return_value = ListResponse(
        models=[ListResponse.Model(model="llama3")]
    )
    mock_instance.show.return_value = create_mock_show_response("default")

    provider = OllamaProvider(model="llama3")

    # This should hit the "return self.model" branch
    family = provider._extract_model_family()  # type: ignore
    assert family == "llama3"


@patch("allos.providers.ollama.Client")
def test_extract_model_family_special_case_in_mappings(MockClient):
    """Test that special cases like 'qwen3:8b' are preserved."""
    mock_instance = MockClient.return_value
    mock_instance.list.return_value = ListResponse(
        models=[ListResponse.Model(model="qwen3:8b")]
    )
    mock_instance.show.return_value = create_mock_show_response("default")

    provider = OllamaProvider(model="qwen3:8b")

    # qwen3:8b is in OLLAMA_CONTEXT_WINDOWS, so should return full name
    family = provider._extract_model_family()  # type: ignore
    assert family == "qwen3:8b"


@patch("allos.providers.ollama.Client")
def test_stream_chat_warns_on_unsupported_model_with_tools(MockClient, caplog):
    """Test that streaming with tools on unsupported model logs a warning."""
    mock_instance = MockClient.return_value
    mock_instance.list.return_value = MOCK_MODEL_LIST
    # Mock show() to return NO tool support
    mock_instance.show.return_value = create_mock_show_response(
        "llama3:latest", supports_tools=False
    )
    mock_instance.chat.return_value = iter(
        [
            {"message": {"role": "assistant", "content": "Hi"}, "done": False},
            {"done": True, "model": "llama3:latest"},
        ]
    )

    # llama3 does NOT support tools
    provider = OllamaProvider(model="llama3:latest")

    # Provide tools to an unsupported model
    list(provider.stream_chat(messages=[], tools=MOCK_TOOLS))

    # Assert the warning was logged
    assert "does not support native tool calling" in caplog.text

    # Assert 'tools' was NOT passed to the client
    call_kwargs = mock_instance.chat.call_args.kwargs
    assert "tools" not in call_kwargs


@patch("allos.providers.ollama.Client")
def test_stream_chat_handles_response_error(MockClient):
    """Test that ResponseError during streaming yields an error chunk."""
    mock_instance = MockClient.return_value
    mock_instance.list.return_value = MOCK_MODEL_LIST
    mock_instance.show.return_value = create_mock_show_response("default")

    # Simulate a ResponseError during streaming
    mock_instance.chat.side_effect = ResponseError("Model crashed", status_code=500)

    provider = OllamaProvider(model="llama3:latest")
    chunks = list(provider.stream_chat([Message(role=MessageRole.USER, content="Hi")]))

    # Should yield exactly one error chunk
    assert len(chunks) == 1
    assert chunks[0].error is not None
    assert "Ollama API error" in chunks[0].error
    assert "Model crashed" in chunks[0].error


@patch("allos.providers.ollama.Client")
def test_stream_chat_handles_request_error(MockClient):
    """Test that RequestError during streaming yields an error chunk."""
    mock_instance = MockClient.return_value
    mock_instance.list.return_value = MOCK_MODEL_LIST
    mock_instance.show.return_value = create_mock_show_response("default")

    # Simulate a RequestError (connection issue)
    mock_instance.chat.side_effect = RequestError("Connection lost")

    provider = OllamaProvider(model="llama3:latest")
    chunks = list(provider.stream_chat([Message(role=MessageRole.USER, content="Hi")]))

    # Should yield exactly one error chunk
    assert len(chunks) == 1
    assert chunks[0].error is not None
    assert "Ollama connection error" in chunks[0].error
    assert "Connection lost" in chunks[0].error


@patch("allos.providers.ollama.Client")
def test_get_context_window_unknown_model_fallback(MockClient, caplog):
    """Test that unknown models fall back to 4096 with a warning."""
    mock_instance = MockClient.return_value
    mock_instance.list.return_value = ListResponse(
        models=[ListResponse.Model(model="totally-unknown-model:v1")]
    )
    # Mock show() to return NO context length
    mock_instance.show.return_value = create_mock_show_response(
        "totally-unknown-model:v1"
    )

    provider = OllamaProvider(model="totally-unknown-model:v1")

    context_window = provider.get_context_window()

    # Should fall back to default
    assert context_window == 4096

    # Should log a warning
    assert "Unknown context window" in caplog.text
    assert "totally-unknown-model:v1" in caplog.text
    assert "Falling back to 4096" in caplog.text


@patch("allos.providers.ollama.Client")
def test_get_context_window_prefix_match(MockClient):
    """Test that prefix matching works for context window detection."""
    mock_instance = MockClient.return_value
    mock_instance.list.return_value = ListResponse(
        models=[ListResponse.Model(model="mistral:7b-instruct")]
    )
    # Mock show() to return context length
    mock_instance.show.return_value = create_mock_show_response(
        "mistral:7b-instruct", context_length=32768
    )

    provider = OllamaProvider(model="mistral:7b-instruct")

    # Should match "mistral" via retrieved context
    assert provider.get_context_window() == 32768


@patch("allos.providers.ollama.Client")
def test_extract_model_family_strips_common_tags(MockClient):
    """Test that common tags like 'instruct' and 'chat' are stripped."""
    mock_instance = MockClient.return_value

    # Test 'instruct' tag
    mock_instance.list.return_value = ListResponse(
        models=[ListResponse.Model(model="mistral:instruct")]
    )
    mock_instance.show.return_value = create_mock_show_response("default")
    provider = OllamaProvider(model="mistral:instruct")
    family = provider._extract_model_family()  # type: ignore
    assert family == "mistral"

    # Test 'chat' tag
    mock_instance.list.return_value = ListResponse(
        models=[ListResponse.Model(model="llama3:chat")]
    )
    mock_instance.show.return_value = create_mock_show_response("default")
    provider = OllamaProvider(model="llama3:chat")
    family = provider._extract_model_family()  # type: ignore
    assert family == "llama3"


@patch("allos.providers.ollama.Client")
def test_extract_model_family_strips_size_tags(MockClient):
    """Test that size tags like '7b', '13b', '70b' are stripped."""
    mock_instance = MockClient.return_value

    test_cases = [
        ("llama3:7b", "llama3"),
        ("mistral:13b", "mistral"),
        ("mixtral:70b", "mixtral"),
    ]

    for model, expected_family in test_cases:
        mock_instance.list.return_value = ListResponse(
            models=[ListResponse.Model(model=model)]
        )
        mock_instance.show.return_value = create_mock_show_response("default")
        provider = OllamaProvider(model=model)
        family = provider._extract_model_family()  # type: ignore
        assert family == expected_family, f"Failed for model: {model}"


@patch("allos.providers.ollama.Client")
def test_model_supports_tools_with_prefix_match(MockClient):
    """Test that tool support detection works with prefix matching."""
    mock_instance = MockClient.return_value

    # llama3.1 should support tools (even with custom suffix)
    mock_instance.list.return_value = ListResponse(
        models=[ListResponse.Model(model="llama3.1:custom")]
    )
    # Mock show() to return tool support
    mock_instance.show.return_value = create_mock_show_response(
        "llama3.1:custom", supports_tools=True
    )
    provider = OllamaProvider(model="llama3.1:custom")
    assert provider._model_supports_tools()  # type: ignore

    # llama3 should NOT support tools
    mock_instance.list.return_value = ListResponse(
        models=[ListResponse.Model(model="llama3:latest")]
    )
    # Mock show() to return NO tool support
    mock_instance.show.return_value = create_mock_show_response(
        "llama3:latest", supports_tools=False
    )
    provider = OllamaProvider(model="llama3:latest")
    assert not provider._model_supports_tools()  # type: ignore


@patch("allos.providers.ollama.Client")
def test_verify_model_available_handles_show_exception(MockClient):
    """Test that exception during show() call is handled gracefully."""
    mock_instance = MockClient.return_value
    mock_instance.list.return_value = MOCK_MODEL_LIST
    # Simulate show() raising an exception
    mock_instance.show.side_effect = Exception("Failed to get model info")

    # Should still initialize successfully (non-fatal error)
    provider = OllamaProvider(model="llama3:latest")
    assert provider.model == "llama3:latest"
    # Capabilities should be None since show() failed
    assert provider._model_context_window is None  # type: ignore
    assert provider._model_supports_tools_capability is None  # type: ignore


@patch("allos.providers.ollama.Client")
def test_model_supports_tools_fallback_to_family_check(MockClient):
    """Test that tool support falls back to family check when show() fails."""
    mock_instance = MockClient.return_value
    mock_instance.list.return_value = ListResponse(
        models=[ListResponse.Model(model="llama3.1:latest")]
    )
    # Simulate show() failing to set capability (exception handled)
    mock_instance.show.side_effect = Exception("Failed to get info")

    provider = OllamaProvider(model="llama3.1:latest")

    # Capability should be None, triggering fallback to family check
    # llama3.1 is in OLLAMA_TOOL_SUPPORTED_MODELS, so should return True
    assert provider._model_supports_tools()  # type: ignore


@patch("allos.providers.ollama.Client")
def test_stream_chat_with_tools_supported_model(MockClient):
    """Test streaming with tools on a supported model."""
    mock_instance = MockClient.return_value
    mock_instance.list.return_value = MOCK_MODEL_LIST
    # Mock show() to return tool support
    mock_instance.show.return_value = create_mock_show_response(
        "qwen3:8b", supports_tools=True
    )
    mock_instance.chat.return_value = iter(
        [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "get_weather",
                                "arguments": {"location": "NYC"},
                            }
                        }
                    ],
                },
                "done": False,
            },
            {"done": True, "model": "qwen3:8b"},
        ]
    )

    provider = OllamaProvider(model="qwen3:8b")
    chunks = list(provider.stream_chat(messages=[], tools=MOCK_TOOLS))

    # Verify tools were passed to the chat call
    call_kwargs = mock_instance.chat.call_args.kwargs
    assert "tools" in call_kwargs

    # Verify we got chunks back
    tool_chunks = [c for c in chunks if c.tool_call_done]
    assert len(tool_chunks) == 1


@patch("allos.providers.ollama.Client")
def test_get_context_window_uses_prefix_match_fallback(MockClient):
    """Test that context window detection falls back to prefix matching."""
    mock_instance = MockClient.return_value
    mock_instance.list.return_value = ListResponse(
        models=[ListResponse.Model(model="llama3:custom-variant")]
    )
    # Mock show() to return no context length info
    mock_instance.show.return_value = create_mock_show_response("llama3:custom-variant")

    provider = OllamaProvider(model="llama3:custom-variant")

    # Should match "llama3" prefix and return 8192
    context_window = provider.get_context_window()
    assert context_window == 8192


@patch("allos.providers.ollama.Client")
def test_retrieve_context_length_handles_exceptions(MockClient):
    """Test that retrieve_context_length handles AttributeError and TypeError."""
    from allos.providers.ollama import retrieve_context_length

    # Test with dict that raises AttributeError
    bad_model_info = {"broken": "data"}
    result = retrieve_context_length(bad_model_info)
    # Should return None when no valid path found
    assert result is None

    # Test with None (will raise TypeError)
    result = retrieve_context_length(None)  # type: ignore
    assert result is None


@patch("allos.providers.ollama.Client")
def test_retrieve_context_length_finds_num_ctx(MockClient):
    """Test that retrieve_context_length successfully extracts num_ctx."""
    from allos.providers.ollama import retrieve_context_length

    # Test direct num_ctx
    model_info = {"num_ctx": 8192}
    result = retrieve_context_length(model_info)
    assert result == 8192

    # Test nested in details
    model_info = {"details": {"num_ctx": 16384}}
    result = retrieve_context_length(model_info)
    assert result == 16384

    # Test nested in parameters
    model_info = {"parameters": {"num_ctx": 32768}}
    result = retrieve_context_length(model_info)
    assert result == 32768

    # Test with context_length suffix
    model_info = {"model_context_length": 128000}
    result = retrieve_context_length(model_info)
    assert result == 128000
