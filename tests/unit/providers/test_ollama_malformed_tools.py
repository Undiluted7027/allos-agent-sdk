# tests/unit/providers/test_ollama_malformed_tools.py

"""Tests for malformed tool call handling in Ollama provider.

These tests verify that the provider gracefully handles various malformed
tool call responses from the Ollama API without crashing.
"""

from typing import Any, Dict
from unittest.mock import patch

import pytest
from ollama._types import ListResponse

from allos.providers.base import Message, MessageRole
from allos.providers.ollama import _OLLAMA_CLIENT_POOL, OllamaProvider
from allos.tools.base import BaseTool, ToolParameter


@pytest.fixture(autouse=True)
def clear_connection_pool():
    """Clear the connection pool before and after each test."""
    _OLLAMA_CLIENT_POOL.clear()
    yield
    _OLLAMA_CLIENT_POOL.clear()


# Mock model list
MOCK_MODEL_LIST: ListResponse = ListResponse(
    models=[ListResponse.Model(model="qwen2.5:latest")]
)


def create_mock_show_response(
    model: str = "qwen2.5:latest", context_length: int = 8192
):
    """Create a mock ShowResponse with tool support."""
    from unittest.mock import Mock

    mock_response = Mock()
    mock_response.model_dump.return_value = {
        "modelfile": f"... num_ctx {context_length} ...",
        "parameters": f"num_ctx {context_length}\nstop [INST]\nstop [/INST]",
        "template": "...",
        "details": {
            "parameter_size": "7B",
            "families": ["qwen2"],
        },
    }
    # Mock tool support
    mock_response.model_info = {"general.file_type": 1}
    return mock_response


# Create a simple mock tool for testing
class MockWeatherTool(BaseTool):
    """Mock weather tool for testing."""

    name: str = "get_weather"
    description: str = "Get weather for a location"
    parameters: list[ToolParameter] = [
        ToolParameter(
            name="location",
            type="string",
            required=True,
            description="Location to get weather for",
        )
    ]

    def execute(self, **kwargs) -> Dict[str, Any]:
        """Mock execution."""
        return {"status": "success", "result": "sunny"}


MOCK_TOOLS = [MockWeatherTool()]


@patch("allos.providers.ollama.Client")
class TestMalformedToolCallsChat:
    """Test malformed tool calls in synchronous chat()."""

    def test_missing_function_field(self, MockClient, caplog):
        """Test tool call missing 'function' field."""
        mock_instance = MockClient.return_value
        mock_instance.list.return_value = MOCK_MODEL_LIST
        mock_instance.show.return_value = create_mock_show_response()

        # Tool call missing 'function' key
        mock_instance.chat.return_value = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_123"}  # Missing 'function'
                ],
            },
            "done": True,
        }

        provider = OllamaProvider(model="qwen2.5:latest")
        messages = [Message(role=MessageRole.USER, content="What's the weather?")]

        response = provider.chat(messages, tools=MOCK_TOOLS)

        # Should not crash, return empty tool_calls with warning
        assert response.tool_calls == []
        assert "missing 'function' field" in caplog.text

    def test_function_not_dict(self, MockClient, caplog):
        """Test tool call with 'function' that's not a dict.

        When function is a string, getattr() returns None for name/arguments,
        so the code logs 'missing function.name' rather than 'not a dict'.
        """
        mock_instance = MockClient.return_value
        mock_instance.list.return_value = MOCK_MODEL_LIST
        mock_instance.show.return_value = create_mock_show_response()

        # 'function' is a string instead of dict
        mock_instance.chat.return_value = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call_123", "function": "not_a_dict"}],
            },
            "done": True,
        }

        provider = OllamaProvider(model="qwen2.5:latest")
        messages = [Message(role=MessageRole.USER, content="What's the weather?")]

        response = provider.chat(messages, tools=MOCK_TOOLS)

        assert response.tool_calls == []
        # When function is not a dict/object with attributes, name extraction fails
        assert "missing 'function.name'" in caplog.text

    def test_missing_function_name(self, MockClient, caplog):
        """Test tool call missing 'function.name' field."""
        mock_instance = MockClient.return_value
        mock_instance.list.return_value = MOCK_MODEL_LIST
        mock_instance.show.return_value = create_mock_show_response()

        # Missing 'name' in function
        mock_instance.chat.return_value = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "function": {
                            "arguments": '{"location": "NYC"}',
                            # Missing 'name'
                        },
                    }
                ],
            },
            "done": True,
        }

        provider = OllamaProvider(model="qwen2.5:latest")
        messages = [Message(role=MessageRole.USER, content="What's the weather?")]

        response = provider.chat(messages, tools=MOCK_TOOLS)

        assert response.tool_calls == []
        assert "missing 'function.name'" in caplog.text

    def test_missing_function_arguments(self, MockClient, caplog):
        """Test tool call missing 'function.arguments' field."""
        mock_instance = MockClient.return_value
        mock_instance.list.return_value = MOCK_MODEL_LIST
        mock_instance.show.return_value = create_mock_show_response()

        # Missing 'arguments' in function
        mock_instance.chat.return_value = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "function": {
                            "name": "get_weather",
                            # Missing 'arguments'
                        },
                    }
                ],
            },
            "done": True,
        }

        provider = OllamaProvider(model="qwen2.5:latest")
        messages = [Message(role=MessageRole.USER, content="What's the weather?")]

        response = provider.chat(messages, tools=MOCK_TOOLS)

        assert response.tool_calls == []
        assert "missing 'function.arguments'" in caplog.text

    def test_tool_calls_not_list(self, MockClient, caplog):
        """Test tool_calls field is not a list."""
        mock_instance = MockClient.return_value
        mock_instance.list.return_value = MOCK_MODEL_LIST
        mock_instance.show.return_value = create_mock_show_response()

        # tool_calls is a dict instead of list
        mock_instance.chat.return_value = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": {"not": "a list"},  # Should be list
            },
            "done": True,
        }

        provider = OllamaProvider(model="qwen2.5:latest")
        messages = [Message(role=MessageRole.USER, content="What's the weather?")]

        response = provider.chat(messages, tools=MOCK_TOOLS)

        assert response.tool_calls == []
        assert "expected list, got dict" in caplog.text

    def test_tool_call_not_dict(self, MockClient, caplog):
        """Test tool call entry is not a dict.

        When tool_call is a string, the code treats it as a typed object and
        uses getattr() which returns None, so it logs 'missing function attribute'.
        """
        mock_instance = MockClient.return_value
        mock_instance.list.return_value = MOCK_MODEL_LIST
        mock_instance.show.return_value = create_mock_show_response()

        # Tool call is a string instead of dict
        mock_instance.chat.return_value = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": ["not_a_dict"],  # Should be list of dicts
            },
            "done": True,
        }

        provider = OllamaProvider(model="qwen2.5:latest")
        messages = [Message(role=MessageRole.USER, content="What's the weather?")]

        response = provider.chat(messages, tools=MOCK_TOOLS)

        assert response.tool_calls == []
        # When tool_call is not a dict/typed-object, getattr returns None for function
        assert "missing 'function' attribute" in caplog.text

    def test_partial_success_mixed_valid_invalid(self, MockClient, caplog):
        """Test mix of valid and invalid tool calls - should parse valid ones."""
        mock_instance = MockClient.return_value
        mock_instance.list.return_value = MOCK_MODEL_LIST
        mock_instance.show.return_value = create_mock_show_response()

        # Mix of valid and malformed tool calls
        mock_instance.chat.return_value = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "NYC"}',
                        },
                    },  # Valid
                    {"id": "call_456"},  # Invalid - missing function
                    {
                        "id": "call_789",
                        "function": {
                            "name": "get_time",
                            "arguments": "{}",
                        },
                    },  # Valid
                ],
            },
            "done": True,
        }

        provider = OllamaProvider(model="qwen2.5:latest")
        messages = [Message(role=MessageRole.USER, content="What's the weather?")]

        response = provider.chat(messages, tools=MOCK_TOOLS)

        # Should have parsed 2 valid tool calls
        assert len(response.tool_calls) == 2
        assert response.tool_calls[0].name == "get_weather"
        assert response.tool_calls[1].name == "get_time"
        assert "missing 'function' field" in caplog.text

    def test_all_invalid_logs_error(self, MockClient, caplog):
        """Test that when all tool calls are invalid, an error is logged."""
        mock_instance = MockClient.return_value
        mock_instance.list.return_value = MOCK_MODEL_LIST
        mock_instance.show.return_value = create_mock_show_response()

        # All tool calls are malformed
        mock_instance.chat.return_value = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_123"},  # Missing function
                    {"id": "call_456", "function": "not_dict"},  # function not dict
                ],
            },
            "done": True,
        }

        provider = OllamaProvider(model="qwen2.5:latest")
        messages = [Message(role=MessageRole.USER, content="What's the weather?")]

        response = provider.chat(messages, tools=MOCK_TOOLS)

        assert response.tool_calls == []
        assert "all were malformed" in caplog.text

    def test_unexpected_exception_in_parsing(self, MockClient, caplog):
        """Test that unexpected exceptions during parsing are caught."""
        mock_instance = MockClient.return_value
        mock_instance.list.return_value = MOCK_MODEL_LIST
        mock_instance.show.return_value = create_mock_show_response()

        # Valid structure that might cause unexpected error
        mock_instance.chat.return_value = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "NYC"}',
                        },
                    }
                ],
            },
            "done": True,
        }

        provider = OllamaProvider(model="qwen2.5:latest")

        # Patch ToolCall to raise exception
        with patch(
            "allos.providers.ollama.ToolCall", side_effect=ValueError("Test error")
        ):
            messages = [Message(role=MessageRole.USER, content="What's the weather?")]
            response = provider.chat(messages, tools=MOCK_TOOLS)

            assert response.tool_calls == []
            assert "Unexpected error parsing tool call" in caplog.text


@patch("allos.providers.ollama.Client")
class TestMalformedToolCallsStreaming:
    """Test malformed tool calls in streaming mode."""

    def test_stream_missing_function_field(self, MockClient, caplog):
        """Test streaming tool call missing 'function' field."""
        mock_instance = MockClient.return_value
        mock_instance.list.return_value = MOCK_MODEL_LIST
        mock_instance.show.return_value = create_mock_show_response()

        # Stream with malformed tool call
        mock_instance.chat.return_value = iter(
            [
                {
                    "message": {"role": "assistant", "content": ""},
                    "done": False,
                },
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"id": "call_123"}],  # Missing 'function'
                    },
                    "done": False,
                },
                {
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                },
            ]
        )

        provider = OllamaProvider(model="qwen2.5:latest")
        messages = [Message(role=MessageRole.USER, content="What's the weather?")]

        chunks = list(provider.stream_chat(messages, tools=MOCK_TOOLS))

        # Should not crash, no tool_call_done chunks
        tool_chunks = [c for c in chunks if c.tool_call_done]
        assert len(tool_chunks) == 0
        assert "missing 'function' field" in caplog.text

    def test_stream_tool_calls_not_list(self, MockClient, caplog):
        """Test streaming with tool_calls that's not a list."""
        mock_instance = MockClient.return_value
        mock_instance.list.return_value = MOCK_MODEL_LIST
        mock_instance.show.return_value = create_mock_show_response()

        mock_instance.chat.return_value = iter(
            [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": "not_a_list",  # Should be list
                    },
                    "done": False,
                },
                {
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                },
            ]
        )

        provider = OllamaProvider(model="qwen2.5:latest")
        messages = [Message(role=MessageRole.USER, content="What's the weather?")]

        chunks = list(provider.stream_chat(messages, tools=MOCK_TOOLS))

        tool_chunks = [c for c in chunks if c.tool_call_done]
        assert len(tool_chunks) == 0
        assert "expected list, got str" in caplog.text

    def test_stream_partial_success(self, MockClient, caplog):
        """Test streaming with mix of valid and invalid tool calls."""
        mock_instance = MockClient.return_value
        mock_instance.list.return_value = MOCK_MODEL_LIST
        mock_instance.show.return_value = create_mock_show_response()

        mock_instance.chat.return_value = iter(
            [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "NYC"}',
                                },
                            },  # Valid
                            {"id": "call_456"},  # Invalid
                        ],
                    },
                    "done": False,
                },
                {
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                },
            ]
        )

        provider = OllamaProvider(model="qwen2.5:latest")
        messages = [Message(role=MessageRole.USER, content="What's the weather?")]

        chunks = list(provider.stream_chat(messages, tools=MOCK_TOOLS))

        # Should have 1 valid tool call
        tool_chunks = [c for c in chunks if c.tool_call_done]
        assert len(tool_chunks) == 1
        assert (
            tool_chunks[0].tool_call_done
            and tool_chunks[0].tool_call_done.name == "get_weather"
        )
        assert "missing 'function' field" in caplog.text

    def test_stream_missing_arguments(self, MockClient, caplog):
        """Test streaming tool call missing arguments."""
        mock_instance = MockClient.return_value
        mock_instance.list.return_value = MOCK_MODEL_LIST
        mock_instance.show.return_value = create_mock_show_response()

        mock_instance.chat.return_value = iter(
            [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "function": {
                                    "name": "get_weather"
                                    # Missing 'arguments'
                                },
                            }
                        ],
                    },
                    "done": False,
                },
                {
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                },
            ]
        )

        provider = OllamaProvider(model="qwen2.5:latest")
        messages = [Message(role=MessageRole.USER, content="What's the weather?")]

        chunks = list(provider.stream_chat(messages, tools=MOCK_TOOLS))

        tool_chunks = [c for c in chunks if c.tool_call_done]
        assert len(tool_chunks) == 0
        assert "missing 'function.arguments'" in caplog.text

    def test_stream_unexpected_exception(self, MockClient, caplog):
        """Test streaming with unexpected exception during parsing."""
        mock_instance = MockClient.return_value
        mock_instance.list.return_value = MOCK_MODEL_LIST
        mock_instance.show.return_value = create_mock_show_response()

        mock_instance.chat.return_value = iter(
            [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "NYC"}',
                                },
                            }
                        ],
                    },
                    "done": False,
                },
                {
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                },
            ]
        )

        provider = OllamaProvider(model="qwen2.5:latest")

        with patch("allos.providers.ollama.ToolCall", side_effect=RuntimeError("Test")):
            messages = [Message(role=MessageRole.USER, content="What's the weather?")]
            chunks = list(provider.stream_chat(messages, tools=MOCK_TOOLS))

            tool_chunks = [c for c in chunks if c.tool_call_done]
            assert len(tool_chunks) == 0
            assert "Unexpected error parsing stream tool call" in caplog.text


# Helper class to simulate typed objects from newer ollama library
class MockTypedObject:
    """Mock typed object to simulate ollama library's typed responses."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


@patch("allos.providers.ollama.Client")
class TestTypedObjectToolCallsChat:
    """Test typed object tool calls in synchronous chat() - newer ollama library format."""

    def test_typed_object_tool_call_success(self, MockClient):
        """Test successful parsing of typed object tool calls."""
        mock_instance = MockClient.return_value
        mock_instance.list.return_value = MOCK_MODEL_LIST
        mock_instance.show.return_value = create_mock_show_response()

        # Create typed object tool call (simulating newer ollama library)
        mock_function = MockTypedObject(
            name="get_weather", arguments={"location": "NYC"}
        )
        mock_tool_call = MockTypedObject(id="call_123", function=mock_function)

        mock_instance.chat.return_value = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [mock_tool_call],
            },
            "done": True,
        }

        provider = OllamaProvider(model="qwen2.5:latest")
        messages = [Message(role=MessageRole.USER, content="What's the weather?")]

        response = provider.chat(messages, tools=MOCK_TOOLS)

        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "get_weather"
        assert response.tool_calls[0].id == "call_123"

    def test_typed_object_missing_function_attribute(self, MockClient, caplog):
        """Test typed object tool call missing function attribute."""
        mock_instance = MockClient.return_value
        mock_instance.list.return_value = MOCK_MODEL_LIST
        mock_instance.show.return_value = create_mock_show_response()

        # Typed object without function attribute
        mock_tool_call = MockTypedObject(id="call_123")

        mock_instance.chat.return_value = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [mock_tool_call],
            },
            "done": True,
        }

        provider = OllamaProvider(model="qwen2.5:latest")
        messages = [Message(role=MessageRole.USER, content="What's the weather?")]

        response = provider.chat(messages, tools=MOCK_TOOLS)

        assert response.tool_calls == []
        assert "missing 'function' attribute" in caplog.text

    def test_typed_object_function_missing_name(self, MockClient, caplog):
        """Test typed object function missing name attribute."""
        mock_instance = MockClient.return_value
        mock_instance.list.return_value = MOCK_MODEL_LIST
        mock_instance.show.return_value = create_mock_show_response()

        # Function object without name
        mock_function = MockTypedObject(arguments={"location": "NYC"})
        mock_tool_call = MockTypedObject(id="call_123", function=mock_function)

        mock_instance.chat.return_value = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [mock_tool_call],
            },
            "done": True,
        }

        provider = OllamaProvider(model="qwen2.5:latest")
        messages = [Message(role=MessageRole.USER, content="What's the weather?")]

        response = provider.chat(messages, tools=MOCK_TOOLS)

        assert response.tool_calls == []
        assert "missing 'function.name'" in caplog.text

    def test_typed_object_function_missing_arguments(self, MockClient, caplog):
        """Test typed object function missing arguments attribute."""
        mock_instance = MockClient.return_value
        mock_instance.list.return_value = MOCK_MODEL_LIST
        mock_instance.show.return_value = create_mock_show_response()

        # Function object without arguments
        mock_function = MockTypedObject(name="get_weather")
        mock_tool_call = MockTypedObject(id="call_123", function=mock_function)

        mock_instance.chat.return_value = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [mock_tool_call],
            },
            "done": True,
        }

        provider = OllamaProvider(model="qwen2.5:latest")
        messages = [Message(role=MessageRole.USER, content="What's the weather?")]

        response = provider.chat(messages, tools=MOCK_TOOLS)

        assert response.tool_calls == []
        assert "missing 'function.arguments'" in caplog.text

    def test_typed_object_generates_id_when_missing(self, MockClient):
        """Test typed object tool call generates ID when not present."""
        mock_instance = MockClient.return_value
        mock_instance.list.return_value = MOCK_MODEL_LIST
        mock_instance.show.return_value = create_mock_show_response()

        # Typed object without id attribute
        mock_function = MockTypedObject(
            name="get_weather", arguments={"location": "NYC"}
        )
        mock_tool_call = MockTypedObject(function=mock_function)

        mock_instance.chat.return_value = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [mock_tool_call],
            },
            "done": True,
        }

        provider = OllamaProvider(model="qwen2.5:latest")
        messages = [Message(role=MessageRole.USER, content="What's the weather?")]

        response = provider.chat(messages, tools=MOCK_TOOLS)

        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].id.startswith("ollama-tool-")

    def test_dict_tool_call_with_typed_function(self, MockClient):
        """Test dict tool call containing a typed function object."""
        mock_instance = MockClient.return_value
        mock_instance.list.return_value = MOCK_MODEL_LIST
        mock_instance.show.return_value = create_mock_show_response()

        # Dict tool call with typed function object inside
        mock_function = MockTypedObject(
            name="get_weather", arguments={"location": "NYC"}
        )

        mock_instance.chat.return_value = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call_123", "function": mock_function}],
            },
            "done": True,
        }

        provider = OllamaProvider(model="qwen2.5:latest")
        messages = [Message(role=MessageRole.USER, content="What's the weather?")]

        response = provider.chat(messages, tools=MOCK_TOOLS)

        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "get_weather"

    def test_typed_object_tool_call_with_dict_function(self, MockClient):
        """Test typed object tool call containing a dict function."""
        mock_instance = MockClient.return_value
        mock_instance.list.return_value = MOCK_MODEL_LIST
        mock_instance.show.return_value = create_mock_show_response()

        # Typed object tool call with dict function inside
        mock_tool_call = MockTypedObject(
            id="call_123",
            function={"name": "get_weather", "arguments": {"location": "NYC"}},
        )

        mock_instance.chat.return_value = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [mock_tool_call],
            },
            "done": True,
        }

        provider = OllamaProvider(model="qwen2.5:latest")
        messages = [Message(role=MessageRole.USER, content="What's the weather?")]

        response = provider.chat(messages, tools=MOCK_TOOLS)

        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "get_weather"
        assert response.tool_calls[0].id == "call_123"


@patch("allos.providers.ollama.Client")
class TestTypedObjectToolCallsStreaming:
    """Test typed object tool calls in streaming mode - newer ollama library format."""

    def test_stream_typed_object_tool_call_success(self, MockClient):
        """Test successful parsing of typed object tool calls in streaming."""
        mock_instance = MockClient.return_value
        mock_instance.list.return_value = MOCK_MODEL_LIST
        mock_instance.show.return_value = create_mock_show_response()

        # Create typed object tool call
        mock_function = MockTypedObject(
            name="get_weather", arguments={"location": "NYC"}
        )
        mock_tool_call = MockTypedObject(id="call_123", function=mock_function)

        mock_instance.chat.return_value = iter(
            [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [mock_tool_call],
                    },
                    "done": False,
                },
                {
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                },
            ]
        )

        provider = OllamaProvider(model="qwen2.5:latest")
        messages = [Message(role=MessageRole.USER, content="What's the weather?")]

        chunks = list(provider.stream_chat(messages, tools=MOCK_TOOLS))

        tool_chunks = [c for c in chunks if c.tool_call_done]
        assert len(tool_chunks) == 1 and tool_chunks[0].tool_call_done
        assert tool_chunks[0].tool_call_done.name == "get_weather"

    def test_stream_typed_object_missing_function(self, MockClient, caplog):
        """Test streaming typed object tool call missing function attribute."""
        mock_instance = MockClient.return_value
        mock_instance.list.return_value = MOCK_MODEL_LIST
        mock_instance.show.return_value = create_mock_show_response()

        # Typed object without function
        mock_tool_call = MockTypedObject(id="call_123")

        mock_instance.chat.return_value = iter(
            [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [mock_tool_call],
                    },
                    "done": False,
                },
                {
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                },
            ]
        )

        provider = OllamaProvider(model="qwen2.5:latest")
        messages = [Message(role=MessageRole.USER, content="What's the weather?")]

        chunks = list(provider.stream_chat(messages, tools=MOCK_TOOLS))

        tool_chunks = [c for c in chunks if c.tool_call_done]
        assert len(tool_chunks) == 0
        assert "missing 'function' attribute" in caplog.text

    def test_stream_typed_object_function_missing_name(self, MockClient, caplog):
        """Test streaming typed object function missing name."""
        mock_instance = MockClient.return_value
        mock_instance.list.return_value = MOCK_MODEL_LIST
        mock_instance.show.return_value = create_mock_show_response()

        # Function without name
        mock_function = MockTypedObject(arguments={"location": "NYC"})
        mock_tool_call = MockTypedObject(id="call_123", function=mock_function)

        mock_instance.chat.return_value = iter(
            [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [mock_tool_call],
                    },
                    "done": False,
                },
                {
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                },
            ]
        )

        provider = OllamaProvider(model="qwen2.5:latest")
        messages = [Message(role=MessageRole.USER, content="What's the weather?")]

        chunks = list(provider.stream_chat(messages, tools=MOCK_TOOLS))

        tool_chunks = [c for c in chunks if c.tool_call_done]
        assert len(tool_chunks) == 0
        assert "missing 'function.name'" in caplog.text

    def test_stream_typed_object_function_missing_arguments(self, MockClient, caplog):
        """Test streaming typed object function missing arguments."""
        mock_instance = MockClient.return_value
        mock_instance.list.return_value = MOCK_MODEL_LIST
        mock_instance.show.return_value = create_mock_show_response()

        # Function without arguments
        mock_function = MockTypedObject(name="get_weather")
        mock_tool_call = MockTypedObject(id="call_123", function=mock_function)

        mock_instance.chat.return_value = iter(
            [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [mock_tool_call],
                    },
                    "done": False,
                },
                {
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                },
            ]
        )

        provider = OllamaProvider(model="qwen2.5:latest")
        messages = [Message(role=MessageRole.USER, content="What's the weather?")]

        chunks = list(provider.stream_chat(messages, tools=MOCK_TOOLS))

        tool_chunks = [c for c in chunks if c.tool_call_done]
        assert len(tool_chunks) == 0
        assert "missing 'function.arguments'" in caplog.text

    def test_stream_dict_tool_call_with_typed_function(self, MockClient):
        """Test streaming dict tool call containing a typed function object."""
        mock_instance = MockClient.return_value
        mock_instance.list.return_value = MOCK_MODEL_LIST
        mock_instance.show.return_value = create_mock_show_response()

        # Dict tool call with typed function inside
        mock_function = MockTypedObject(
            name="get_weather", arguments={"location": "NYC"}
        )

        mock_instance.chat.return_value = iter(
            [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"id": "call_123", "function": mock_function}],
                    },
                    "done": False,
                },
                {
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                },
            ]
        )

        provider = OllamaProvider(model="qwen2.5:latest")
        messages = [Message(role=MessageRole.USER, content="What's the weather?")]

        chunks = list(provider.stream_chat(messages, tools=MOCK_TOOLS))

        tool_chunks = [c for c in chunks if c.tool_call_done]
        assert len(tool_chunks) == 1 and tool_chunks[0].tool_call_done
        assert tool_chunks[0].tool_call_done.name == "get_weather"

    def test_stream_typed_object_generates_id_when_missing(self, MockClient):
        """Test streaming typed object generates ID when not present."""
        mock_instance = MockClient.return_value
        mock_instance.list.return_value = MOCK_MODEL_LIST
        mock_instance.show.return_value = create_mock_show_response()

        # Typed object without id
        mock_function = MockTypedObject(
            name="get_weather", arguments={"location": "NYC"}
        )
        mock_tool_call = MockTypedObject(function=mock_function)

        mock_instance.chat.return_value = iter(
            [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [mock_tool_call],
                    },
                    "done": False,
                },
                {
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                },
            ]
        )

        provider = OllamaProvider(model="qwen2.5:latest")
        messages = [Message(role=MessageRole.USER, content="What's the weather?")]

        chunks = list(provider.stream_chat(messages, tools=MOCK_TOOLS))

        tool_chunks = [c for c in chunks if c.tool_call_done]
        assert len(tool_chunks) == 1 and tool_chunks[0].tool_call_done
        assert tool_chunks[0].tool_call_done.id.startswith("ollama-tool-")

    def test_stream_typed_object_tool_call_with_dict_function(self, MockClient):
        """Test streaming typed object tool call containing a dict function."""
        mock_instance = MockClient.return_value
        mock_instance.list.return_value = MOCK_MODEL_LIST
        mock_instance.show.return_value = create_mock_show_response()

        # Typed object tool call with dict function inside
        mock_tool_call = MockTypedObject(
            id="call_123",
            function={"name": "get_weather", "arguments": {"location": "NYC"}},
        )

        mock_instance.chat.return_value = iter(
            [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [mock_tool_call],
                    },
                    "done": False,
                },
                {
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                },
            ]
        )

        provider = OllamaProvider(model="qwen2.5:latest")
        messages = [Message(role=MessageRole.USER, content="What's the weather?")]

        chunks = list(provider.stream_chat(messages, tools=MOCK_TOOLS))

        tool_chunks = [c for c in chunks if c.tool_call_done]
        assert len(tool_chunks) == 1 and tool_chunks[0].tool_call_done
        assert tool_chunks[0].tool_call_done.name == "get_weather"
        assert tool_chunks[0].tool_call_done.id == "call_123"
