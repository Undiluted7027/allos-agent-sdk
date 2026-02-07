# tests/unit/providers/test_google_provider.py

"""Unit tests for Google/Gemini provider."""

import sys
from unittest.mock import MagicMock, patch

import pytest

# Skip entire module if Python < 3.10
pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 10), reason="Google provider requires Python 3.10+"
)

# Conditional imports - only import if Python >= 3.10
if sys.version_info >= (3, 10):
    from google.genai import errors as genai_errors
    from google.genai import types

    from allos.providers.google import GoogleProvider

from allos.providers.base import Message, MessageRole, ToolCall  # noqa: E402
from allos.tools.base import BaseTool, ToolParameter  # noqa: E402
from allos.utils.errors import ProviderError  # noqa: E402


class MockTool(BaseTool):
    name: str = "search"
    description: str = "Search tool"
    parameters: list[ToolParameter] = [
        ToolParameter(
            name="query",
            type="string",
            required=True,
            description="Query",
        )
    ]

    def execute(self, **kwargs):
        return {}


MOCK_TOOLS = [MockTool()]


class TestGoogleProviderInit:
    """Tests for GoogleProvider initialization"""

    @patch("allos.providers.google.genai")
    def test_init_with_api_key(self, mock_genai):
        """Test initialization with Gemini API key."""

        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.list.return_value = [
            types.Model(name="models/gemini-2.0-flash", input_token_limit=1048576)
        ]

        provider = GoogleProvider(model="gemini-2.0-flash", api_key="test-key")

        assert provider.model == "gemini-2.0-flash"
        assert not provider.vertexai  # pyright: ignore[reportAttributeAccessIssue]
        mock_genai.Client.assert_called_once_with(api_key="test-key")

    @patch("allos.providers.google.genai")
    def test_init_with_vertexai(self, mock_genai):
        """Test initialization with Vertex AI."""

        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.list.return_value = [
            types.Model(name="gemini-2.0-flash", input_token_limit=1048576)
        ]

        provider = GoogleProvider(
            model="gemini-2.0-flash",
            vertexai=True,
            project="test-project",
            location="us-central1",
        )

        assert provider.vertexai is True  # pyright: ignore[reportAttributeAccessIssue]
        assert provider.project == "test-project"  # pyright: ignore[reportAttributeAccessIssue]
        mock_genai.Client.assert_called_once_with(
            vertexai=True, project="test-project", location="us-central1"
        )

    @patch("allos.providers.google.genai")
    def test_init_model_not_available(self, mock_genai):
        """Test error when model is not available."""
        from allos.providers.google import GoogleProvider
        from allos.utils.errors import ProviderError

        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.list.return_value = []  # No models available

        with pytest.raises(ProviderError) as exc_info:
            GoogleProvider(model="nonexistent-model", api_key="test-key")

        assert "not available" in str(exc_info.value)

    @patch("allos.providers.google.genai")
    def test_init_client_error(self, mock_genai):
        """Test handling of client initialization errors."""

        mock_genai.Client.side_effect = genai_errors.ClientError(
            403, {"name": "Unauthorized"}
        )

        with pytest.raises(ProviderError) as exc_info:
            GoogleProvider(model="gemini-2.0-flash", api_key="bad-key")

        assert "Authentication" in str(exc_info.value)

    @patch("allos.providers.google.genai")
    def test_init_server_error(self, mock_genai):
        """Test handling of server errors during init."""

        mock_genai.Client.side_effect = genai_errors.ServerError(
            500, response_json={"name": "Server Down"}
        )

        with pytest.raises(ProviderError) as exc_info:
            GoogleProvider(model="gemini-2.0-flash", api_key="test-key")

        assert "server error" in str(exc_info.value).lower()


class TestGoogleProviderCheckEnvConfig:
    """Tests for check_env_config classmethod."""

    def test_check_env_config_with_google_api_key(self, monkeypatch):
        """Test check_env_config with GOOGLE_API_KEY set."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)

        from allos.providers.google import GoogleProvider

        is_configured, message = GoogleProvider.check_env_config()

        assert is_configured is True
        assert "GOOGLE_API_KEY (Set)" in message

    def test_check_env_config_with_gemini_api_key(self, monkeypatch):
        """Test check_env_config with GEMINI_API_KEY set."""
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)

        from allos.providers.google import GoogleProvider

        is_configured, message = GoogleProvider.check_env_config()

        assert is_configured is True
        assert "GEMINI_API_KEY (Set)" in message

    def test_check_env_config_with_vertex_ai(self, monkeypatch):
        """Test check_env_config with Vertex AI configuration."""
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
        monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-west1")

        from allos.providers.google import GoogleProvider

        is_configured, message = GoogleProvider.check_env_config()

        assert is_configured is True
        assert "Vertex AI" in message
        assert "PROJECT=Set" in message
        assert "LOCATION=Set" in message

    def test_check_env_config_vertex_ai_default_location(self, monkeypatch):
        """Test check_env_config with Vertex AI using default location."""
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
        monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)

        from allos.providers.google import GoogleProvider

        is_configured, message = GoogleProvider.check_env_config()

        assert is_configured is True
        assert "us-central1" in message

    def test_check_env_config_not_configured(self, monkeypatch):
        """Test check_env_config when nothing is configured."""
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)

        from allos.providers.google import GoogleProvider

        is_configured, message = GoogleProvider.check_env_config()

        assert is_configured is False
        assert "Not Set" in message


class TestGoogleProviderChat:
    """Tests for chat method."""

    @patch("allos.providers.google.genai")
    def test_chat_simple_response(self, mock_genai):
        """Test basic chat functionality."""
        from allos.providers.base import Message, MessageRole
        from allos.providers.google import GoogleProvider

        # Setup mocks
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.list.return_value = [
            types.Model(name="models/gemini-2.0-flash", input_token_limit=1048576)
        ]

        mock_response = MagicMock()
        mock_response.text = "Hello, world!"
        mock_response.function_calls = None
        mock_response.usage_metadata = MagicMock(
            prompt_token_count=10, candidates_token_count=5
        )
        mock_client.models.generate_content.return_value = mock_response

        provider = GoogleProvider(model="gemini-2.0-flash", api_key="test-key")

        messages = [Message(role=MessageRole.USER, content="Hello")]
        response = provider.chat(messages)

        assert response.content == "Hello, world!"
        assert response.tool_calls == []

    @patch("allos.providers.google.genai")
    def test_chat_with_system_message(self, mock_genai):
        """Test chat with system message."""
        from allos.providers.base import Message, MessageRole
        from allos.providers.google import GoogleProvider

        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.list.return_value = [
            types.Model(name="models/gemini-2.0-flash", input_token_limit=1048576)
        ]

        mock_response = MagicMock()
        mock_response.text = "Response"
        mock_response.function_calls = None
        mock_response.usage_metadata = None
        mock_client.models.generate_content.return_value = mock_response

        provider = GoogleProvider(model="gemini-2.0-flash", api_key="test-key")

        messages = [
            Message(role=MessageRole.SYSTEM, content="You are helpful"),
            Message(role=MessageRole.USER, content="Hello"),
        ]
        provider.chat(messages)

        # Verify system instruction was passed in config
        call_kwargs = mock_client.models.generate_content.call_args
        config = call_kwargs.kwargs.get("config")
        assert config.system_instruction == "You are helpful"

    @patch("allos.providers.google.genai")
    def test_chat_with_tool_call(self, mock_genai):
        """Test chat that returns a tool call."""
        from allos.providers.base import Message, MessageRole
        from allos.providers.google import GoogleProvider
        from allos.tools.base import BaseTool, ToolParameter

        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.list.return_value = [
            types.Model(name="models/gemini-2.0-flash", input_token_limit=1048576)
        ]

        # Mock function call response
        mock_fc = MagicMock()
        mock_fc.name = "test_tool"
        mock_fc.args = {"arg1": "value1"}

        mock_response = MagicMock()
        mock_response.text = None
        mock_response.function_calls = [mock_fc]
        mock_response.usage_metadata = None
        mock_client.models.generate_content.return_value = mock_response

        provider = GoogleProvider(model="gemini-2.0-flash", api_key="test-key")

        # Create a mock tool
        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "test_tool"
        mock_tool.description = "A test tool"
        mock_tool.parameters = [
            ToolParameter(
                name="arg1", type="string", description="An argument", required=True
            )
        ]

        messages = [Message(role=MessageRole.USER, content="Use the tool")]
        response = provider.chat(messages, tools=[mock_tool])

        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "test_tool"
        assert response.tool_calls[0].arguments == {"arg1": "value1"}

    @patch("allos.providers.google.genai")
    def test_chat_api_error(self, mock_genai):
        """Test handling of API errors during chat."""
        from google.genai import errors as genai_errors

        from allos.providers.base import Message, MessageRole
        from allos.providers.google import GoogleProvider
        from allos.utils.errors import ProviderError

        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.list.return_value = [
            types.Model(name="models/gemini-2.0-flash", input_token_limit=1048576)
        ]

        mock_client.models.generate_content.side_effect = genai_errors.ClientError(
            429, {"name": "Rate limit exceeded"}
        )

        provider = GoogleProvider(model="gemini-2.0-flash", api_key="test-key")

        with pytest.raises(ProviderError) as exc_info:
            provider.chat([Message(role=MessageRole.USER, content="Hello")])

        assert "client error" in str(exc_info.value).lower()


class TestGoogleProviderStreamChat:
    """Tests for stream_chat method."""

    @patch("allos.providers.google.genai")
    def test_stream_chat_text_chunks(self, mock_genai):
        """Test streaming text response."""
        from allos.providers.base import Message, MessageRole
        from allos.providers.google import GoogleProvider

        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.list.return_value = [
            types.Model(name="models/gemini-2.0-flash", input_token_limit=1048576)
        ]

        # Mock streaming chunks
        chunk1 = MagicMock()
        chunk1.text = "Hello"
        chunk1.function_calls = None

        chunk2 = MagicMock()
        chunk2.text = " World"
        chunk2.function_calls = None

        mock_client.models.generate_content_stream.return_value = iter([chunk1, chunk2])

        provider = GoogleProvider(model="gemini-2.0-flash", api_key="test-key")

        messages = [Message(role=MessageRole.USER, content="Hello")]
        chunks = list(provider.stream_chat(messages))

        # Should have text chunks plus final metadata chunk
        text_chunks = [c for c in chunks if c.content]
        assert len(text_chunks) >= 2
        assert text_chunks[0].content == "Hello"
        assert text_chunks[1].content == " World"

    @patch("allos.providers.google.genai")
    def test_stream_chat_with_function_call(self, mock_genai):
        """Test streaming response with function call."""
        from allos.providers.base import Message, MessageRole
        from allos.providers.google import GoogleProvider

        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.list.return_value = [
            types.Model(name="models/gemini-2.0-flash", input_token_limit=1048576)
        ]

        # Mock function call in stream
        mock_fc = MagicMock()
        mock_fc.name = "search"
        mock_fc.args = {"query": "test"}

        chunk = MagicMock()
        chunk.text = None
        chunk.function_calls = [mock_fc]

        mock_client.models.generate_content_stream.return_value = iter([chunk])

        provider = GoogleProvider(model="gemini-2.0-flash", api_key="test-key")

        messages = [Message(role=MessageRole.USER, content="Search for test")]
        chunks = list(provider.stream_chat(messages))

        tool_chunks = [c for c in chunks if c.tool_call_done]
        assert len(tool_chunks) >= 1
        assert (
            tool_chunks[0].tool_call_done
            and tool_chunks[0].tool_call_done.name == "search"
        )


class TestGoogleProviderMessageConversion:
    """Tests for message conversion methods."""

    def test_convert_messages_user_only(self):
        """Test converting user messages."""
        from allos.providers.base import Message, MessageRole
        from allos.providers.google import GoogleProvider

        messages = [Message(role=MessageRole.USER, content="Hello")]
        system, contents = GoogleProvider._convert_messages(messages)  # pyright: ignore[reportAttributeAccessIssue]

        assert system is None
        assert len(contents) == 1
        assert contents[0].role == "user"

    def test_convert_messages_with_tool_result(self):
        """Test converting tool result messages."""
        from allos.providers.base import Message, MessageRole
        from allos.providers.google import GoogleProvider

        messages = [
            Message(role=MessageRole.TOOL, content="Tool result", tool_call_id="search")
        ]
        system, contents = GoogleProvider._convert_messages(messages)  # pyright: ignore[reportAttributeAccessIssue]

        assert len(contents) == 1
        assert contents[0].role == "user"
        # Should contain function_response part

    def test_convert_messages_assistant_with_tool_calls(self):
        """Test message conversion for assistant messages containing tool calls."""

        messages = [
            Message(role=MessageRole.USER, content="What's the weather?"),
            Message(
                role=MessageRole.ASSISTANT,
                content="Let me check that for you.",
                tool_calls=[
                    ToolCall(
                        name="get_weather",
                        arguments={"location": "San Francisco"},
                        id="call_123",
                    )
                ],
            ),
            Message(
                role=MessageRole.TOOL,
                content='{"temp": 72, "condition": "sunny"}',
                tool_call_id="call_123",
            ),
        ]

        system_instruction, contents = GoogleProvider._convert_messages(messages)  # pyright: ignore[reportAttributeAccessIssue]

        # Verify assistant message with tool call was converted correctly
        assert len(contents) == 3

        # Check the assistant message (index 1)
        assistant_content = contents[1]
        assert assistant_content.role == "model"
        assert len(assistant_content.parts) == 2  # text + function_call

        # Verify function call part exists
        function_parts = [
            p
            for p in assistant_content.parts
            if getattr(p, "function_call", None) is not None
        ]
        assert len(function_parts) == 1

    @patch("allos.providers.google.genai")
    def test_stream_chat_with_system_instruction_and_tools(self, mock_genai):
        """Test stream_chat with both system instruction and tools."""

        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.list.return_value = [
            types.Model(name="models/gemini-2.0-flash", input_token_limit=1048576)
        ]

        # Mock streaming response
        mock_chunk = MagicMock()
        mock_chunk.text = "Using tool..."
        mock_chunk.function_calls = None
        mock_chunk.usage_metadata = None
        mock_client.models.generate_content_stream.return_value = iter([mock_chunk])

        provider = GoogleProvider(model="gemini-2.0-flash", api_key="test-key")

        messages = [
            Message(role=MessageRole.SYSTEM, content="You are helpful"),
            Message(role=MessageRole.USER, content="Search for something"),
        ]

        chunks = list(provider.stream_chat(messages, tools=MOCK_TOOLS))

        # Verify generate_content_stream was called with config including tools
        call_kwargs = mock_client.models.generate_content_stream.call_args
        config = call_kwargs.kwargs["config"]

        assert "system_instruction" in config or hasattr(config, "system_instruction")
        assert "tools" in config or hasattr(config, "tools")
        assert len(chunks) > 0


class TestGoogleProviderContextWindow:
    """Tests for get_context_window method."""

    @patch("allos.providers.google.genai")
    def test_get_context_window_from_model_info(self, mock_genai):
        """Test getting context window from model info."""
        from allos.providers.google import GoogleProvider

        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client

        mock_model = MagicMock()
        mock_model.name = "models/gemini-2.0-flash"
        mock_model.input_token_limit = 1048576
        mock_client.models.list.return_value = [mock_model]

        provider = GoogleProvider(model="gemini-2.0-flash", api_key="test-key")

        assert provider.get_context_window() == 1048576

    @patch("allos.providers.google.genai")
    def test_get_context_window_fallback_to_static(self, mock_genai):
        """Test fallback to static context window mapping."""
        from allos.providers.google import GoogleProvider

        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client

        mock_model = MagicMock()
        mock_model.name = "models/gemini-2.0-flash"
        mock_model.input_token_limit = None  # No dynamic info
        mock_client.models.list.return_value = [mock_model]

        provider = GoogleProvider(model="gemini-2.0-flash", api_key="test-key")
        provider._model_context_window = None  # pyright: ignore[reportAttributeAccessIssue] # Clear dynamic value

        # Should fall back to MODEL_CONTEXT_WINDOWS lookup
        context = provider.get_context_window()
        assert context == 1048576  # Either from map or default

    @patch("allos.providers.google.genai")
    def test_get_context_window_unknown_model_fallback(self, mock_genai):
        """Test context window fallback for unknown/custom models."""
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client

        # Model info without input_token_liit
        mock_client.models.list.return_value = [
            types.Model(name="models/custom-model-v1")  # No input token limit
        ]

        provider = GoogleProvider(model="custom-model-v1", api_key="test-key")

        # Should fall back to default
        context_window = provider.get_context_window()

        assert context_window == 4096  # Default fallback


class TestGoogleProviderErrorHandling:
    """Comprehensive error handling tests."""

    @patch("allos.providers.google.genai")
    def test_init_api_error_catch_all(self, mock_genai):
        """Test handling of generic APIError during init."""
        mock_genai.Client.side_effect = genai_errors.APIError(520, "Generic API error")

        with pytest.raises(ProviderError) as exc_info:
            GoogleProvider(model="gemini-2.0-flash", api_key="test-key")

        assert "Google API error" in str(exc_info.value)

    @patch("allos.providers.google.genai")
    def test_init_model_verification_api_error(self, mock_genai):
        """Test APIError during model verification."""
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client

        # models.list() raises APIError
        mock_client.models.list.side_effect = genai_errors.APIError(
            410, "Could not list models"
        )

        with pytest.raises(ProviderError) as exc_info:
            GoogleProvider(model="gemini-2.0-flash", api_key="test-key")

        assert "Could not verify model" in str(exc_info.value)

    @patch("allos.providers.google.genai")
    def test_chat_server_error(self, mock_genai):
        """Test ServerError handling during chat."""
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.list.return_value = [
            types.Model(name="models/gemini-2.0-flash", input_token_limit=1048576)
        ]

        # generate_content raises ServerError
        mock_client.models.generate_content.side_effect = genai_errors.ServerError(
            500, response_json={"error": "Server overloaded"}
        )

        provider = GoogleProvider(model="gemini-2.0-flash", api_key="test-key")

        with pytest.raises(ProviderError) as exc_info:
            provider.chat([Message(role=MessageRole.USER, content="Hello")])

        assert "server error" in str(exc_info.value).lower()

    @patch("allos.providers.google.genai")
    def test_chat_generic_api_error(self, mock_genai):
        """Test generic APIError handling during chat."""
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.list.return_value = [
            types.Model(name="models/gemini-2.0-flash", input_token_limit=1048576)
        ]

        mock_client.models.generate_content.side_effect = genai_errors.APIError(
            500, "Unknown API error"
        )

        provider = GoogleProvider(model="gemini-2.0-flash", api_key="test-key")

        with pytest.raises(ProviderError) as exc_info:
            provider.chat([Message(role=MessageRole.USER, content="Hello")])

        assert "Google API error" in str(exc_info.value)

    @patch("allos.providers.google.genai")
    def test_stream_chat_api_error(self, mock_genai):
        """Test APIError handling during stream_chat."""
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.list.return_value = [
            types.Model(name="models/gemini-2.0-flash", input_token_limit=1048576)
        ]

        # generate_content_stream raises APIError
        mock_client.models.generate_content_stream.side_effect = genai_errors.APIError(
            500, "Streaming failed"
        )

        provider = GoogleProvider(model="gemini-2.0-flash", api_key="test-key")

        with pytest.raises(ProviderError) as exc_info:
            list(
                provider.stream_chat([Message(role=MessageRole.USER, content="Hello")])
            )

        assert "streaming error" in str(exc_info.value).lower()
