# tests/unit/providers/test_google_provider.py

"""Tests for Google provider.

NOTE: These tests require Python 3.10+ due to google-auth dependency.
On Python 3.9, these tests are automatically skipped.
"""

import json
import os
import sys
from unittest.mock import MagicMock, mock_open, patch

import pytest

# Skip entire module on Python 3.9 or lower
if sys.version_info < (3, 10):
    pytest.skip(
        "Google provider requires Python 3.10+ (google-auth>=2.48.0 dependency). "
        "All Google provider tests skipped on Python 3.9.",
        allow_module_level=True,
    )

from google.auth.exceptions import DefaultCredentialsError
from google.genai import errors as genai_errors
from google.genai import types

from allos.providers.base import Message, MessageRole, ToolCall
from allos.providers.google import GoogleProvider
from allos.tools.base import BaseTool, ToolParameter
from allos.utils.errors import ProviderError


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


@pytest.fixture
def mock_service_account_json():
    """Valid service account JSON structure."""
    return {
        "type": "service_account",
        "project_id": "allos-test-project-from-json",
        "private_key_id": "key123",
        "private_key": "-----BEGIN KEY-----\nMOCK_KEY\n-----END PRIVATE KEY-----",
        "client_email": "allos@allos-test-project-from-json.iam.gserviceaccount.com",
        "client_id": "123456789",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/allos%40allos-test-project-from-json.iam.gserviceaccount.com",
        "universe_domain": "googleapis.com",
    }


@pytest.fixture
def mock_credentials():
    """Mock google.auth.credentials.Credentials object."""
    mock_creds = MagicMock()
    mock_creds.project_id = None
    return mock_creds


@pytest.fixture
def mock_genai_client():
    """Mock genai.Client with models list."""
    mock_client = MagicMock()
    mock_client.models.list.return_value = [
        types.Model(
            name="publishers/google/models/gemini-2.0-flash",
            input_token_limit=1048576,
        )
    ]
    return mock_client


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
    @patch("allos.providers.google.os.path.exists")
    @patch("google.oauth2.service_account.Credentials.from_service_account_file")
    def test_init_with_vertexai(
        self, mock_from_sa_file, mock_exists, mock_genai, monkeypatch
    ):
        """Test initialization with Vertex AI."""
        monkeypatch.setenv(
            "GOOGLE_APPLICATION_CREDENTIALS", "/path/to/test_credentials.json"
        )

        # Mock file system and credentials
        mock_exists.return_value = True
        mock_creds = MagicMock()
        mock_from_sa_file.return_value = mock_creds

        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.list.return_value = [
            types.Model(
                name="publishers/google/models/gemini-2.0-flash",
                input_token_limit=1048576,
            )
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
            vertexai=True,
            project="test-project",
            location="us-central1",
            credentials=mock_creds,
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

        error_msg = str(exc_info.value)
        assert "not available" in error_msg

    @patch("allos.providers.google.genai")
    def test_init_model_not_available_with_suggestions(self, mock_genai):
        """Test error message includes similar model suggestions."""
        from allos.providers.google import GoogleProvider
        from allos.utils.errors import ProviderError

        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client

        # Mock available models with similar names
        mock_models = [
            types.Model(name="models/gemini-2.0-flash", input_token_limit=1048576),
            types.Model(name="models/gemini-2.5-flash", input_token_limit=1048576),
            types.Model(name="models/gemini-1.5-flash", input_token_limit=1048576),
            types.Model(name="models/gemini-2.0-pro", input_token_limit=2097152),
        ]
        mock_client.models.list.return_value = mock_models

        with pytest.raises(ProviderError) as exc_info:
            # Request a typo version
            GoogleProvider(model="gemini-2.0-flsh", api_key="test-key")

        error_msg = str(exc_info.value)
        assert "not available" in error_msg
        assert "Did you mean" in error_msg
        # Should suggest the close match
        assert "gemini-2.0-flash" in error_msg

    @patch("allos.providers.google.genai")
    def test_init_model_not_available_with_prefix_suggestions(self, mock_genai):
        """Test error message suggests models with similar prefix when no close match."""
        from allos.providers.google import GoogleProvider
        from allos.utils.errors import ProviderError

        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client

        # Mock available models
        mock_models = [
            types.Model(name="models/gemini-2.0-flash", input_token_limit=1048576),
            types.Model(name="models/gemini-2.5-flash", input_token_limit=1048576),
            types.Model(name="models/claude-3-opus", input_token_limit=200000),
        ]
        mock_client.models.list.return_value = mock_models

        with pytest.raises(ProviderError) as exc_info:
            # Request a completely different model in gemini family
            GoogleProvider(model="gemini-xyz-123", api_key="test-key")

        error_msg = str(exc_info.value)
        assert "not available" in error_msg
        # Should suggest gemini models since prefix matches
        assert "gemini" in error_msg.lower()

    @patch("allos.providers.google.genai")
    def test_init_client_error(self, mock_genai):
        """Test handling of client initialization errors."""

        mock_genai.Client.side_effect = genai_errors.ClientError(
            403, {"message": "Unauthorized"}
        )

        with pytest.raises(ProviderError) as exc_info:
            GoogleProvider(model="gemini-2.0-flash", api_key="bad-key")

        assert "Authentication" in str(exc_info.value)

    @patch("allos.providers.google.genai")
    def test_init_server_error(self, mock_genai):
        """Test handling of server errors during init."""

        mock_genai.Client.side_effect = genai_errors.ServerError(
            500, response_json={"message": "Server Down"}
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

    @patch("allos.providers.google.os.path.exists")
    def test_check_env_with_sa_file_exists(self, mock_exists, monkeypatch):
        """Test environment check when service account file exists."""
        mock_exists.return_value = True
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/path/to/sa.json")
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
        monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)

        is_configured, message = GoogleProvider.check_env_config()

        assert is_configured is True
        assert "Service Account" in message
        assert "sa.json" in message

    @patch("allos.providers.google.os.path.exists")
    def test_check_env_with_sa_file_not_exists(self, mock_exists, monkeypatch):
        """Test environment check when service account file doesn't exist."""
        mock_exists.return_value = False
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/path/to/missing.json")
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
        monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)

        configured, message = GoogleProvider.check_env_config()

        assert configured is False
        assert "non-existent file" in message
        assert "/path/to/missing.json" in message

    @patch("google.auth.default")
    def test_check_env_with_adc_auto_detect(self, mock_auth_default, monkeypatch):
        """Test environment check with ADC auto-detection."""
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
        monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
        mock_creds = MagicMock()
        mock_auth_default.return_value = (mock_creds, "auto-detected-project")

        configured, message = GoogleProvider.check_env_config()

        assert configured is True
        assert "ADC" in message
        assert "auto-detected-project" in message

    @patch("google.auth.default")
    def test_check_env_with_adc_default_credentials_error(
        self, mock_auth_default, monkeypatch
    ):
        """Test environment check when ADC fails."""
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
        monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
        mock_auth_default.side_effect = DefaultCredentialsError("No credentials")

        configured, message = GoogleProvider.check_env_config()

        assert configured is False
        assert "No authentication method configured" in message

    def test_check_env_with_no_auth(self):
        """Test environment check with no authentication configured."""
        # Clear all environment variables
        env_vars = [
            "GOOGLE_API_KEY",
            "GEMINI_API_KEY",
            "GOOGLE_APPLICATION_CREDENTIALS",
            "GOOGLE_CLOUD_PROJECT",
        ]

        with patch.dict(os.environ, dict.fromkeys(env_vars, ""), clear=True):
            with patch("google.auth.default") as mock_auth:
                mock_auth.side_effect = DefaultCredentialsError("No credentials")

                configured, message = GoogleProvider.check_env_config()

                assert configured is False
                assert "No authentication method configured" in message

    def test_check_env_with_vertex_project(self, monkeypatch):
        """Test environment check with GOOGLE_CLOUD_PROJECT set."""
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-vertex-project")
        monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-west1")

        configured, message = GoogleProvider.check_env_config()

        assert configured is True
        assert "Vertex AI ADC" in message
        assert "test-vertex-project" in message
        assert "us-west1" in message


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

        # Create properly structured response with parts
        mock_part = MagicMock()
        mock_part.text = "Hello, world!"
        mock_part.function_call = None

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
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

        # Mock the part containing the function call
        mock_part = MagicMock()
        mock_part.text = None  # No text content, only function call
        mock_part.function_call = mock_fc
        mock_part.thought_signature = None

        # Mock the content containing the part
        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        # Mock the candidate containing the content
        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_response = MagicMock()
        mock_response.text = None
        mock_response.candidates = [mock_candidate]
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
            429, {"message": "Rate limit exceeded"}
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

        # Mock streaming chunks with proper structure
        # Chunk 1
        part1 = MagicMock()
        part1.text = "Hello"
        part1.function_call = None

        content1 = MagicMock()
        content1.parts = [part1]

        candidate1 = MagicMock()
        candidate1.content = content1

        chunk1 = MagicMock()
        chunk1.candidates = [candidate1]
        chunk1.function_calls = None

        # Chunk 2
        part2 = MagicMock()
        part2.text = " World"
        part2.function_call = None

        content2 = MagicMock()
        content2.parts = [part2]

        candidate2 = MagicMock()
        candidate2.content = content2

        chunk2 = MagicMock()
        chunk2.candidates = [candidate2]
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

        # Mock the part containing the function call
        mock_part = MagicMock()
        mock_part.text = None  # No text content, only function call
        mock_part.function_call = mock_fc
        mock_part.thought_signature = None

        # Mock the content containing the part
        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        # Mock the candidate containing the content
        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        chunk = MagicMock()
        chunk.text = None
        chunk.function_calls = [mock_fc]  # Still set this for backward compatibility
        chunk.candidates = [mock_candidate]

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

    @patch("allos.providers.google.genai")
    def test_convert_messages_user_only(self, mock_genai):
        """Test converting user messages."""
        from allos.providers.base import Message, MessageRole
        from allos.providers.google import GoogleProvider

        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.list.return_value = [
            types.Model(name="models/gemini-2.0-flash", input_token_limit=1048576)
        ]

        provider = GoogleProvider(model="gemini-2.0-flash", api_key="test-key")
        messages = [Message(role=MessageRole.USER, content="Hello")]
        system, contents = provider._convert_messages(messages)  # pyright: ignore[reportAttributeAccessIssue]

        assert system is None
        assert len(contents) == 1
        assert contents[0].role == "user"

    @patch("allos.providers.google.genai")
    def test_convert_messages_with_tool_result(self, mock_genai):
        """Test converting tool result messages."""
        from allos.providers.base import Message, MessageRole
        from allos.providers.google import GoogleProvider

        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.list.return_value = [
            types.Model(name="models/gemini-2.0-flash", input_token_limit=1048576)
        ]

        provider = GoogleProvider(model="gemini-2.0-flash", api_key="test-key")
        messages = [
            Message(role=MessageRole.TOOL, content="Tool result", tool_call_id="search")
        ]
        system, contents = provider._convert_messages(messages)  # pyright: ignore[reportAttributeAccessIssue]

        assert len(contents) == 1
        assert contents[0].role == "user"
        # Should contain function_response part

    @patch("allos.providers.google.genai")
    def test_convert_messages_assistant_with_tool_calls(self, mock_genai):
        """Test message conversion for assistant messages containing tool calls."""
        from allos.providers.google import GoogleProvider

        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.list.return_value = [
            types.Model(name="models/gemini-2.0-flash", input_token_limit=1048576)
        ]

        provider = GoogleProvider(model="gemini-2.0-flash", api_key="test-key")

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

        system_instruction, contents = provider._convert_messages(messages)  # pyright: ignore[reportAttributeAccessIssue]

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
        mock_genai.Client.side_effect = genai_errors.APIError(
            520, {"message": "Generic API error"}
        )

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
            410, {"message": "Could not list models"}
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
            500, response_json={"message": "Server overloaded"}
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
            500, {"message": "Unknown API error"}
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
            500, {"message": "Streaming failed"}
        )

        provider = GoogleProvider(model="gemini-2.0-flash", api_key="test-key")

        with pytest.raises(ProviderError) as exc_info:
            list(
                provider.stream_chat([Message(role=MessageRole.USER, content="Hello")])
            )

        assert "streaming error" in str(exc_info.value).lower()


class TestGoogleProviderInitErrors:
    """Tests for initialization error handling."""

    @patch("allos.providers.google.genai")
    @patch("allos.providers.google.os.path.exists")
    @patch("google.oauth2.service_account.Credentials.from_service_account_file")
    def test_init_vertexai_missing_project(
        self, mock_from_sa_file, mock_exists, mock_genai, monkeypatch
    ):
        """Test VertexAI init fails when project ID is missing."""
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/path/to/sa.json")

        # Mock file exists but return None project
        mock_exists.return_value = True
        mock_creds = MagicMock()
        mock_from_sa_file.return_value = mock_creds

        # Mock to read file without project_id
        mock_file_data = json.dumps(
            {
                "type": "service_account",
                "client_email": "test@test.iam.gserviceaccount.com",
                # Note: no project_id field
            }
        )

        with patch("builtins.open", mock_open(read_data=mock_file_data)):
            with pytest.raises(ProviderError) as exc_info:
                GoogleProvider(
                    model="gemini-2.0-flash",
                    vertexai=True,
                    # No project parameter, no GOOGLE_CLOUD_PROJECT
                )

            assert "Vertex AI requires a project ID" in str(exc_info.value)
            assert "1. 'project' parameter" in str(exc_info.value)

    @patch("allos.providers.google.genai")
    @patch("google.auth.default")
    def test_init_vertexai_adc_fallback(self, mock_auth_default, mock_genai):
        """Test VertexAI initialization with ADC fallback (no explicit creds)."""
        # Mock ADC returns credentials and project
        mock_creds = MagicMock()
        mock_auth_default.return_value = (mock_creds, "adc-project")

        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.list.return_value = [
            types.Model(
                name="publishers/google/models/gemini-2.0-flash",
                input_token_limit=1048576,
            )
        ]

        provider = GoogleProvider(
            model="gemini-2.0-flash",
            vertexai=True,
            project="explicit-project",
        )

        assert provider.project == "explicit-project"  # pyright: ignore[reportAttributeAccessIssue]
        # Should call Client WITHOUT credentials (ADC fallback)
        mock_genai.Client.assert_called_once_with(
            vertexai=True,
            project="explicit-project",
            location="us-central1",
        )

    @patch("allos.providers.google.genai")
    def test_init_generic_exception(self, mock_genai):
        """Test initialization with generic non-API exception."""
        mock_genai.Client.side_effect = ValueError("Network error")

        with pytest.raises(ProviderError) as exc_info:
            GoogleProvider(
                model="gemini-2.0-flash",
                api_key="test-key",
            )

        assert "Failed to initialize Google client" in str(exc_info.value)
        assert "Network error" in str(exc_info.value)

    @patch("allos.providers.google.genai")
    def test_init_client_error(self, mock_genai):
        """Test initialization with genai ClientError (4xx)."""
        error = genai_errors.ClientError(403, {"message": "Invalid API key"})
        error.message = "Invalid API key"
        mock_genai.Client.side_effect = error

        with pytest.raises(ProviderError) as exc_info:
            GoogleProvider(
                model="gemini-2.0-flash",
                api_key="bad-key",
            )

        assert "Authentication or configuration error" in str(exc_info.value)

    @patch("allos.providers.google.genai")
    def test_init_server_error(self, mock_genai):
        """Test initialization with genai ServerError (5xx)."""
        error = genai_errors.ServerError(500, {"message": "Internal server error"})
        error.message = "Internal server error"
        mock_genai.Client.side_effect = error

        with pytest.raises(ProviderError) as exc_info:
            GoogleProvider(
                model="gemini-2.0-flash",
                api_key="test-key",
            )

        assert "Google API server error" in str(exc_info.value)

    @patch("allos.providers.google.genai")
    def test_init_api_error(self, mock_genai):
        """Test initialization with generic genai APIError."""
        error = genai_errors.APIError(500, {"message": "API Error"})
        error.message = "API error"
        mock_genai.Client.side_effect = error

        with pytest.raises(ProviderError) as exc_info:
            GoogleProvider(
                model="gemini-2.0-flash",
                api_key="test-key",
            )

        assert "Google API error" in str(exc_info.value)


class TestGoogleProviderCredentialLoading:
    """Tests for _load_vertex_credentials priority chain."""

    @patch("allos.providers.google.genai")
    def test_load_with_explicit_credentials_object(self, mock_genai, mock_credentials):
        """Test Priority 1: Explicit credentials object."""
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.list.return_value = [
            types.Model(
                name="publishers/google/models/gemini-2.0-flash",
                input_token_limit=1048576,
            )
        ]

        provider = GoogleProvider(
            model="gemini-2.0-flash",
            vertexai=True,
            project="test-project",
            credentials=mock_credentials,
        )

        assert provider.project == "test-project"  # pyright: ignore[reportAttributeAccessIssue]
        # Should pass the explicit credentials
        mock_genai.Client.assert_called_once_with(
            vertexai=True,
            project="test-project",
            location="us-central1",
            credentials=mock_credentials,
        )

    @patch("allos.providers.google.genai")
    @patch("allos.providers.google.os.path.exists")
    @patch("google.oauth2.service_account.Credentials.from_service_account_file")
    def test_load_with_credentials_path(
        self, mock_from_sa_file, mock_exists, mock_genai, mock_credentials
    ):
        """Test Priority 2: credentials_path parameter."""
        mock_exists.return_value = True
        mock_from_sa_file.return_value = mock_credentials

        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.list.return_value = [
            types.Model(
                name="publishers/google/models/gemini-2.0-flash",
                input_token_limit=1048576,
            )
        ]

        GoogleProvider(
            model="gemini-2.0-flash",
            vertexai=True,
            project="test-project",
            credentials_path="/explicit/path/to/sa.json",
        )

        mock_from_sa_file.assert_called_once_with(
            "/explicit/path/to/sa.json",
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        mock_genai.Client.assert_called_once_with(
            vertexai=True,
            project="test-project",
            location="us-central1",
            credentials=mock_credentials,
        )

    @patch("allos.providers.google.genai")
    @patch("google.oauth2.service_account.Credentials.from_service_account_info")
    def test_load_with_credentials_json_dict(
        self, mock_from_sa_info, mock_genai, mock_credentials, mock_service_account_json
    ):
        """Test Priority 4: credentials_json as dict."""
        mock_from_sa_info.return_value = mock_credentials

        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.list.return_value = [
            types.Model(
                name="publishers/google/models/gemini-2.0-flash",
                input_token_limit=1048576,
            )
        ]

        GoogleProvider(
            model="gemini-2.0-flash",
            vertexai=True,
            project="test-project",
            credentials_json=mock_service_account_json,
        )

        mock_from_sa_info.assert_called_once_with(
            mock_service_account_json,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        mock_genai.Client.assert_called_once_with(
            vertexai=True,
            project="test-project",
            location="us-central1",
            credentials=mock_credentials,
        )

    @patch("allos.providers.google.genai")
    @patch("google.oauth2.service_account.Credentials.from_service_account_info")
    def test_load_with_credentials_json_string(
        self, mock_from_sa_info, mock_genai, mock_credentials, mock_service_account_json
    ):
        """Test Priority 4: credentials_json as JSON string."""
        mock_from_sa_info.return_value = mock_credentials

        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.list.return_value = [
            types.Model(
                name="publishers/google/models/gemini-2.0-flash",
                input_token_limit=1048576,
            )
        ]

        json_string = json.dumps(mock_service_account_json)

        GoogleProvider(
            model="gemini-2.0-flash",
            vertexai=True,
            project="test-project",
            credentials_json=json_string,
        )

        mock_from_sa_info.assert_called_once()
        assert mock_from_sa_info.call_args[0][0] == mock_service_account_json

    @patch("allos.providers.google.genai")
    @patch("google.auth.default")
    @patch("google.auth.impersonated_credentials.Credentials")
    def test_load_with_impersonation(
        self, mock_impersonated, mock_auth_default, mock_genai, mock_credentials
    ):
        """Test Priority 5: Service account impersonation."""
        # Mock ADC source credentials
        source_creds = MagicMock()
        mock_auth_default.return_value = (source_creds, "source-project")

        # Mock impersonated credentials
        mock_impersonated.return_value = mock_credentials

        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.list.return_value = [
            types.Model(
                name="publishers/google/models/gemini-2.0-flash",
                input_token_limit=1048576,
            )
        ]

        GoogleProvider(
            model="gemini-2.0-flash",
            vertexai=True,
            project="test-project",
            impersonate_service_account="sa@project.iam.gserviceaccount.com",
        )

        mock_impersonated.assert_called_once_with(
            source_credentials=source_creds,
            target_principal="sa@project.iam.gserviceaccount.com",
            target_scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )

    @patch("allos.providers.google.genai")
    @patch("google.auth.default")
    def test_load_with_adc_fallback(self, mock_auth_default, mock_genai):
        """Test Priority 6: ADC fallback with project detection."""
        mock_creds = MagicMock()
        mock_auth_default.return_value = (mock_creds, "detected-project")

        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.list.return_value = [
            types.Model(
                name="publishers/google/models/gemini-2.0-flash",
                input_token_limit=1048576,
            )
        ]

        # No explicit credentials - should fall back to ADC
        provider = GoogleProvider(
            model="gemini-2.0-flash",
            vertexai=True,
            # Use detected project
        )

        assert provider.project == "detected-project"  # pyright: ignore[reportAttributeAccessIssue]
        # ADC means credentials=None (not passed)
        mock_genai.Client.assert_called_once_with(
            vertexai=True,
            project="detected-project",
            location="us-central1",
        )


class TestGoogleProviderFileCredentials:
    """Tests for _load_credentials_from_file method."""

    @patch("allos.providers.google.genai")
    @patch("allos.providers.google.os.path.exists")
    def test_file_not_found(self, mock_exists, mock_genai, monkeypatch):
        """Test error when service account file doesn't exist."""
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/nonexistent/sa.json")
        mock_exists.return_value = False

        with pytest.raises(ProviderError) as exc_info:
            GoogleProvider(
                model="gemini-2.0-flash",
                vertexai=True,
                project="test-project",
            )

        assert "Service account file not found" in str(exc_info.value)
        assert "/nonexistent/sa.json" in str(exc_info.value)

    @patch("allos.providers.google.genai")
    @patch("allos.providers.google.os.path.exists")
    @patch("google.oauth2.service_account.Credentials.from_service_account_file")
    def test_file_invalid_json(
        self, mock_from_sa_file, mock_exists, mock_genai, monkeypatch, mock_credentials
    ):
        """Test error when service account file contains invalid JSON."""
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/path/to/invalid.json")
        mock_exists.return_value = True
        # Mock credential loading to succeed
        mock_from_sa_file.return_value = mock_credentials

        # Mock file read for project extraction with invalid JSON
        mock_file = mock_open(read_data="{ invalid json }")
        with patch("builtins.open", mock_file):
            with pytest.raises(ProviderError) as exc_info:
                GoogleProvider(
                    model="gemini-2.0-flash",
                    vertexai=True,
                    # No project - will try to extract from file
                )

            assert "Invalid JSON in service account file" in str(exc_info.value)

    @patch("allos.providers.google.genai")
    @patch("allos.providers.google.os.path.exists")
    @patch("google.oauth2.service_account.Credentials.from_service_account_file")
    def test_file_with_project_extraction(
        self,
        mock_from_sa_file,
        mock_exists,
        mock_genai,
        mock_credentials,
        mock_service_account_json,
        monkeypatch,
    ):
        """Test project extraction from service account JSON."""
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/path/to/sa.json")
        mock_exists.return_value = True
        mock_from_sa_file.return_value = mock_credentials

        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.list.return_value = [
            types.Model(
                name="publishers/google/models/gemini-2.0-flash",
                input_token_limit=1048576,
            )
        ]

        # Mock file read for project extraction
        with patch(
            "builtins.open", mock_open(read_data=json.dumps(mock_service_account_json))
        ):
            provider = GoogleProvider(
                model="gemini-2.0-flash",
                vertexai=True,
                # No explicit project - should extract from file
            )

        # Should extract project from JSON
        assert provider.project == "allos-test-project-from-json"  # pyright: ignore[reportAttributeAccessIssue]

    @patch("allos.providers.google.genai")
    @patch("allos.providers.google.os.path.exists")
    @patch("google.oauth2.service_account.Credentials.from_service_account_file")
    def test_file_with_explicit_project_priority(
        self,
        mock_from_sa_file,
        mock_exists,
        mock_genai,
        mock_credentials,
        mock_service_account_json,
        monkeypatch,
    ):
        """Test explicit project takes priority over file project."""
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/path/to/sa.json")
        mock_exists.return_value = True
        mock_from_sa_file.return_value = mock_credentials

        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.list.return_value = [
            types.Model(
                name="publishers/google/models/gemini-2.0-flash",
                input_token_limit=1048576,
            )
        ]

        with patch(
            "builtins.open", mock_open(read_data=json.dumps(mock_service_account_json))
        ):
            provider = GoogleProvider(
                model="gemini-2.0-flash",
                vertexai=True,
                project="explicit-project",  # Explicit project
            )

        # Should use explicit project, not the one from file
        assert provider.project == "explicit-project"  # pyright: ignore[reportAttributeAccessIssue]

    @patch("allos.providers.google.genai")
    @patch("allos.providers.google.os.path.exists")
    @patch("google.oauth2.service_account.Credentials.from_service_account_file")
    def test_file_generic_exception(
        self, mock_from_sa_file, mock_exists, mock_genai, monkeypatch
    ):
        """Test generic exception handling in file loading."""
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/path/to/sa.json")
        mock_exists.return_value = True
        mock_from_sa_file.side_effect = IOError("Permission denied")

        with pytest.raises(ProviderError) as exc_info:
            GoogleProvider(
                model="gemini-2.0-flash",
                vertexai=True,
                project="test-project",
            )

        assert "Failed to load credentials from" in str(exc_info.value)
        assert "Permission denied" in str(exc_info.value)


class TestGoogleProviderADC:
    """Tests for _load_adc_credentials method."""

    @patch("allos.providers.google.genai")
    @patch("google.auth.default")
    def test_adc_with_explicit_project(self, mock_auth_default, mock_genai):
        """Test ADC with explicit project parameter."""
        mock_creds = MagicMock()
        mock_auth_default.return_value = (mock_creds, "detected-project")

        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.list.return_value = [
            types.Model(
                name="publishers/google/models/gemini-2.0-flash",
                input_token_limit=1048576,
            )
        ]

        provider = GoogleProvider(
            model="gemini-2.0-flash",
            vertexai=True,
            project="explicit-project",  # Explicit project
        )

        # Should use explicit, not detected
        assert provider.project == "explicit-project"  # pyright: ignore[reportAttributeAccessIssue]
        mock_genai.Client.assert_called_once_with(
            vertexai=True,
            project="explicit-project",
            location="us-central1",
        )

    @patch("allos.providers.google.genai")
    @patch("google.auth.default")
    def test_adc_with_detected_project(self, mock_auth_default, mock_genai):
        """Test ADC with auto-detected project."""
        mock_creds = MagicMock()
        mock_auth_default.return_value = (mock_creds, "auto-detected-project")

        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.list.return_value = [
            types.Model(
                name="publishers/google/models/gemini-2.0-flash",
                input_token_limit=1048576,
            )
        ]

        provider = GoogleProvider(
            model="gemini-2.0-flash",
            vertexai=True,
            # No explicit project
        )

        # Should use detected project
        assert provider.project == "auto-detected-project"  # pyright: ignore[reportAttributeAccessIssue]

    @patch("allos.providers.google.genai")
    @patch("google.auth.default")
    def test_adc_not_configured(self, mock_auth_default, mock_genai):
        """Test ADC when credentials are not configured."""
        mock_auth_default.side_effect = DefaultCredentialsError(
            "Could not automatically determine credentials"
        )

        with pytest.raises(ProviderError) as exc_info:
            GoogleProvider(
                model="gemini-2.0-flash",
                vertexai=True,
                project="test-project",
            )

        error_msg = str(exc_info.value)
        assert "Application Default Credentials (ADC) not found" in error_msg
        assert "gcloud auth application-default login" in error_msg
        assert "credentials_path=" in error_msg

    @patch("allos.providers.google.genai")
    @patch("google.auth.default")
    def test_adc_project_detection_failure(self, mock_auth_default, mock_genai):
        """Test ADC when project detection fails but credentials exist."""
        # ADC exists but can't detect project (None)
        mock_creds = MagicMock()
        mock_auth_default.return_value = (mock_creds, None)

        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.list.return_value = [
            types.Model(
                name="publishers/google/models/gemini-2.0-flash",
                input_token_limit=1048576,
            )
        ]

        # Should fail because no project available
        with pytest.raises(ProviderError) as exc_info:
            GoogleProvider(
                model="gemini-2.0-flash",
                vertexai=True,
                # No explicit project, and ADC can't detect it
            )

        assert "Vertex AI requires a project ID" in str(exc_info.value)

    @patch("allos.providers.google.genai")
    @patch("google.auth.default")
    def test_adc_with_env_project(self, mock_auth_default, mock_genai, monkeypatch):
        """Test ADC with GOOGLE_CLOUD_PROJECT environment variable."""
        mock_creds = MagicMock()
        mock_auth_default.return_value = (mock_creds, None)

        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "env-project")

        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.list.return_value = [
            types.Model(
                name="publishers/google/models/gemini-2.0-flash",
                input_token_limit=1048576,
            )
        ]

        provider = GoogleProvider(
            model="gemini-2.0-flash",
            vertexai=True,
        )

        # Should use project from environment
        assert provider.project == "env-project"  # pyright: ignore[reportAttributeAccessIssue]

    @patch("allos.providers.google.genai")
    @patch("google.auth.default")
    @patch("allos.providers.google.logger")
    def test_adc_generic_exception_with_explicit_project(
        self, mock_logger, mock_auth_default, mock_genai, monkeypatch
    ):
        """Test ADC when generic exception occurs but explicit project is provided."""
        # google.auth.default() raises a generic exception (not DefaultCredentialsError)
        mock_auth_default.side_effect = RuntimeError("Unexpected error")

        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.list.return_value = [
            types.Model(
                name="publishers/google/models/gemini-2.0-flash",
                input_token_limit=1048576,
            )
        ]

        # Should succeed because we have explicit project
        provider = GoogleProvider(
            model="gemini-2.0-flash",
            vertexai=True,
            project="explicit-project",
        )

        # Should log warning about project detection failure
        mock_logger.warning.assert_called_once()
        warning_call = mock_logger.warning.call_args[0][0]
        assert "Could not detect project from ADC" in warning_call
        assert "Unexpected error" in warning_call

        # Should still work with explicit project
        assert provider.project == "explicit-project"  # pyright: ignore[reportAttributeAccessIssue]


class TestGoogleProviderJSONCredentials:
    """Tests for _load_credentials_from_json method."""

    @patch("allos.providers.google.genai")
    @patch("google.oauth2.service_account.Credentials.from_service_account_info")
    def test_json_as_dict_success(
        self, mock_from_sa_info, mock_genai, mock_credentials, mock_service_account_json
    ):
        """Test loading credentials from JSON dict."""
        mock_from_sa_info.return_value = mock_credentials

        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.list.return_value = [
            types.Model(
                name="publishers/google/models/gemini-2.0-flash",
                input_token_limit=1048576,
            )
        ]

        provider = GoogleProvider(
            model="gemini-2.0-flash",
            vertexai=True,
            project="test-project",
            credentials_json=mock_service_account_json,
        )

        mock_from_sa_info.assert_called_once_with(
            mock_service_account_json,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        assert provider.project == "test-project"  # pyright: ignore[reportAttributeAccessIssue]

    @patch("allos.providers.google.genai")
    @patch("google.oauth2.service_account.Credentials.from_service_account_info")
    def test_json_as_string_success(
        self, mock_from_sa_info, mock_genai, mock_credentials, mock_service_account_json
    ):
        """Test loading credentials from JSON string."""
        mock_from_sa_info.return_value = mock_credentials

        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.list.return_value = [
            types.Model(
                name="publishers/google/models/gemini-2.0-flash",
                input_token_limit=1048576,
            )
        ]

        json_string = json.dumps(mock_service_account_json)

        GoogleProvider(
            model="gemini-2.0-flash",
            vertexai=True,
            project="test-project",
            credentials_json=json_string,
        )

        # Should parse string to dict
        call_args = mock_from_sa_info.call_args[0][0]
        assert call_args == mock_service_account_json

    @patch("allos.providers.google.genai")
    def test_json_invalid_type(self, mock_genai):
        """Test error when credentials_json is invalid type."""
        with pytest.raises(ProviderError) as exc_info:
            GoogleProvider(
                model="gemini-2.0-flash",
                vertexai=True,
                project="test-project",
                credentials_json=12345,  # Invalid type
            )

        assert "credentials_json must be a dict or JSON string" in str(exc_info.value)

    @patch("allos.providers.google.genai")
    def test_json_invalid_string(self, mock_genai):
        """Test error when credentials_json is invalid JSON string."""
        with pytest.raises(ProviderError) as exc_info:
            GoogleProvider(
                model="gemini-2.0-flash",
                vertexai=True,
                project="test-project",
                credentials_json="{ invalid json }",
            )

        assert "Invalid JSON in credentials_json" in str(exc_info.value)

    @patch("allos.providers.google.genai")
    @patch("google.oauth2.service_account.Credentials.from_service_account_info")
    def test_json_project_extraction(
        self, mock_from_sa_info, mock_genai, mock_credentials, mock_service_account_json
    ):
        """Test project extraction from credentials_json."""
        mock_from_sa_info.return_value = mock_credentials

        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.list.return_value = [
            types.Model(
                name="publishers/google/models/gemini-2.0-flash",
                input_token_limit=1048576,
            )
        ]

        provider = GoogleProvider(
            model="gemini-2.0-flash",
            vertexai=True,
            # No explicit project
            credentials_json=mock_service_account_json,
        )

        # Should extract project from JSON
        assert provider.project == "allos-test-project-from-json"  # pyright: ignore[reportAttributeAccessIssue]

    @patch("allos.providers.google.genai")
    @patch("google.oauth2.service_account.Credentials.from_service_account_info")
    def test_json_explicit_project_priority(
        self, mock_from_sa_info, mock_genai, mock_credentials, mock_service_account_json
    ):
        """Test explicit project takes priority over JSON project."""
        mock_from_sa_info.return_value = mock_credentials

        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.list.return_value = [
            types.Model(
                name="publishers/google/models/gemini-2.0-flash",
                input_token_limit=1048576,
            )
        ]

        provider = GoogleProvider(
            model="gemini-2.0-flash",
            vertexai=True,
            project="explicit-project",
            credentials_json=mock_service_account_json,
        )

        # Should use explicit, not from JSON
        assert provider.project == "explicit-project"  # pyright: ignore[reportAttributeAccessIssue]

    @patch("allos.providers.google.genai")
    @patch("google.oauth2.service_account.Credentials.from_service_account_info")
    def test_json_without_project_field(
        self, mock_from_sa_info, mock_genai, mock_credentials, monkeypatch
    ):
        """Test JSON without project_id field."""
        mock_from_sa_info.return_value = mock_credentials

        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.list.return_value = [
            types.Model(
                name="publishers/google/models/gemini-2.0-flash",
                input_token_limit=1048576,
            )
        ]

        json_without_project = {
            "type": "service_account",
            "client_email": "test@test.iam.gserviceaccount.com",
            # No project_id
        }

        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "env-project")

        provider = GoogleProvider(
            model="gemini-2.0-flash",
            vertexai=True,
            credentials_json=json_without_project,
        )

        # Should fall back to environment variable
        assert provider.project == "env-project"  # pyright: ignore[reportAttributeAccessIssue]

    @patch("allos.providers.google.genai")
    @patch("google.oauth2.service_account.Credentials.from_service_account_info")
    def test_json_generic_exception(self, mock_from_sa_info, mock_genai):
        """Test generic exception handling in JSON loading."""
        mock_from_sa_info.side_effect = ValueError("Missing required fields")

        with pytest.raises(ProviderError) as exc_info:
            GoogleProvider(
                model="gemini-2.0-flash",
                vertexai=True,
                project="test-project",
                credentials_json={"type": "service_account"},
            )

        assert "Failed to load credentials from JSON" in str(exc_info.value)
        assert "Missing required fields" in str(exc_info.value)


class TestGoogleProviderImpersonation:
    """Tests for _load_impersonated_credentials method."""

    @patch("allos.providers.google.genai")
    @patch("google.auth.default")
    @patch("google.auth.impersonated_credentials.Credentials")
    def test_impersonation_success_explicit_project(
        self, mock_impersonated, mock_auth_default, mock_genai, mock_credentials
    ):
        """Test successful impersonation with explicit project."""
        source_creds = MagicMock()
        mock_auth_default.return_value = (source_creds, "source-project")
        mock_impersonated.return_value = mock_credentials

        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.list.return_value = [
            types.Model(
                name="publishers/google/models/gemini-2.0-flash",
                input_token_limit=1048576,
            )
        ]

        provider = GoogleProvider(
            model="gemini-2.0-flash",
            vertexai=True,
            project="explicit-project",
            impersonate_service_account="target@project.iam.gserviceaccount.com",
        )

        mock_impersonated.assert_called_once_with(
            source_credentials=source_creds,
            target_principal="target@project.iam.gserviceaccount.com",
            target_scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        assert provider.project == "explicit-project"  # pyright: ignore[reportAttributeAccessIssue]

    @patch("allos.providers.google.genai")
    @patch("google.auth.default")
    @patch("google.auth.impersonated_credentials.Credentials")
    def test_impersonation_detected_project(
        self, mock_impersonated, mock_auth_default, mock_genai, mock_credentials
    ):
        """Test impersonation with detected project from source credentials."""
        source_creds = MagicMock()
        mock_auth_default.return_value = (source_creds, "detected-project")
        mock_impersonated.return_value = mock_credentials

        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.list.return_value = [
            types.Model(
                name="publishers/google/models/gemini-2.0-flash",
                input_token_limit=1048576,
            )
        ]

        provider = GoogleProvider(
            model="gemini-2.0-flash",
            vertexai=True,
            # No explicit project
            impersonate_service_account="target@project.iam.gserviceaccount.com",
        )

        # Should use detected project
        assert provider.project == "detected-project"  # pyright: ignore[reportAttributeAccessIssue]

    @patch("allos.providers.google.genai")
    @patch("google.auth.default")
    def test_impersonation_failure_permissions(self, mock_auth_default, mock_genai):
        """Test impersonation failure due to insufficient permissions."""
        source_creds = MagicMock()
        mock_auth_default.return_value = (source_creds, "source-project")

        # Import inside test to patch properly
        with patch("google.auth.impersonated_credentials.Credentials") as mock_imp:
            mock_imp.side_effect = Exception("Permission denied")

            with pytest.raises(ProviderError) as exc_info:
                GoogleProvider(
                    model="gemini-2.0-flash",
                    vertexai=True,
                    project="test-project",
                    impersonate_service_account="target@project.iam.gserviceaccount.com",
                )

            error_msg = str(exc_info.value)
            assert "Failed to impersonate service account" in error_msg
            assert "target@project.iam.gserviceaccount.com" in error_msg
            assert "roles/iam.serviceAccountTokenCreator" in error_msg

    @patch("allos.providers.google.genai")
    @patch("google.auth.default")
    @patch("google.auth.impersonated_credentials.Credentials")
    def test_impersonation_custom_scopes(
        self, mock_impersonated, mock_auth_default, mock_genai, mock_credentials
    ):
        """Test impersonation with custom scopes."""
        source_creds = MagicMock()
        mock_auth_default.return_value = (source_creds, "source-project")
        mock_impersonated.return_value = mock_credentials

        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.list.return_value = [
            types.Model(
                name="publishers/google/models/gemini-2.0-flash",
                input_token_limit=1048576,
            )
        ]

        custom_scopes = [
            "https://www.googleapis.com/auth/cloud-platform",
            "https://www.googleapis.com/auth/compute",
        ]

        GoogleProvider(
            model="gemini-2.0-flash",
            vertexai=True,
            project="test-project",
            impersonate_service_account="target@project.iam.gserviceaccount.com",
            impersonation_scopes=custom_scopes,
        )

        mock_impersonated.assert_called_once_with(
            source_credentials=source_creds,
            target_principal="target@project.iam.gserviceaccount.com",
            target_scopes=custom_scopes,
        )

    @patch("allos.providers.google.genai")
    @patch("google.auth.default")
    @patch("google.auth.impersonated_credentials.Credentials")
    @patch("allos.providers.google.logger")
    def test_impersonation_logging(
        self,
        mock_logger,
        mock_impersonated,
        mock_auth_default,
        mock_genai,
        mock_credentials,
    ):
        """Test that impersonation logs details."""
        source_creds = MagicMock()
        mock_auth_default.return_value = (source_creds, "source-project")
        mock_impersonated.return_value = mock_credentials

        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.list.return_value = [
            types.Model(
                name="publishers/google/models/gemini-2.0-flash",
                input_token_limit=1048576,
            )
        ]

        GoogleProvider(
            model="gemini-2.0-flash",
            vertexai=True,
            project="test-project",
            impersonate_service_account="target@project.iam.gserviceaccount.com",
        )

        # Check that impersonation was logged
        debug_calls = [call[0][0] for call in mock_logger.debug.call_args_list]
        assert any("Impersonating" in call for call in debug_calls)


class TestGoogleProviderThoughtSignatures:
    """Test thought signatures functionality in Google provider."""

    @patch("allos.providers.google.genai")
    def test_parse_response_with_thought_signature(self, mock_genai):
        """Test _parse_response extracts thought signatures from function calls."""
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.list.return_value = [
            types.Model(name="models/gemini-2.0-flash", input_token_limit=1048576)
        ]

        # Mock function call response with thought signature
        mock_fc = MagicMock()
        mock_fc.name = "search"
        mock_fc.args = {"query": "test"}

        # Mock the part with thought signature
        mock_part = MagicMock()
        mock_part.text = None  # No text content, only function call
        mock_part.function_call = mock_fc
        mock_part.thought_signature = b"mock_thought_signature_bytes"

        # Mock the content containing the part
        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        # Mock the candidate
        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        # Mock the response
        mock_response = MagicMock()
        mock_response.text = None
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = None

        provider = GoogleProvider(model="gemini-2.0-flash", api_key="test-key")

        # Parse the response
        content, tool_calls, thought_signatures = provider._parse_response(  # pyright: ignore[reportAttributeAccessIssue]
            mock_response
        )  # pyright: ignore[reportAttributeAccessIssue]

        # Verify thought signature was extracted
        assert thought_signatures is not None
        assert len(thought_signatures) == 1
        # The tool_call_id is generated, so we check the value exists
        signature_values = list(thought_signatures.values())
        assert signature_values[0] == b"mock_thought_signature_bytes"

    @patch("allos.providers.google.genai")
    def test_chat_with_thought_signatures(self, mock_genai):
        """Test full chat flow with thought signatures in response."""
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.list.return_value = [
            types.Model(name="models/gemini-2.0-flash", input_token_limit=1048576)
        ]

        # Mock function call with thought signature
        mock_fc = MagicMock()
        mock_fc.name = "search"
        mock_fc.args = {"query": "test"}

        mock_part = MagicMock()
        mock_part.text = None  # No text content, only function call
        mock_part.function_call = mock_fc
        mock_part.thought_signature = b"signature_data"

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_response = MagicMock()
        mock_response.text = None
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = None
        mock_client.models.generate_content.return_value = mock_response

        provider = GoogleProvider(model="gemini-2.0-flash", api_key="test-key")

        messages = [Message(role=MessageRole.USER, content="Search for test")]
        response = provider.chat(messages, tools=MOCK_TOOLS)

        # Verify thought signatures are in response
        assert response.thought_signatures is not None
        assert len(response.thought_signatures) == 1
        assert b"signature_data" in response.thought_signatures.values()

    @patch("allos.providers.google.genai")
    def test_convert_messages_with_thought_signatures(self, mock_genai):
        """Test _convert_messages includes thought signatures in tool calls."""
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.list.return_value = [
            types.Model(name="models/gemini-2.0-flash", input_token_limit=1048576)
        ]

        provider = GoogleProvider(model="gemini-2.0-flash", api_key="test-key")

        # Create messages with tool calls that have thought signatures
        messages = [
            Message(role=MessageRole.USER, content="Search for something"),
            Message(
                role=MessageRole.ASSISTANT,
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_123",
                        name="search",
                        arguments={"query": "test"},
                    )
                ],
                thought_signatures={"call_123": b"assistant_thought_sig"},
            ),
        ]

        system_instruction, contents = provider._convert_messages(messages)  # pyright: ignore[reportAttributeAccessIssue]

        # Find the assistant content with function call
        assistant_content = contents[1]
        assert assistant_content.role == "model"

        # Find the function call part
        fc_parts = [
            p
            for p in assistant_content.parts
            if hasattr(p, "function_call") and p.function_call
        ]
        assert len(fc_parts) == 1

        # Verify thought signature was attached
        fc_part = fc_parts[0]
        assert hasattr(fc_part, "thought_signature")
        assert fc_part.thought_signature == b"assistant_thought_sig"

    @patch("allos.providers.google.genai")
    def test_stream_chat_with_thought_signatures(self, mock_genai):
        """Test stream_chat yields thought signatures."""
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.list.return_value = [
            types.Model(name="models/gemini-2.0-flash", input_token_limit=1048576)
        ]

        # Mock streaming chunk with function call and thought signature
        mock_fc = MagicMock()
        mock_fc.name = "search"
        mock_fc.args = {"query": "test"}

        mock_part = MagicMock()
        mock_part.function_call = mock_fc
        mock_part.thought_signature = b"stream_thought_sig"

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_chunk = MagicMock()
        mock_chunk.text = None
        mock_chunk.function_calls = [mock_fc]
        mock_chunk.candidates = [mock_candidate]

        mock_client.models.generate_content_stream.return_value = iter([mock_chunk])

        provider = GoogleProvider(model="gemini-2.0-flash", api_key="test-key")

        messages = [Message(role=MessageRole.USER, content="Search")]
        chunks = list(provider.stream_chat(messages, tools=MOCK_TOOLS))

        # Find chunks with thought signatures
        thought_chunks = [c for c in chunks if c.thought_signatures]
        assert len(thought_chunks) >= 1

        # Verify the thought signature is present
        thought_sig_found = False
        for chunk in thought_chunks:
            if chunk.thought_signatures:
                if b"stream_thought_sig" in chunk.thought_signatures.values():
                    thought_sig_found = True
                    break
        assert thought_sig_found

    @patch("allos.providers.google.genai")
    def test_stream_chat_empty_candidates(self, mock_genai):
        """Test stream_chat handles empty candidates gracefully."""
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.list.return_value = [
            types.Model(name="models/gemini-2.0-flash", input_token_limit=1048576)
        ]

        # Mock chunk with function_calls but empty candidates
        mock_fc = MagicMock()
        mock_fc.name = "search"
        mock_fc.args = {"query": "test"}

        mock_chunk = MagicMock()
        mock_chunk.text = None
        mock_chunk.function_calls = [mock_fc]
        mock_chunk.candidates = []  # Empty candidates

        mock_client.models.generate_content_stream.return_value = iter([mock_chunk])

        provider = GoogleProvider(model="gemini-2.0-flash", api_key="test-key")

        messages = [Message(role=MessageRole.USER, content="Search")]

        # Should not raise an error
        chunks = list(provider.stream_chat(messages))
        # Should get at least the final metadata chunk
        assert len(chunks) > 0

    @patch("allos.providers.google.genai")
    def test_stream_chat_no_content_or_parts(self, mock_genai):
        """Test stream_chat handles candidates without content/parts."""
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.list.return_value = [
            types.Model(name="models/gemini-2.0-flash", input_token_limit=1048576)
        ]

        # Mock chunk with candidates but no content
        mock_fc = MagicMock()
        mock_fc.name = "search"
        mock_fc.args = {"query": "test"}

        mock_candidate = MagicMock()
        mock_candidate.content = None  # No content

        mock_chunk = MagicMock()
        mock_chunk.text = None
        mock_chunk.function_calls = [mock_fc]
        mock_chunk.candidates = [mock_candidate]

        mock_client.models.generate_content_stream.return_value = iter([mock_chunk])

        provider = GoogleProvider(model="gemini-2.0-flash", api_key="test-key")

        messages = [Message(role=MessageRole.USER, content="Search")]

        # Should not raise an error
        chunks = list(provider.stream_chat(messages))
        assert len(chunks) > 0

    @patch("allos.providers.google.genai")
    def test_stream_chat_part_without_function_call(self, mock_genai):
        """Test stream_chat handles parts without function_call (text parts)."""
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.list.return_value = [
            types.Model(name="models/gemini-2.0-flash", input_token_limit=1048576)
        ]

        # Mock chunk with function_calls flag but parts that don't have function_call
        mock_fc = MagicMock()
        mock_fc.name = "search"
        mock_fc.args = {"query": "test"}

        # Create a text part without function_call
        mock_text_part = MagicMock()
        mock_text_part.function_call = None  # No function call on this part
        mock_text_part.text = "Some text"

        # Create a function call part
        mock_fc_part = MagicMock()
        mock_fc_part.function_call = mock_fc
        mock_fc_part.thought_signature = None

        mock_content = MagicMock()
        # Mix of parts: text part and function call part
        mock_content.parts = [mock_text_part, mock_fc_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_chunk = MagicMock()
        mock_chunk.text = None
        mock_chunk.function_calls = [mock_fc]
        mock_chunk.candidates = [mock_candidate]

        mock_client.models.generate_content_stream.return_value = iter([mock_chunk])

        provider = GoogleProvider(model="gemini-2.0-flash", api_key="test-key")

        messages = [Message(role=MessageRole.USER, content="Search")]

        # Should not raise an error, should skip the text part and process FC part
        chunks = list(provider.stream_chat(messages))

        # Should have tool call chunks
        tool_chunks = [c for c in chunks if c.tool_call_done]
        assert len(tool_chunks) >= 1
        assert tool_chunks[0].tool_call_done.name == "search"  # pyright: ignore[reportOptionalMemberAccess]


class TestGoogleProviderHelperMethods:
    """Test helper methods for model suggestions and error messages."""

    @patch("allos.providers.google.genai")
    def test_extract_model_id_with_models_prefix(self, mock_genai):
        """Test extracting model ID from full model name with models/ prefix."""
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.list.return_value = [
            types.Model(name="models/gemini-2.0-flash", input_token_limit=1048576)
        ]

        provider = GoogleProvider(model="gemini-2.0-flash", api_key="test-key")

        # Test with models/ prefix
        model_id = provider._extract_model_id("models/gemini-2.0-flash")  # pyright: ignore[reportAttributeAccessIssue]
        assert model_id == "models/gemini-2.0-flash"

    @patch("allos.providers.google.genai")
    def test_extract_model_id_with_publishers_prefix(self, mock_genai):
        """Test extracting model ID from Vertex AI full name."""
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.list.return_value = [
            types.Model(
                name="publishers/google/models/gemini-2.0-flash",
                input_token_limit=1048576,
            )
        ]

        provider = GoogleProvider(
            model="gemini-2.0-flash",
            vertexai=True,
            project="test-project",
            credentials=MagicMock(),
        )

        # Test with publishers/google/models/ prefix
        model_id = provider._extract_model_id(  # pyright: ignore[reportAttributeAccessIssue]
            "publishers/google/models/gemini-2.0-flash"
        )
        assert model_id == "gemini-2.0-flash"

    @patch("allos.providers.google.genai")
    def test_extract_model_id_without_prefix(self, mock_genai):
        """Test extracting model ID when no prefix exists."""
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_client.models.list.return_value = [
            types.Model(name="models/gemini-2.0-flash", input_token_limit=1048576)
        ]

        provider = GoogleProvider(model="gemini-2.0-flash", api_key="test-key")

        # Test without any prefix (edge case - should return as-is)
        model_id = provider._extract_model_id("gemini-2.0-flash")  # pyright: ignore[reportAttributeAccessIssue]
        assert model_id == "gemini-2.0-flash"

    @patch("allos.providers.google.genai")
    def test_find_similar_models_with_typo(self, mock_genai):
        """Test finding similar models when user makes a typo."""
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_models = [
            types.Model(name="models/gemini-2.0-flash", input_token_limit=1048576),
            types.Model(name="models/gemini-2.5-flash", input_token_limit=1048576),
            types.Model(name="models/gemini-1.5-pro", input_token_limit=2097152),
        ]
        mock_client.models.list.return_value = mock_models

        provider = GoogleProvider(model="gemini-2.0-flash", api_key="test-key")

        # Test with typo: "flsh" instead of "flash"
        suggestions = provider._find_similar_models("gemini-2.0-flsh", mock_models)  # pyright: ignore[reportAttributeAccessIssue]

        # Should suggest the close match
        assert "models/gemini-2.0-flash" in suggestions
        assert len(suggestions) > 0

    @patch("allos.providers.google.genai")
    def test_find_similar_models_with_prefix_match(self, mock_genai):
        """Test finding similar models using prefix when no close match."""
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_models = [
            types.Model(name="models/gemini-2.0-flash", input_token_limit=1048576),
            types.Model(name="models/gemini-2.5-pro", input_token_limit=1048576),
            types.Model(name="models/claude-3-opus", input_token_limit=200000),
        ]
        mock_client.models.list.return_value = mock_models

        provider = GoogleProvider(model="gemini-2.0-flash", api_key="test-key")

        # Test with completely different suffix but same prefix
        suggestions = provider._find_similar_models("gemini-xyz-123", mock_models)  # pyright: ignore[reportAttributeAccessIssue]

        # Should suggest models with "gemini" prefix
        assert any("gemini" in s for s in suggestions)
        # Should not suggest claude
        assert not any("claude" in s for s in suggestions)

    @patch("allos.providers.google.genai")
    def test_find_similar_models_no_match(self, mock_genai):
        """Test finding similar models when nothing matches."""
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client
        mock_models = [
            types.Model(name="models/gemini-2.0-flash", input_token_limit=1048576),
            types.Model(name="models/gemini-2.5-flash", input_token_limit=1048576),
        ]
        mock_client.models.list.return_value = mock_models

        provider = GoogleProvider(model="gemini-2.0-flash", api_key="test-key")

        # Test with completely unrelated model name
        suggestions = provider._find_similar_models(  # pyright: ignore[reportAttributeAccessIssue]
            "totally-different-xyz", mock_models
        )

        # Should return something (fallback to prefix or empty)
        assert isinstance(suggestions, list)
