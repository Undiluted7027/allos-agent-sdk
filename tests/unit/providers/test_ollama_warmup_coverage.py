# tests/unit/providers/test_ollama_warmup_coverage.py

"""Unit tests to cover warm-up detection code paths without actual sleep delays.

These tests use time.time mocking to simulate slow requests and ensure all
warm-up related code paths are covered.
"""

from unittest.mock import MagicMock, patch

import pytest
from ollama._types import ListResponse

from allos.providers.base import Message, MessageRole
from allos.providers.ollama import (
    _MODEL_WARMUP_TRACKER,
    _OLLAMA_CLIENT_POOL,
    OllamaProvider,
)


@pytest.fixture(autouse=True)
def clear_tracking():
    """Clear connection pool and warm-up tracker before and after each test."""
    _OLLAMA_CLIENT_POOL.clear()
    _MODEL_WARMUP_TRACKER.clear()
    yield
    _OLLAMA_CLIENT_POOL.clear()
    _MODEL_WARMUP_TRACKER.clear()


# Mock model list
MOCK_MODEL_LIST: ListResponse = ListResponse(
    models=[ListResponse.Model(model="llama3.1:latest")]
)


def create_mock_show_response(context_length: int = 8192):
    """Create a mock ShowResponse."""
    mock_response = MagicMock()
    mock_response.model_dump.return_value = {
        "modelfile": f"... num_ctx {context_length} ...",
        "parameters": f"num_ctx {context_length}",
        "template": "...",
        "details": {"parameter_size": "8B"},
    }
    return mock_response


@patch("allos.providers.ollama.Client")
@patch("allos.providers.ollama.time")
class TestWarmUpCoverage:
    """Test warm-up detection code coverage without actual delays."""

    def test_chat_warmup_metadata_path(self, mock_time, MockClient, caplog):
        """Test that warm-up detection triggers metadata path in chat()."""
        mock_instance = MockClient.return_value
        mock_instance.list.return_value = MOCK_MODEL_LIST
        mock_instance.show.return_value = create_mock_show_response()

        # Mock chat response
        mock_instance.chat.return_value = {
            "message": {"role": "assistant", "content": "Hello"},
            "done": True,
            "prompt_eval_count": 10,
            "eval_count": 5,
        }

        # Mock time to simulate slow first request (>=10s)
        # time.time() is called multiple times: start_time, elapsed check, _build_metadata
        mock_time.time.side_effect = [0.0, 10.5, 10.5]

        provider = OllamaProvider(model="llama3.1:latest")
        messages = [Message(role=MessageRole.USER, content="Hi")]

        response = provider.chat(messages)

        # Should log warm-up message (lines 499-503)
        assert "loading into memory" in caplog.text
        assert "llama3.1:latest" in caplog.text

        # Metadata should include warm-up info (lines 441-442 in _build_metadata)
        assert response.metadata.provider_specific.ollama is not None
        assert response.metadata.provider_specific.ollama.warm_up is True
        assert response.metadata.provider_specific.ollama.warm_up_duration_seconds >= 10.0

    def test_stream_warmup_metadata_path(self, mock_time, MockClient, caplog):
        """Test that warm-up detection triggers metadata path in stream_chat()."""
        mock_instance = MockClient.return_value
        mock_instance.list.return_value = MOCK_MODEL_LIST
        mock_instance.show.return_value = create_mock_show_response()

        # Mock streaming response
        mock_instance.chat.return_value = iter([
            {"message": {"role": "assistant", "content": "Hello"}, "done": False},
            {
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "prompt_eval_count": 10,
                "eval_count": 5,
            },
        ])

        # Mock time to simulate slow first request (>=10s)
        # time.time() called: start_time, elapsed check, _build_metadata
        mock_time.time.side_effect = [0.0, 11.0, 11.0]

        provider = OllamaProvider(model="llama3.1:latest")
        messages = [Message(role=MessageRole.USER, content="Hi")]

        chunks = list(provider.stream_chat(messages))

        # Should log warm-up message (lines 646-650)
        assert "loading into memory" in caplog.text

        # Get final metadata
        metadata_chunks = [c for c in chunks if c.final_metadata]
        assert len(metadata_chunks) == 1
        metadata = metadata_chunks[0].final_metadata

        # Metadata should include warm-up info
        assert metadata.provider_specific.ollama is not None
        assert metadata.provider_specific.ollama.warm_up is True
        assert metadata.provider_specific.ollama.warm_up_duration_seconds >= 10.0

    def test_chat_fast_request_no_warmup_metadata(self, mock_time, MockClient):
        """Test that fast first request doesn't set warm-up metadata."""
        mock_instance = MockClient.return_value
        mock_instance.list.return_value = MOCK_MODEL_LIST
        mock_instance.show.return_value = create_mock_show_response()

        mock_instance.chat.return_value = {
            "message": {"role": "assistant", "content": "Hello"},
            "done": True,
            "prompt_eval_count": 10,
            "eval_count": 5,
        }

        # Mock time to simulate fast request (<10s)
        # time.time() called: start_time, elapsed check, _build_metadata
        mock_time.time.side_effect = [0.0, 0.5, 0.5]

        provider = OllamaProvider(model="llama3.1:latest")
        messages = [Message(role=MessageRole.USER, content="Hi")]

        response = provider.chat(messages)

        # Metadata should NOT have warm-up flag
        assert (
            response.metadata.provider_specific.ollama is None
            or not response.metadata.provider_specific.ollama.warm_up
        )

    def test_stream_fast_request_no_warmup_metadata(self, mock_time, MockClient):
        """Test that fast first streaming request doesn't set warm-up metadata."""
        mock_instance = MockClient.return_value
        mock_instance.list.return_value = MOCK_MODEL_LIST
        mock_instance.show.return_value = create_mock_show_response()

        mock_instance.chat.return_value = iter([
            {"message": {"role": "assistant", "content": "Hello"}, "done": False},
            {
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "prompt_eval_count": 10,
                "eval_count": 5,
            },
        ])

        # Mock time to simulate fast request (<10s)
        # time.time() called: start_time, elapsed check, _build_metadata
        mock_time.time.side_effect = [0.0, 1.0, 1.0]

        provider = OllamaProvider(model="llama3.1:latest")
        messages = [Message(role=MessageRole.USER, content="Hi")]

        chunks = list(provider.stream_chat(messages))

        # Get final metadata
        metadata_chunks = [c for c in chunks if c.final_metadata]
        assert len(metadata_chunks) == 1
        metadata = metadata_chunks[0].final_metadata

        # Metadata should NOT have warm-up flag
        assert (
            metadata.provider_specific.ollama is None
            or not metadata.provider_specific.ollama.warm_up
        )
