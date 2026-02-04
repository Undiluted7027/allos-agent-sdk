# tests/unit/providers/test_ollama_warmup.py

"""Tests for model warm-up detection in Ollama provider.

These tests verify that the provider correctly detects and logs model warm-up
(first request taking ≥10s), and that warm-up information is included in metadata.
"""

import time
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
class TestWarmUpDetectionChat:
    """Test warm-up detection in synchronous chat()."""

    @pytest.mark.performance
    def test_fast_first_request_no_warmup(self, MockClient, caplog):
        """Test that fast first request (<10s) does not trigger warm-up notice."""
        mock_instance = MockClient.return_value
        mock_instance.list.return_value = MOCK_MODEL_LIST
        mock_instance.show.return_value = create_mock_show_response()

        # Fast response (< 10s)
        mock_instance.chat.return_value = {
            "message": {"role": "assistant", "content": "Hello"},
            "done": True,
            "prompt_eval_count": 10,
            "eval_count": 5,
        }

        provider = OllamaProvider(model="llama3.1:latest")
        messages = [Message(role=MessageRole.USER, content="Hi")]

        response = provider.chat(messages)

        # Should not log warm-up notice
        assert "loading into memory" not in caplog.text

        # Metadata should not have warm-up info
        assert (
            response.metadata.provider_specific.ollama is None
            or not response.metadata.provider_specific.ollama.warm_up
        )

    @pytest.mark.performance
    def test_slow_first_request_warmup_detected(self, MockClient, caplog):
        """Test that slow first request (≥10s) triggers warm-up notice."""
        mock_instance = MockClient.return_value
        mock_instance.list.return_value = MOCK_MODEL_LIST
        mock_instance.show.return_value = create_mock_show_response()

        # Simulate slow response (10s)
        def slow_chat(**kwargs):
            time.sleep(10.1)  # Just over threshold
            return {
                "message": {"role": "assistant", "content": "Hello"},
                "done": True,
                "prompt_eval_count": 10,
                "eval_count": 5,
            }

        mock_instance.chat.side_effect = slow_chat

        provider = OllamaProvider(model="llama3.1:latest")
        messages = [Message(role=MessageRole.USER, content="Hi")]

        response = provider.chat(messages)

        # Should log warm-up notice
        assert "loading into memory" in caplog.text
        assert "llama3.1:latest" in caplog.text
        assert "Subsequent requests will be faster" in caplog.text

        # Metadata should have warm-up info
        assert response.metadata.provider_specific.ollama is not None
        assert response.metadata.provider_specific.ollama.warm_up is True
        assert response.metadata.provider_specific.ollama.warm_up_duration_seconds
        assert (
            response.metadata.provider_specific.ollama.warm_up_duration_seconds >= 10.0
        )

    @pytest.mark.performance
    def test_second_request_no_warmup_notice(self, MockClient, caplog):
        """Test that second request doesn't trigger warm-up notice even if slow."""
        mock_instance = MockClient.return_value
        mock_instance.list.return_value = MOCK_MODEL_LIST
        mock_instance.show.return_value = create_mock_show_response()

        # First request (fast)
        mock_instance.chat.return_value = {
            "message": {"role": "assistant", "content": "Hello"},
            "done": True,
            "prompt_eval_count": 10,
            "eval_count": 5,
        }

        provider = OllamaProvider(model="llama3.1:latest")
        messages = [Message(role=MessageRole.USER, content="Hi")]

        # First request
        provider.chat(messages)

        # Clear log
        caplog.clear()

        # Second request (simulate it being slow)
        def slow_chat(**kwargs):
            time.sleep(10.1)
            return {
                "message": {"role": "assistant", "content": "World"},
                "done": True,
                "prompt_eval_count": 10,
                "eval_count": 5,
            }

        mock_instance.chat.side_effect = slow_chat
        response2 = provider.chat(messages)

        # Should NOT log warm-up notice (not first request)
        assert "loading into memory" not in caplog.text

        # Metadata should not have warm-up flag
        assert (
            response2.metadata.provider_specific.ollama is None
            or not response2.metadata.provider_specific.ollama.warm_up
        )

    @pytest.mark.performance
    def test_warmup_tracked_per_model_and_host(self, MockClient, caplog):
        """Test that warm-up is tracked separately for different model+host combinations."""
        mock_instance = MockClient.return_value
        mock_instance.list.return_value = MOCK_MODEL_LIST
        mock_instance.show.return_value = create_mock_show_response()

        def slow_chat(**kwargs):
            time.sleep(10.1)
            return {
                "message": {"role": "assistant", "content": "Hello"},
                "done": True,
                "prompt_eval_count": 10,
                "eval_count": 5,
            }

        mock_instance.chat.side_effect = slow_chat

        # First provider with default host
        provider1 = OllamaProvider(model="llama3.1:latest")
        messages = [Message(role=MessageRole.USER, content="Hi")]

        response1 = provider1.chat(messages)
        assert "loading into memory" in caplog.text
        # Metadata should have a warmup flag
        assert (
            response1.metadata.provider_specific.ollama is not None
            and response1.metadata.provider_specific.ollama.warm_up
        )
        caplog.clear()

        # Second provider with different host (should trigger again)
        provider2 = OllamaProvider(
            model="llama3.1:latest", host="http://other-host:11434"
        )
        response2 = provider2.chat(messages)

        # Should log again for different host
        assert "loading into memory" in caplog.text
        # Metadata should have warm-up flag
        assert (
            response2.metadata.provider_specific.ollama is not None
            and response2.metadata.provider_specific.ollama.warm_up
        )


@patch("allos.providers.ollama.Client")
class TestWarmUpDetectionStreaming:
    """Test warm-up detection in streaming mode."""

    @pytest.mark.performance
    def test_stream_fast_first_request_no_warmup(self, MockClient, caplog):
        """Test that fast first streaming request doesn't trigger warm-up."""
        mock_instance = MockClient.return_value
        mock_instance.list.return_value = MOCK_MODEL_LIST
        mock_instance.show.return_value = create_mock_show_response()

        # Fast streaming response
        mock_instance.chat.return_value = iter(
            [
                {"message": {"role": "assistant", "content": "Hello"}, "done": False},
                {
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                    "prompt_eval_count": 10,
                    "eval_count": 5,
                },
            ]
        )

        provider = OllamaProvider(model="llama3.1:latest")
        messages = [Message(role=MessageRole.USER, content="Hi")]

        chunks = list(provider.stream_chat(messages))

        # Should not log warm-up
        assert "loading into memory" not in caplog.text

        # Final metadata should not have warm-up
        metadata_chunks = [c for c in chunks if c.final_metadata]
        assert len(metadata_chunks) == 1
        metadata = metadata_chunks[0].final_metadata
        assert metadata
        assert (
            metadata.provider_specific.ollama is None
            or not metadata.provider_specific.ollama.warm_up
        )

    @pytest.mark.performance
    def test_stream_slow_first_request_warmup_detected(self, MockClient, caplog):
        """Test that slow first streaming request triggers warm-up notice."""
        mock_instance = MockClient.return_value
        mock_instance.list.return_value = MOCK_MODEL_LIST
        mock_instance.show.return_value = create_mock_show_response()

        # Simulate slow streaming response
        def slow_stream(**kwargs):
            time.sleep(10.1)  # Simulate delay
            return iter(
                [
                    {
                        "message": {"role": "assistant", "content": "Hello"},
                        "done": False,
                    },
                    {
                        "message": {"role": "assistant", "content": ""},
                        "done": True,
                        "prompt_eval_count": 10,
                        "eval_count": 5,
                    },
                ]
            )

        mock_instance.chat.side_effect = slow_stream

        provider = OllamaProvider(model="llama3.1:latest")
        messages = [Message(role=MessageRole.USER, content="Hi")]

        chunks = list(provider.stream_chat(messages))

        # Should log warm-up
        assert "loading into memory" in caplog.text

        # Final metadata should have warm-up info
        metadata_chunks = [c for c in chunks if c.final_metadata]
        assert len(metadata_chunks) == 1
        metadata = metadata_chunks[0].final_metadata
        assert metadata
        assert metadata.provider_specific.ollama is not None
        assert metadata.provider_specific.ollama.warm_up is True
        assert metadata.provider_specific.ollama.warm_up_duration_seconds
        assert metadata.provider_specific.ollama.warm_up_duration_seconds >= 10.0

    @pytest.mark.performance
    def test_stream_second_request_no_warmup(self, MockClient, caplog):
        """Test that second streaming request doesn't trigger warm-up."""
        mock_instance = MockClient.return_value
        mock_instance.list.return_value = MOCK_MODEL_LIST
        mock_instance.show.return_value = create_mock_show_response()

        # First request (fast)
        mock_instance.chat.return_value = iter(
            [
                {"message": {"role": "assistant", "content": "First"}, "done": False},
                {
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                    "prompt_eval_count": 10,
                    "eval_count": 5,
                },
            ]
        )

        provider = OllamaProvider(model="llama3.1:latest")
        messages = [Message(role=MessageRole.USER, content="Hi")]

        # First request
        chunks1 = list(provider.stream_chat(messages))
        # Metadata should not have warm-up flag
        metadata_chunks = [c for c in chunks1 if c.final_metadata]
        metadata1 = metadata_chunks[0].final_metadata
        assert metadata1
        assert (
            metadata1.provider_specific.ollama is None
            or not metadata1.provider_specific.ollama.warm_up
        )
        caplog.clear()

        # Second request (slow)
        def slow_stream(**kwargs):
            time.sleep(10.1)
            return iter(
                [
                    {
                        "message": {"role": "assistant", "content": "Second"},
                        "done": False,
                    },
                    {
                        "message": {"role": "assistant", "content": ""},
                        "done": True,
                        "prompt_eval_count": 10,
                        "eval_count": 5,
                    },
                ]
            )

        mock_instance.chat.side_effect = slow_stream
        chunks2 = list(provider.stream_chat(messages))

        # Should NOT log warm-up (not first request)
        assert "loading into memory" not in caplog.text

        # Metadata should not have warm-up flag
        metadata_chunks = [c for c in chunks2 if c.final_metadata]
        metadata = metadata_chunks[0].final_metadata
        assert metadata
        assert (
            metadata.provider_specific.ollama is None
            or not metadata.provider_specific.ollama.warm_up
        )
