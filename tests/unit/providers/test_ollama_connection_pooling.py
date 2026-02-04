"""Tests for Ollama provider connection pooling functionality."""

import threading
from unittest.mock import MagicMock, patch

import pytest

from allos.providers.ollama import _OLLAMA_CLIENT_POOL, OllamaProvider


@pytest.fixture(autouse=True)
def clear_connection_pool():
    """Clear the connection pool before and after each test."""
    # Clear before test
    _OLLAMA_CLIENT_POOL.clear()
    yield
    # Clear after test
    _OLLAMA_CLIENT_POOL.clear()


class TestOllamaConnectionPooling:
    """Test suite for Ollama connection pooling."""

    @patch("allos.providers.ollama.Client")
    def test_connection_pool_reuses_client_same_host(self, mock_client_class):
        """Test that multiple providers with same host reuse the same client."""
        # Setup mock with proper response structure
        mock_model = MagicMock()
        mock_model.model = "llama3"

        mock_list_response = MagicMock()
        mock_list_response.models = [mock_model]

        mock_show_response = MagicMock()
        mock_show_response.modelinfo = None

        mock_client = MagicMock()
        mock_client.list.return_value = mock_list_response
        mock_client.show.return_value = mock_show_response
        mock_client_class.return_value = mock_client

        # Create two providers with same host (default)
        provider1 = OllamaProvider("llama3")
        provider2 = OllamaProvider("llama3")

        # Should only create one client
        assert mock_client_class.call_count == 1
        # Both providers should share the same client instance
        assert provider1.client is provider2.client  # pyright: ignore[reportAttributeAccessIssue]

    @patch("allos.providers.ollama.Client")
    def test_connection_pool_creates_separate_clients_different_hosts(
        self, mock_client_class
    ):
        """Test that providers with different hosts get separate clients."""
        # Setup mock with proper response structure
        mock_model = MagicMock()
        mock_model.model = "llama3"

        mock_list_response = MagicMock()
        mock_list_response.models = [mock_model]

        mock_show_response = MagicMock()
        mock_show_response.modelinfo = None

        # Return different client instances for different calls
        mock_client1 = MagicMock()
        mock_client1.list.return_value = mock_list_response
        mock_client1.show.return_value = mock_show_response

        mock_client2 = MagicMock()
        mock_client2.list.return_value = mock_list_response
        mock_client2.show.return_value = mock_show_response

        mock_client_class.side_effect = [mock_client1, mock_client2]

        # Create providers with different hosts
        provider1 = OllamaProvider("llama3", host="http://localhost:11434")
        provider2 = OllamaProvider("llama3", host="http://localhost:11435")

        # Should create two separate clients
        assert mock_client_class.call_count == 2
        # Providers should have different client instances
        assert provider1.client is not provider2.client  # pyright: ignore[reportAttributeAccessIssue]

    @patch("allos.providers.ollama.Client")
    def test_connection_pool_reuses_client_explicit_host(self, mock_client_class):
        """Test that providers with same explicit host reuse client."""
        # Setup mock with proper response structure - include all 3 models
        mock_model1 = MagicMock()
        mock_model1.model = "llama3"
        mock_model2 = MagicMock()
        mock_model2.model = "qwen2"
        mock_model3 = MagicMock()
        mock_model3.model = "llama3:7b"

        mock_list_response = MagicMock()
        mock_list_response.models = [mock_model1, mock_model2, mock_model3]

        mock_show_response = MagicMock()
        mock_show_response.modelinfo = None

        mock_client = MagicMock()
        mock_client.list.return_value = mock_list_response
        mock_client.show.return_value = mock_show_response
        mock_client_class.return_value = mock_client

        # Create three providers with same explicit host
        host = "http://localhost:11434"
        provider1 = OllamaProvider("llama3", host=host)
        provider2 = OllamaProvider("qwen2", host=host)
        provider3 = OllamaProvider("llama3:7b", host=host)

        # Should only create one client
        assert mock_client_class.call_count == 1
        # All providers should share the same client instance
        assert provider1.client is provider2.client  # pyright: ignore[reportAttributeAccessIssue]
        assert provider2.client is provider3.client  # pyright: ignore[reportAttributeAccessIssue]

    @patch("allos.providers.ollama.Client")
    def test_connection_pool_thread_safety(self, mock_client_class):
        """Test that connection pool is thread-safe."""
        # Setup mock with proper response structure
        mock_model = MagicMock()
        mock_model.model = "llama3"

        mock_list_response = MagicMock()
        mock_list_response.models = [mock_model]

        mock_show_response = MagicMock()
        mock_show_response.modelinfo = None

        mock_client = MagicMock()
        mock_client.list.return_value = mock_list_response
        mock_client.show.return_value = mock_show_response
        mock_client_class.return_value = mock_client

        providers = []
        errors = []

        def create_provider():
            try:
                provider = OllamaProvider("llama3")
                providers.append(provider)
            except Exception as e:
                errors.append(e)

        # Create 10 providers concurrently
        threads = [threading.Thread(target=create_provider) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No errors should occur
        assert len(errors) == 0
        # All providers should be created
        assert len(providers) == 10
        # Client should only be created once despite concurrent access
        assert mock_client_class.call_count == 1
        # All providers should share the same client
        first_client = providers[0].client
        assert all(p.client is first_client for p in providers)

    @patch("allos.providers.ollama.Client")
    def test_connection_pool_different_models_same_client(self, mock_client_class):
        """Test that different models on same host share client."""
        # Setup mock
        mock_client = MagicMock()
        mock_model1 = MagicMock()
        mock_model1.model = "llama3"
        mock_model2 = MagicMock()
        mock_model2.model = "qwen2"
        mock_model3 = MagicMock()
        mock_model3.model = "mistral"

        mock_list_response = MagicMock()
        mock_list_response.models = [mock_model1, mock_model2, mock_model3]
        mock_client.list.return_value = mock_list_response
        mock_client.show.return_value = {
            "model_info": {},
            "details": {"parameter_size": "7B"},
        }
        mock_client_class.return_value = mock_client

        # Create providers for different models
        provider_llama = OllamaProvider("llama3")
        provider_qwen = OllamaProvider("qwen2")
        provider_mistral = OllamaProvider("mistral")

        # Should only create one client
        assert mock_client_class.call_count == 1
        # All providers should share the same client
        assert provider_llama.client is provider_qwen.client  # pyright: ignore[reportAttributeAccessIssue]
        assert provider_qwen.client is provider_mistral.client  # pyright: ignore[reportAttributeAccessIssue]

    @patch("allos.providers.ollama.Client")
    def test_connection_pool_preserves_model_specific_state(self, mock_client_class):
        """Test that each provider maintains its own model-specific state."""
        # Setup mock
        mock_client = MagicMock()
        mock_model1 = MagicMock()
        mock_model1.model = "llama3"
        mock_model2 = MagicMock()
        mock_model2.model = "qwen2"

        mock_list_response = MagicMock()
        mock_list_response.models = [mock_model1, mock_model2]
        mock_client.list.return_value = mock_list_response
        mock_client.show.return_value = {
            "model_info": {},
            "details": {"parameter_size": "7B"},
        }
        mock_client_class.return_value = mock_client

        # Create providers for different models
        provider1 = OllamaProvider("llama3")
        provider2 = OllamaProvider("qwen2")

        # Shared client
        assert provider1.client is provider2.client  # pyright: ignore[reportAttributeAccessIssue]

        # But different model names
        assert provider1.model == "llama3"
        assert provider2.model == "qwen2"

        # And can have different retry configs
        provider3 = OllamaProvider("llama3", max_stream_retries=5)
        assert provider3._max_stream_retries == 5  # pyright: ignore[reportAttributeAccessIssue]
        assert provider1._max_stream_retries == 3  # pyright: ignore[reportAttributeAccessIssue] # default

    @patch("allos.providers.ollama.Client")
    @patch.dict("os.environ", {"OLLAMA_HOST": "http://env-host:11434"})
    def test_connection_pool_respects_env_var(self, mock_client_class):
        """Test that connection pool respects OLLAMA_HOST environment variable."""
        # Setup mock with proper response structure - include both models
        mock_model1 = MagicMock()
        mock_model1.model = "llama3"
        mock_model2 = MagicMock()
        mock_model2.model = "qwen2"

        mock_list_response = MagicMock()
        mock_list_response.models = [mock_model1, mock_model2]

        mock_show_response = MagicMock()
        mock_show_response.modelinfo = None

        mock_client = MagicMock()
        mock_client.list.return_value = mock_list_response
        mock_client.show.return_value = mock_show_response
        mock_client_class.return_value = mock_client

        # Create provider without explicit host (should use env var)
        provider1 = OllamaProvider("llama3")

        # Create another provider with same env host
        provider2 = OllamaProvider("qwen2")

        # Should only create one client (same host from env)
        assert mock_client_class.call_count == 1
        assert provider1.client is provider2.client  # pyright: ignore[reportAttributeAccessIssue]

        # Verify client was created with env host
        mock_client_class.assert_called_with(host="http://env-host:11434")
