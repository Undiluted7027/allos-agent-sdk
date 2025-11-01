# tests/unit/test_providers.py


import pytest

from allos.providers.base import (
    BaseProvider,
    Message,
    MessageRole,
    ProviderResponse,
)
from allos.providers.registry import ProviderRegistry, _provider_registry, provider
from allos.utils.errors import ConfigurationError


# A dummy provider for testing registration.
# Note: The @provider decorator is removed from here and applied inside tests
# to improve test isolation and avoid import-time side effects.
class DummyProvider(BaseProvider):
    def __init__(self, model: str, api_key: str = "dummy_key"):
        super().__init__(model=model, api_key=api_key)

    def chat(self, messages: list[Message], **kwargs) -> ProviderResponse:
        return ProviderResponse(content="dummy response")

    def get_context_window(self) -> int:
        """A dummy implementation for the abstract method."""
        return 4096  # Return a default value


class TestProviderBase:
    """Tests for the base provider data structures."""

    def test_message_role_enum(self):
        assert MessageRole.USER == "user"
        assert MessageRole.SYSTEM == "system"

    def test_message_dataclass(self):
        msg = Message(role=MessageRole.USER, content="Hello")
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"
        assert msg.tool_calls == []

    def test_provider_response_dataclass(self):
        resp = ProviderResponse(content="World")
        assert resp.content == "World"
        assert resp.tool_calls == []


class TestProviderRegistry:
    """Tests for the provider registry and factory."""

    def setup_method(self):
        """Ensure a clean registry for each test method."""
        # This is a bit of a hack to reset the global registry for tests
        # In a real app, registration happens once at import time.
        self._original_registry = _provider_registry.copy()
        _provider_registry.clear()

    def teardown_method(self):
        """Restore the original registry state."""
        _provider_registry.clear()
        _provider_registry.update(self._original_registry)

    def test_provider_registration(self):
        """Test that the @provider decorator correctly registers a class."""
        assert "test_provider" not in ProviderRegistry.list_providers()

        @provider("test_provider")
        class TestProvider(BaseProvider):
            def chat(self, messages: list[Message], **kwargs) -> ProviderResponse:
                return ProviderResponse(content="test")

        assert "test_provider" in ProviderRegistry.list_providers()

    def test_registration_fails_for_non_provider_class(self):
        """
        Test that the @provider decorator raises a TypeError if the decorated
        class is not a subclass of BaseProvider.
        """
        with pytest.raises(TypeError) as excinfo:

            @provider("not_a_provider")  # type: ignore
            class NotAProvider:
                pass  # This class does not inherit from BaseProvider

        assert "Registered class must be a subclass of BaseProvider" in str(
            excinfo.value
        )

    def test_list_providers(self):
        """Test listing of registered providers."""

        @provider("provider_a")
        class ProviderA(BaseProvider):
            def chat(self, messages: list[Message], **kwargs):
                pass

        @provider("provider_b")
        class ProviderB(BaseProvider):
            def chat(self, messages: list[Message], **kwargs):
                pass

        assert sorted(ProviderRegistry.list_providers()) == ["provider_a", "provider_b"]

    def test_get_provider_success(self):
        """Test successfully getting a provider instance."""
        # Register the DummyProvider for this test
        provider("dummy_for_test")(DummyProvider)

        instance = ProviderRegistry.get_provider(
            "dummy_for_test", model="dummy-model", api_key="123"
        )
        assert isinstance(instance, DummyProvider)
        assert instance.model == "dummy-model"
        assert instance.provider_specific_kwargs["api_key"] == "123"

    def test_get_provider_not_found(self):
        """Test that getting a non-existent provider raises an error."""
        with pytest.raises(ConfigurationError) as excinfo:
            ProviderRegistry.get_provider("non_existent_provider")
        assert "Provider 'non_existent_provider' not found" in str(excinfo.value)

    def test_duplicate_registration_raises_error(self):
        """Test that registering a provider with the same name twice fails."""

        @provider("duplicate_name")
        class Provider1(BaseProvider):
            def chat(self, messages: list[Message], **kwargs):
                pass

        with pytest.raises(ValueError) as excinfo:

            @provider("duplicate_name")
            class Provider2(BaseProvider):
                def chat(self, messages: list[Message], **kwargs):
                    pass

        assert "Provider 'duplicate_name' is already registered" in str(excinfo.value)
