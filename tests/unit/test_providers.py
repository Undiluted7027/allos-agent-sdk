# tests/unit/test_providers.py


import sys

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

    def test_unimplemented_abstract_methods_raise_error(self):
        """Test that calling abstract methods on a non-implemented subclass raises NotImplementedError."""

        class UnimplementedProvider(BaseProvider):
            # This class doesn't implement the abstract methods
            pass

        # Instantiation will fail because abstract methods are not implemented
        with pytest.raises(TypeError):
            provider = UnimplementedProvider(
                model="test"
            )  # pyright: ignore[reportAbstractUsage]

        # To test the `raise` statement itself, we need a partial implementation
        class PartiallyImplementedProvider(BaseProvider):
            def chat(self, messages, **kwargs):
                return super().chat(  # pyright: ignore[reportAbstractUsage]
                    messages, **kwargs
                )  # pyright: ignore[reportAbstractUsage] # Call the abstract method

            def get_context_window(self) -> int:
                return super().get_context_window()  # type: ignore # Call the abstract method

        provider = PartiallyImplementedProvider(model="test")
        with pytest.raises(NotImplementedError):
            provider.chat([])
        with pytest.raises(NotImplementedError):
            provider.get_context_window()

    def test_base_provider_repr(self):
        """Test the __repr__ string representation of the provider."""
        # We can use our DummyProvider for this test
        provider = DummyProvider(model="dummy-model-123")
        representation = repr(provider)
        assert representation == "DummyProvider(model='dummy-model-123')"


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


class TestProviderInit:
    """
    Tests the provider __init__.py import logic to ensure it gracefully handles
    missing optional dependencies.
    """

    def setup_method(self):
        """Save the original registry state and clear it for an isolated test run."""
        self._original_registry = _provider_registry.copy()
        # Save only the modules we are about to manipulate
        self._original_sys_modules = {
            name: mod
            for name, mod in sys.modules.items()
            if name.startswith("allos.providers")
        }
        _provider_registry.clear()

    def teardown_method(self):
        """Restore the original registry and sys.modules state to ensure test isolation."""
        _provider_registry.clear()
        _provider_registry.update(self._original_registry)
        # Restore sys.modules to its pre-test state
        for name, mod in self._original_sys_modules.items():
            if mod is not None:
                sys.modules[name] = mod
        # Re-import the main module to ensure its state is restored for other test files
        import importlib

        importlib.import_module("allos.providers")

    def _unload_provider_modules(self, monkeypatch):
        """Helper to remove all provider-related modules from the import cache."""
        modules_to_unload = [
            m
            for m in sys.modules
            if m.startswith("allos.providers")
            and m
            not in {
                "allos.providers.base",
                "allos.providers.registry",
            }
        ]
        for module_name in modules_to_unload:
            monkeypatch.delitem(sys.modules, module_name, raising=False)

    def test_init_handles_missing_openai_library(self, monkeypatch):
        """
        Tests that `allos.providers` can be imported even if 'openai' is not installed.
        """
        # 1. Simulate the 'openai' package not being installed.
        monkeypatch.setitem(sys.modules, "openai", None)

        # 2. Unload all provider-related modules from the cache. This is the crucial step.
        self._unload_provider_modules(monkeypatch)

        # 3. Re-import the module to trigger the try/except registration logic.
        import allos.providers  # noqa: F401

        # 4. Assert that the openai provider is missing, but anthropic was registered.
        registered_providers = ProviderRegistry.list_providers()
        assert "openai" not in registered_providers
        assert "anthropic" in registered_providers

    def test_init_handles_missing_anthropic_library(self, monkeypatch):
        """
        Tests that `allos.providers` can be imported even if 'anthropic' is not installed.
        """
        monkeypatch.setitem(sys.modules, "anthropic", None)
        self._unload_provider_modules(monkeypatch)

        import allos.providers  # noqa: F401

        registered_providers = ProviderRegistry.list_providers()
        assert "anthropic" not in registered_providers
        assert "openai" in registered_providers

    def test_init_handles_all_libraries_missing(self, monkeypatch):
        """
        Tests that `allos.providers` can be imported even if all optional provider
        libraries are missing.
        """
        monkeypatch.setitem(sys.modules, "openai", None)
        monkeypatch.setitem(sys.modules, "anthropic", None)
        self._unload_provider_modules(monkeypatch)

        import allos.providers  # noqa: F401

        assert ProviderRegistry.list_providers() == []
