# tests/unit/test_providers.py


import sys
from typing import Tuple
from unittest.mock import patch

import pytest

from allos.providers.base import (
    BaseProvider,
    Message,
    MessageRole,
    ProviderResponse,
)
from allos.providers.metadata import (
    Latency,
    Metadata,
    ModelConfiguration,
    ModelInfo,
    ProviderSpecific,
    QualitySignals,
    SdkInfo,
    ToolInfo,
    Usage,
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
        return ProviderResponse(
            content="dummy response",
            metadata=Metadata(
                status="success",
                model=ModelInfo(
                    provider="mock",
                    model_id="mock-model",
                    configuration=ModelConfiguration(max_output_tokens=100),
                ),
                usage=Usage(),
                latency=Latency(total_duration_ms=100),
                tools=ToolInfo(tools_available=[]),
                quality_signals=QualitySignals(),
                provider_specific=ProviderSpecific(),
                sdk=SdkInfo(sdk_version="test"),
            ),
        )

    def stream_chat(self, messages, **kwargs):
        yield from []  # A simple generator implementation

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

    def test_provider_response_dataclass(self, mock_metadata: Metadata):
        resp = ProviderResponse(content="World", metadata=mock_metadata)
        assert resp.content == "World"
        assert resp.tool_calls == []

    def test_unimplemented_abstract_methods_raise_error(self):
        """Test that calling abstract methods on a non-implemented subclass raises NotImplementedError."""

        class UnimplementedProvider(BaseProvider):
            # This class doesn't implement the abstract methods
            pass

        # Instantiation will fail because abstract methods are not implemented
        with pytest.raises(TypeError):
            provider = UnimplementedProvider(model="test")  # pyright: ignore[reportAbstractUsage]

        # To test the `raise` statement itself, we need a partial implementation
        class PartiallyImplementedProvider(BaseProvider):
            def chat(self, messages, **kwargs):
                return super().chat(  # pyright: ignore[reportAbstractUsage]
                    messages, **kwargs
                )  # pyright: ignore[reportAbstractUsage] # Call the abstract method

            def get_context_window(self) -> int:
                return super().get_context_window()  # type: ignore # Call the abstract method

            def stream_chat(self, messages, **kwargs):
                yield from super().stream_chat(messages, **kwargs)  # type: ignore

        provider = PartiallyImplementedProvider(model="test")
        with pytest.raises(NotImplementedError):
            provider.chat([])
        with pytest.raises(NotImplementedError):
            provider.get_context_window()
        with pytest.raises(NotImplementedError):
            # We must consume the generator to trigger the error
            list(provider.stream_chat([]))

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

        # Register a dummy implementation for 'chat_completions'
        # which is the target for most aliases
        @provider("chat_completions")
        class MockChatCompletions(BaseProvider):
            def chat(self, messages, **kwargs):
                pass

            def get_context_window(self):
                return 100

            def stream_chat(self, messages, **kwargs):
                yield from []  # A simple generator implementation

    def teardown_method(self):
        """Restore the original registry state."""
        _provider_registry.clear()
        _provider_registry.update(self._original_registry)

    def test_get_provider_alias_auto_config(self):
        """Test retrieving a provider via an alias (e.g., 'groq')."""
        # Groq maps to chat_completions implementation
        instance = ProviderRegistry.get_provider(
            "groq", api_key="explicit_key", model="groq-model-123"
        )

        assert isinstance(instance, BaseProvider)
        # Check that base_url was injected from the config map
        assert (
            instance.provider_specific_kwargs["base_url"]
            == "https://api.groq.com/openai/v1"
        )
        # Check that api_key was passed through
        assert instance.provider_specific_kwargs["api_key"] == "explicit_key"

    def test_get_provider_alias_env_var_injection(self, monkeypatch):
        """Test that the registry automatically injects the correct env var for an alias."""
        monkeypatch.setenv("TOGETHER_API_KEY", "env_var_key")

        # Request 'together' without explicit key
        instance = ProviderRegistry.get_provider("together", model="together-model-123")

        # Should have picked up the key from env
        assert instance.provider_specific_kwargs["api_key"] == "env_var_key"
        assert (
            instance.provider_specific_kwargs["base_url"]
            == "https://api.together.xyz/v1"
        )

    def test_get_provider_alias_missing_implementation(self):
        """Test error when an alias points to an unregistered implementation."""
        # Unregister chat_completions to force the error
        del _provider_registry["chat_completions"]

        with pytest.raises(ConfigurationError) as excinfo:
            ProviderRegistry.get_provider("groq")

        assert (
            "implementation 'chat_completions' for alias 'groq' is not registered"
            in str(excinfo.value)
        )

    def test_provider_registration(self, mock_metadata: Metadata):
        """Test that the @provider decorator correctly registers a class."""
        assert "test_provider" not in ProviderRegistry.list_providers()

        @provider("test_provider")
        class TestProvider(BaseProvider):
            def chat(self, messages: list[Message], **kwargs) -> ProviderResponse:
                return ProviderResponse(content="test", metadata=mock_metadata)

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

        # Check for containment instead of exact list equality
        # because the registry now includes dynamic aliases.
        providers = ProviderRegistry.list_providers()
        assert "provider_a" in providers
        assert "provider_b" in providers

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

    def test_init_handles_missing_ollama_library(self, monkeypatch):
        """
        Tests that `allos.providers` can be imported even if 'ollama' is not installed.
        """
        monkeypatch.setitem(sys.modules, "ollama", None)
        self._unload_provider_modules(monkeypatch)

        import allos.providers  # noqa: F401

        registered_providers = ProviderRegistry.list_providers()
        assert "anthropic" in registered_providers
        assert "openai" in registered_providers
        assert "ollama" not in registered_providers

    def test_init_handles_missing_google_library(self, monkeypatch):
        """
        Tests that `allos.providers` can be imported even if 'google' is not installed.
        """
        monkeypatch.setitem(sys.modules, "google", None)
        self._unload_provider_modules(monkeypatch)

        import allos.providers  # noqa: F401

        registered_providers = ProviderRegistry.list_providers(
            include_unavailable=False
        )
        assert "anthropic" in registered_providers
        assert "openai" in registered_providers
        assert "ollama" in registered_providers
        assert "google" not in registered_providers

    def test_init_handles_all_libraries_missing(self, monkeypatch):
        """
        Tests that `allos.providers` can be imported even if all optional provider
        libraries are missing.
        """
        monkeypatch.setitem(sys.modules, "openai", None)
        monkeypatch.setitem(sys.modules, "anthropic", None)
        monkeypatch.setitem(sys.modules, "ollama", None)
        monkeypatch.setitem(sys.modules, "google", None)
        self._unload_provider_modules(monkeypatch)

        import allos.providers  # noqa: F401

        providers = ProviderRegistry.list_providers(include_unavailable=False)
        assert "openai" not in providers
        assert "anthropic" not in providers
        assert "google" not in providers
        # But we expect 'ollama_compat' to be there as it's an alias
        assert "ollama_compat" in providers
        assert "ollama" not in providers

    def test_get_env_var_name_for_unknown_provider_returns_none(self):
        """Test that get_env_var_name returns None for a completely unknown provider."""
        # 'unknown_provider' is not an alias and not in the (cleared) registry
        env_var = ProviderRegistry.get_env_var_name("unknown_provider")
        assert env_var is None


class TestProviderRegistryEnvChecks:
    """Tests for provider environment configuration checks."""

    def setup_method(self):
        self._original_registry = _provider_registry.copy()
        _provider_registry.clear()

    def teardown_method(self):
        """Restore registry."""
        _provider_registry.clear()
        _provider_registry.update(self._original_registry)

    def test_check_provider_env_alias_with_base_url_env_var_set(self, monkeypatch):
        """Test check_provider_env for ollama_compat with OLLAMA_HOST set."""
        monkeypatch.setenv("OLLAMA_HOST", "http://custom:11434")

        is_configured, message = ProviderRegistry.check_provider_env("ollama_compat")

        assert is_configured is True
        assert "OLLAMA_HOST (Set)" in message
        assert "No API key required" in message

    def test_check_provider_env_alias_with_base_url_env_var_default(self, monkeypatch):
        """Test check_provider_env for ollama_compat using default URL."""
        monkeypatch.delenv("OLLAMA_HOST", raising=False)

        is_configured, message = ProviderRegistry.check_provider_env("ollama_compat")

        assert is_configured is True
        assert "Using default" in message

    def test_check_provider_env_alias_with_env_var_set(self, monkeypatch):
        """Test check_provider_env for alias with API key set."""
        monkeypatch.setenv("GROQ_API_KEY", "test-key")

        is_configured, message = ProviderRegistry.check_provider_env("groq")

        assert is_configured is True
        assert "GROQ_API_KEY (Set)" in message

    def test_check_provider_env_alias_with_env_var_missing(self, monkeypatch):
        """Test check_provider_env for alias with API key missing."""
        monkeypatch.delenv("GROQ_API_KEY", raising=False)

        is_configured, message = ProviderRegistry.check_provider_env("groq")

        assert is_configured is False
        assert "GROQ_API_KEY (Not Set)" in message

    def test_check_provider_env_alias_no_env_var_required(self):
        """Test check_provider_env for alias with no env_var field."""
        # Create a test alias with env_var=None but requires_auth=True (default)
        # This is a theoretical edge case - ollama_compat has requires_auth = False
        from allos.providers.registry import OPENAI_COMPATIBLE_PROVIDERS

        # Temporarily add a test alias
        original = OPENAI_COMPATIBLE_PROVIDERS.get("test_alias")
        OPENAI_COMPATIBLE_PROVIDERS["test_alias"] = {
            "env_var": None,
            "base_url": "http://test.url",
            "implementations": "chat_completions",
            # requires_auth defaults to True
        }

        try:
            is_configured, message = ProviderRegistry.check_provider_env("test_alias")
            assert is_configured is True
            assert message == "N/A"
        finally:
            if original:
                OPENAI_COMPATIBLE_PROVIDERS["test_alias"] = original
            else:
                del OPENAI_COMPATIBLE_PROVIDERS["test_alias"]

    def test_check_provider_env_registered_provider(self, monkeypatch):
        """Test check_provider_env delegates to provider's check_env_config."""

        @provider("test_check_provider")
        class TestCheckProvider(BaseProvider):
            env_var = "TEST_API_KEY"

            @classmethod
            def check_env_config(cls) -> Tuple[bool, str]:
                return (True, "Custom check passed")

            def chat(self, messages, **kwargs):
                yield from []

            def get_context_window(self) -> int:
                return 4096

        is_configured, message = ProviderRegistry.check_provider_env(
            "test_check_provider"
        )
        assert is_configured is True
        assert message == "Custom check passed"

    def test_check_provider_env_google_python_version_check(self, monkeypatch):
        """Test check_provider_env for google when Python < 3.10"""
        import sys

        # Only run this test if we can mock the version
        if sys.version_info >= (3, 10):
            # Mock a lower version
            with patch.object(sys, "version_info", (3, 9, 0)):
                # Clear google from history if present
                _provider_registry.pop("google", None)

                is_configured, message = ProviderRegistry.check_provider_env("google")

                assert is_configured is False
                assert "requires Python 3.10+" in message

    def test_check_provider_env_unknown_provider(self):
        """Test check_provider_env for unknown provider."""
        is_configured, message = ProviderRegistry.check_provider_env("nonexistent")

        assert is_configured is False
        assert message == "Provider not found"

    def test_get_provider_ollama_compat_with_env_var(self, monkeypatch):
        """Test get_provider for ollama_compat uses OLLAMA_HOST."""

        # Register chat_completions implementation
        @provider("chat_completions")
        class MockChatCompletions(BaseProvider):
            def chat(self, messages, **kwargs):
                pass

            def stream_chat(self, messages, **kwargs):
                yield from []

            def get_context_window(self) -> int:
                return 4096

        monkeypatch.setenv("OLLAMA_HOST", "http://custom-host:11434")
        instance = ProviderRegistry.get_provider("ollama_compat", model="llama3")

        # Base URL should /v1 appended to OLLAMA_HOST
        assert (
            instance.provider_specific_kwargs["base_url"]
            == "http://custom-host:11434/v1"
        )
        # Should have dummy API key
        assert instance.provider_specific_kwargs["api_key"] == "ollama"

    def test_get_provider_ollama_compat_no_env_var(self, monkeypatch):
        """Test get_provider for ollama_compat uses default URL."""

        @provider("chat_completions")
        class MockChatCompletions(BaseProvider):
            def chat(self, messages, **kwargs):
                pass

            def stream_chat(self, messages, **kwargs):
                yield from []

            def get_context_window(self):
                return 4096

        monkeypatch.delenv("OLLAMA_HOST", raising=False)

        instance = ProviderRegistry.get_provider("ollama_compat", model="llama3")

        assert (
            instance.provider_specific_kwargs["base_url"] == "http://localhost:11434/v1"
        )
        assert instance.provider_specific_kwargs["api_key"] == "ollama"

    def test_list_providers_includes_unavailable(self, monkeypatch):
        """Test list_providers includes unavailable providers like google on Python < 3.10."""
        import sys

        # Clear registry
        _provider_registry.clear()

        # Mock Python 3.9
        with patch.object(sys, "version_info", (3, 9, 0)):
            providers = ProviderRegistry.list_providers(include_unavailable=True)

            # Google should be added even though not registered
            assert "google" in providers

    def test_get_env_var_name_registered_provider(self):
        """Test get_env_var_name for directly registered provider."""

        @provider("test_env_var_provider")
        class TestEnvVarProvider(BaseProvider):
            env_var = "TEST_PROVIDER_KEY"

            def chat(self, messages, **kwargs):
                pass

            def stream_chat(self, messages, **kwargs):
                yield from []

            def get_context_window(self):
                return 4096

        env_var = ProviderRegistry.get_env_var_name("test_env_var_provider")

        assert env_var == "TEST_PROVIDER_KEY"

    def test_get_env_var_name_registered_provider_no_env_var(self):
        """Test get_env_var_name when provider has no env_var attribute."""

        @provider("test_no_env_provider")
        class TestNoEnvProvider(BaseProvider):
            # No env_var attribute
            def chat(self, messages, **kwargs):
                pass

            def stream_chat(self, messages, **kwargs):
                yield from []

            def get_context_window(self):
                return 4096

        env_var = ProviderRegistry.get_env_var_name("test_no_env_provider")

        assert env_var is None

    def test_get_env_var_name_for_openai_completions_compatible_provider(self):
        """Test get_env_var_name for OpenAI-compatible alias providers."""

        # Test with a known OpenAI-compatible provider
        env_var = ProviderRegistry.get_env_var_name("groq")

        assert env_var == "GROQ_API_KEY"

        # Test with another alias
        env_var = ProviderRegistry.get_env_var_name("together")
        assert env_var == "TOGETHER_API_KEY"

    def test_check_env_config_no_env_var_required(self):
        """Test check_env_config when provider has no env_var requirement."""

        @provider("test_no_env_var_provider")
        class TestNoEnvVarProvider(BaseProvider):
            env_var = None  # No environment variable required

            def chat(self, messages, **kwargs):
                pass

            def stream_chat(self, messages, **kwargs):
                yield from []

            def get_context_window(self) -> int:
                return 4096

        is_configured, message = TestNoEnvVarProvider.check_env_config()

        assert is_configured is True
        assert message == "N/A"


class TestDetermineModelGoogle:
    """Tests for determine_model with Google provider."""

    def test_determine_model_google_default(self):
        """Test that Google provider has a default model."""
        from allos.cli.utils import determine_model

        result = determine_model("google", None)
        assert result == "gemini-2.5-flash-lite"

    def test_determine_model_google_explicit(self):
        """Test that explicit model overrides default for Google."""
        from allos.cli.utils import determine_model

        result = determine_model("google", "gemini-2.0-flash")
        assert result == "gemini-2.0-flash"


class TestValidateApiKeyOllamaCompat:
    """Tests for validate_api_key with ollama_compat provider."""

    def test_validate_api_key_ollama_compat_no_auth_required(self):
        """Test that ollama_compat doesn't require API key."""
        from allos.cli.utils import validate_api_key

        result, missing_var = validate_api_key("ollama_compat", None)

        assert result is True
        assert missing_var == ""
