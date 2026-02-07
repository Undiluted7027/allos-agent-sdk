# allos/providers/registry.py

"""A registry for discovering and instantiating LLM providers.

This module uses a decorator-based pattern to allow provider implementations
to register themselves automatically. The ProviderRegistry then acts as a
factory to create provider instances on demand.
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Type

from ..utils.errors import ConfigurationError
from .base import BaseProvider

# The global registry dictionary mapping provider names to their classes
_provider_registry: Dict[str, Type[BaseProvider]] = {}

# Configuration for known OpenAI-compatible providers
OPENAI_COMPATIBLE_PROVIDERS: Dict[str, Dict[str, Any]] = {
    "together": {
        "env_var": "TOGETHER_API_KEY",
        "base_url": "https://api.together.xyz/v1",
        "implementation": "chat_completions",
    },
    "groq": {
        "env_var": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "implementation": "chat_completions",
    },
    "mistral": {
        "env_var": "MISTRAL_API_KEY",
        "base_url": "https://api.mistral.ai/v1",
        "implementation": "chat_completions",
    },
    "deepseek": {
        "env_var": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "implementation": "chat_completions",
    },
    "openrouter": {
        "env_var": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "implementation": "chat_completions",
    },
    "cohere": {
        "env_var": "COHERE_API_KEY",
        "base_url": "https://api.cohere.ai/compatibility/v1",
        "implementation": "chat_completions",
    },
    "portkey": {
        "env_var": "PORTKEY_API_KEY",
        "base_url": "https://api.portkey.ai/v1/",
        "provider": "openai",
        "implementation": "chat_completions",
    },
    # Ollama is unique: it can be used via compatibility mode (chat_completions)
    # OR via its native library (future implementation).
    "ollama_compat": {
        "env_var": None,
        "base_url": "http://localhost:11434/v1",
        "base_url_env_var": "OLLAMA_HOST",  # Override base URL from this
        "implementation": "chat_completions",
        "requires_auth": False,
    },
}


def provider(name: str):
    """A decorator to register a new provider class.

    Usage:
        @provider("openai")
        class OpenAIProvider(BaseProvider):
            ...
    """

    def decorator(cls: Type[BaseProvider]) -> Type[BaseProvider]:
        if not issubclass(cls, BaseProvider):
            raise TypeError("Registered class must be a subclass of BaseProvider.")
        if name in _provider_registry:
            raise ValueError(f"Provider '{name}' is already registered.")
        _provider_registry[name] = cls
        return cls

    return decorator


class ProviderRegistry:
    """A factory class for creating LLM provider instances."""

    @classmethod
    def get_provider(cls, name: str, **kwargs) -> BaseProvider:
        """Get an instance of a registered provider.

        Intelligently handles known OpenAI-compatible providers (like 'together', 'groq')
        by configuring the generic 'chat_completions' provider automatically.

        Args:
            name: The name of the provider (e.g., "openai", "together").
            **kwargs: Arguments to pass to the provider's constructor (e.g., model).

        Returns:
            An instance of the requested provider.

        Raises:
            ConfigurationError: If the provider is not registered or an alias's implementation is missing.
        """
        implementation_class = None
        config_overrides = {}

        # Check if it's a known OpenAI-compatible provider alias
        if name in OPENAI_COMPATIBLE_PROVIDERS:
            config = OPENAI_COMPATIBLE_PROVIDERS[name]
            implementation_name = config["implementation"]

            if implementation_name not in _provider_registry:
                raise ConfigurationError(
                    f"The implementation '{implementation_name}' for alias '{name}' is not registered."
                )
            implementation_class = _provider_registry[implementation_name]

            # Apply default from the alias config
            # Only set base_url if not already provided by user
            # Check env var first, then apply default
            if "base_url" not in kwargs or kwargs["base_url"] is None:
                base_url_env_var = config.get("base_url_env_var")
                if base_url_env_var and base_url_env_var in os.environ:
                    # OLLAMA_HOST does not include /v1 for
                    # Chat Completions, so append it
                    host = os.environ[base_url_env_var].rstrip("/")
                    config_overrides["base_url"] = f"{host}/v1"
                else:
                    config_overrides["base_url"] = config["base_url"]

            # Auto-detect API key if not provided
            if "api_key" not in kwargs or kwargs["api_key"] is None:
                env_var = config.get("env_var")
                requires_auth = config.get("requires_auth", True)  # Default True

                if env_var and env_var in os.environ:
                    config_overrides["api_key"] = os.environ[env_var]
                elif not requires_auth:
                    # Provider doesn't need auth, but OpenAI client requires a value
                    config_overrides["api_key"] = "ollama"  # Dummy value

        # Check if it's a directly registered provider
        elif name in _provider_registry:
            implementation_class = _provider_registry[name]

        else:
            raise ConfigurationError(
                f"Provider '{name}' not found. "
                f"Available: {', '.join(cls.list_providers())}"
            )
        # Merge overrides with user arguments (user args take precedence where applicable,
        # but we already handled the logic for defaults above)
        final_kwargs = {**kwargs, **config_overrides}

        return implementation_class(**final_kwargs)

    @classmethod
    def list_providers(cls, include_unavailable: bool = True) -> List[str]:
        """List the names of all registered providers AND known aliases.

        Args:
            include_unavailable: If True, includes providers that may not be
                                available due to Python version requirements.
        """
        direct = list(_provider_registry.keys())
        aliases = list(OPENAI_COMPATIBLE_PROVIDERS.keys())
        all_providers = sorted(set(direct + aliases))
        if include_unavailable:
            # Add known providers that might not be registered due to version
            import sys

            if "google" not in all_providers and sys.version_info < (3, 10):
                all_providers.append("google")
            all_providers = sorted(set(all_providers))

        return all_providers

    @classmethod
    def get_env_var_name(cls, provider_name: str) -> Optional[str]:
        """Dynamically gets the expected env var name for a provider.

        It checks aliases first, then the registered provider class itself.
        """
        # 1. Check if it's an alias with a specific env var
        if provider_name in OPENAI_COMPATIBLE_PROVIDERS:
            return OPENAI_COMPATIBLE_PROVIDERS[provider_name].get("env_var")

        # 2. Check if it's a natively registered provider
        if provider_name in _provider_registry:
            provider_class = _provider_registry[provider_name]
            return getattr(provider_class, "env_var", None)

        # Return None if no specific variable is found.
        return None

    @classmethod
    def check_provider_env(cls, provider_name: str) -> Tuple[bool, str]:
        """Check environment configuration for a provider.

        This method delegates to the provider's check_env_config classmethod,
        which allows each provider to implement its own validation logic.

        Args:
            provider_name: Name of the provider (e.g., "anthropic", "openai")

        Returns:
            Tuple of (is_configured, display_message)
        """
        import sys

        # Check aliases first
        if provider_name in OPENAI_COMPATIBLE_PROVIDERS:
            config = OPENAI_COMPATIBLE_PROVIDERS[provider_name]
            env_var = config.get("env_var")
            requires_auth = config.get("requires_auth", True)
            base_url_env_var = config.get("base_url_env_var")

            # Build status message
            parts = []

            # Check base URL env var
            if base_url_env_var:
                if base_url_env_var in os.environ:
                    parts.append(f"{base_url_env_var} (Set)")
                else:
                    parts.append(f"{base_url_env_var} (Using default)")

            # Check auth requirement
            if not requires_auth:
                parts.append("No API key required")
                return (True, ", ".join(parts) if parts else "Ready")

            if env_var is None:
                return (True, ", ".join(parts) if parts else "N/A")

            if env_var in os.environ:
                parts.append(f"{env_var} (Set)")
                return (True, ", ".join(parts))

            parts.append(f"{env_var} (Not Set)")
            return (False, ", ".join(parts))

        # Check registered providers
        if provider_name in _provider_registry:
            provider_class = _provider_registry[provider_name]
            return provider_class.check_env_config()
        # Provider not registered - check if it's a known provider with version requirements
        if provider_name == "google" and sys.version_info < (3, 10):
            return (
                False,
                f"Requires Python 3.10+ (current: {sys.version_info[0]}.{sys.version_info[1]})",
            )
        return (False, "Provider not found")
