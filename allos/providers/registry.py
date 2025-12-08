# allos/providers/registry.py

"""A registry for discovering and instantiating LLM providers.

This module uses a decorator-based pattern to allow provider implementations
to register themselves automatically. The ProviderRegistry then acts as a
factory to create provider instances on demand.
"""

import os
from typing import Dict, List, Optional, Type

from ..utils.errors import ConfigurationError
from .base import BaseProvider

# The global registry dictionary mapping provider names to their classes
_provider_registry: Dict[str, Type[BaseProvider]] = {}

# Configuration for known OpenAI-compatible providers
OPENAI_COMPATIBLE_PROVIDERS: Dict[str, Dict[str, Optional[str]]] = {
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
        "implementation": "chat_completions",
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
            if "base_url" not in kwargs or kwargs["base_url"] is None:
                config_overrides["base_url"] = config["base_url"]

            # Auto-detect API key if not provided
            if "api_key" not in kwargs or kwargs["api_key"] is None:
                env_var = config["env_var"]
                if env_var and env_var in os.environ:
                    config_overrides["api_key"] = os.environ[env_var]

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
    def list_providers(cls) -> List[str]:
        """List the names of all registered providers AND known aliases."""
        direct = list(_provider_registry.keys())
        aliases = list(OPENAI_COMPATIBLE_PROVIDERS.keys())
        return sorted(set(direct + aliases))

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
