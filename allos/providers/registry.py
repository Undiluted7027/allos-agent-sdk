# allos/providers/registry.py

"""
A registry for discovering and instantiating LLM providers.

This module uses a decorator-based pattern to allow provider implementations
to register themselves automatically. The ProviderRegistry then acts as a
factory to create provider instances on demand.
"""

from typing import Dict, List, Type

from ..utils.errors import ConfigurationError
from .base import BaseProvider

# The global registry dictionary mapping provider names to their classes
_provider_registry: Dict[str, Type[BaseProvider]] = {}


def provider(name: str):
    """
    A decorator to register a new provider class.

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
    """
    A factory class for creating LLM provider instances.
    """

    @classmethod
    def get_provider(cls, name: str, **kwargs) -> BaseProvider:
        """
        Get an instance of a registered provider.

        Args:
            name: The name of the provider (e.g., "openai").
            **kwargs: Arguments to pass to the provider's constructor (e.g., model).

        Returns:
            An instance of the requested provider.

        Raises:
            ConfigurationError: If the provider is not registered.
        """
        if name not in _provider_registry:
            raise ConfigurationError(
                f"Provider '{name}' not found. "
                f"Available providers: {', '.join(cls.list_providers())}"
            )
        provider_class = _provider_registry[name]
        return provider_class(**kwargs)

    @classmethod
    def list_providers(cls) -> List[str]:
        """
        List the names of all registered providers.
        """
        return sorted(_provider_registry.keys())
