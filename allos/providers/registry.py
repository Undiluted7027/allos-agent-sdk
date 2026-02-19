# allos/providers/registry.py

"""A registry for discovering and instantiating LLM providers.

This module uses a decorator-based pattern to allow provider implementations
to register themselves automatically. The ProviderRegistry then acts as a
factory to create provider instances on demand.
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Type

from ..utils.errors import ConfigurationError, ProviderError
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
    "cohere_compat": {
        "env_var": "COHERE_API_KEY",
        "base_url": "https://api.cohere.ai/compatibility/v1",
        "implementation": "chat_completions",
    },
    "portkey": {
        "env_var": "PORTKEY_API_KEY",
        "base_url": "https://api.portkey.ai/v1",
        "provider": "openai",
        "implementation": "chat_completions",
    },
    "zai": {
        "env_var": "ZAI_API_KEY",
        "base_url": "https://api.z.ai/api/paas/v4",
        "provider": "z.ai",
        "implementation": "chat_completions",
    },
    "zenmusk": {
        "env_var": "ZENMUSK_API_KEY",
        "base_url": "https://zenmux.ai/api/v1",
        "provider": "ZenMusk",
        "implementation": "chat_completions",
    },
    "ionet": {
        "env_var": "IOINTELLIGENCE_API_KEY",
        "base_url": "https://api.intelligence.io.solutions/api/v1",
        "provider": "IO.NET",
        "implementation": "chat_completions",
    },
    "nvidia": {
        "env_var": "NVIDIA_API_KEY",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "provider": "Nvidia",
        "implementation": "chat_completions",
    },
    "fastrouter": {
        "env_var": "FASTROUTER_API_KEY",
        "base_url": "https://api.fastrouter.ai/api/v1",
        "provider": "FastRouter",
        "implementation": "chat_completions",
    },
    "iflow": {
        "env_var": "IFLOW_API_KEY",
        "base_url": "https://apis.iflow.cn/v1",
        "provider": "iFlow",
        "implementation": "chat_completions",
    },
    "modelscope": {
        "env_var": "MODELSCOPE_ACCESS_TOKEN",
        "base_url": "https://api-inference.modelscope.ai/v1",
        "provider": "ModelScope",
        "implementation": "chat_completions",
    },
    "llama": {
        "env_var": "LLAMA_API_KEY",
        "base_url": "https://api.llama.com/v1",
        "provider": "Llama",
        "implementation": "chat_completions",
    },
    "inference": {
        "env_var": "CEREBRAS_API_KEY",
        "base_url": "https://api.cerebras.ai/v1",
        "provider": "Cerebras",
        "implementation": "chat_completions",
    },
    "deep_infra": {
        "env_var": "DEEPINFRA_API_KEY",
        "base_url": "https://api.deepinfra.com/v1/openai",
        "provider": "Deep Infra",
        "implementation": "chat_completions",
    },
    "xiaomi": {
        "env_var": "MIMO_API_KEY",
        "base_url": "https://api.xiaomimimo.com/v1",
        "provider": "Xiaomi",
        "implementation": "chat_completions",
    },
    "synthetic": {
        "env_var": "SYNTHETIC_API_KEY",
        "base_url": "https://api.synthetic.new/openai/v1",
        "provider": "Synthetic",
        "implementation": "chat_completions",
    },
    "nebius": {
        "env_var": "NEBIUS_API_KEY",
        "base_url": "https://api.tokenfactory.nebius.com/v1/",
        "provider": "Nebius Token Factory",
        "implementation": "chat_completions",
    },
    "qiniu": {
        "env_var": "QINIU_API_KEY",
        "base_url": "https://api.qnaigc.com/v1",
        "provider": "Qiniu",
        "implementation": "chat_completions",
    },
    "ollama_cloud": {
        "env_var": "OLLAMA_API_KEY",
        "base_url": "https://ollama.com/api/chat/v1",
        "provider": "Ollama Cloud",
        "implementation": "chat_completions",
    },
    "scaleway": {
        "env_var": "SCW_API_KEY",
        "base_url": "https://api.scaleway.ai/v1",
        "provider": "Scaleway",
        "implementation": "chat_completions",
    },
    "cloudflare_ai_gateway": {
        "env_var": "CF_AIG_TOKEN",
        "base_url": "https://gateway.ai.cloudflare.com/v1/{account_id}/{gateway_id}/compat",  # Replace account_id, and gateway_id
        "base_url_env_var": "CF_AIG_URL",
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

        import sys

        # Check Python version compatibility for Google provider
        if name == "google" and sys.version_info < (3, 10):
            raise ProviderError(
                f"Google provider requires Python 3.10 or higher due to "
                f"google-auth dependency. Current version: "
                f"{sys.version_info.major}.{sys.version_info.minor}.\n"
                f"Available providers on Python 3.9: openai, anthropic, ollama, "
                f"chat_completions\n"
                f"Please upgrade to Python 3.10+ or use a different provider.",
                provider="google",
            )

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
                f"Google provider requires Python 3.10+ (current: {sys.version_info[0]}.{sys.version_info[1]})",
            )
        return (False, "Provider not found")
