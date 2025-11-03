# allos/providers/__init__.py

"""
The `providers` module contains the abstraction layer and concrete implementations
for interacting with various Large Language Model (LLM) providers.
"""

# --- Side-effect imports to register the providers ---
# These imports are for side-effects: they trigger the @provider decorator
# in each file, populating the ProviderRegistry.
from .base import (
    BaseProvider,
    Message,
    MessageRole,
    ProviderResponse,
    ToolCall,
)
from .registry import ProviderRegistry, provider

# We attempt to import each provider module. If the import fails because the
# underlying library (e.g., 'openai', 'anthropic') is not installed, we
# catch the ImportError and simply continue. This allows the core SDK to function
# even if no providers are installed.

try:
    from . import openai  # noqa: F401
except ImportError:
    pass

try:
    from . import anthropic  # noqa: F401
except ImportError:
    pass

__all__ = [
    "BaseProvider",
    "Message",
    "MessageRole",
    "ProviderResponse",
    "ToolCall",
    "ProviderRegistry",
    "provider",
]
