# allos/providers/__init__.py

"""
The `providers` module contains the abstraction layer and concrete implementations
for interacting with various Large Language Model (LLM) providers.
"""

# --- Side-effect imports to register the providers ---
# These imports are for side-effects: they trigger the @provider decorator
# in each file, populating the ProviderRegistry.
from . import anthropic, openai  # noqa: F401
from .base import (
    BaseProvider,
    Message,
    MessageRole,
    ProviderResponse,
    ToolCall,
)
from .registry import ProviderRegistry, provider

__all__ = [
    "BaseProvider",
    "Message",
    "MessageRole",
    "ProviderResponse",
    "ToolCall",
    "ProviderRegistry",
    "provider",
]
