# allos/providers/__init__.py

"""The foundation of the Allos SDK's LLM-agnostic architecture.

This module provides the core abstraction layer that allows the `Agent` to
interact with any supported Large Language Model (LLM) through a single,
standardized interface. This decouples the agent's core logic from the specific
implementation details of any given LLM provider.

The architecture is designed to be pluggable and extensible:
 - `BaseProvider`: An abstract class that defines the contract all provider
   implementations must follow (e.g., `chat`, `stream_chat`).
 - `ProviderRegistry`: A factory class that discovers and instantiates providers.
 - `@provider` decorator: A simple mechanism for new provider classes to
   register themselves with the registry automatically.
 - Standardized Data Structures: Classes like `Message`, `ToolCall`, and
   `ProviderResponse` ensure a consistent data format across all providers.

A key design feature of this module is its handling of optional dependencies.
Provider implementations (like `openai`, `anthropic`) are imported within
`try...except` blocks. This allows the core Allos SDK to function even if the
underlying third-party libraries (e.g., `openai`, `anthropic`) are not
installed, preventing dependency errors and making the core framework lightweight.
"""

# --- Side-effect imports to register the providers ---
# These imports are for side-effects: they trigger the @provider decorator
# in each file, populating the ProviderRegistry.
from ..utils.logging import logger
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
# We also catch AttributeError to handle cases where the module exists in
# sys.modules but is set to None (common in testing scenarios).

try:
    from . import openai  # noqa: F401
except (ImportError, AttributeError):
    logger.debug("Skipped optional provider: openai")

try:
    from . import anthropic  # noqa: F401
except (ImportError, AttributeError):
    logger.debug("Skipped optional provider: anthropic")


try:
    from . import chat_completions  # noqa: F401
except (ImportError, AttributeError):
    logger.debug("Skipped optional provider: chat_completions")

__all__ = [
    "BaseProvider",
    "Message",
    "MessageRole",
    "ProviderResponse",
    "ToolCall",
    "ProviderRegistry",
    "provider",
]
