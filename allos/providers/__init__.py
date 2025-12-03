# allos/providers/__init__.py

"""
The `providers` module contains the abstraction layer and concrete implementations
for interacting with various Large Language Model (LLM) providers.
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
