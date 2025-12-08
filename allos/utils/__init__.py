# allos/utils/__init__.py

"""Provides common, cross-cutting utilities for the Allos SDK.

This package serves as a foundational layer for the entire SDK, containing a
collection of helper modules that address common concerns such as error handling,
secure file operations, logging, and token counting.

By centralizing these utilities, the module promotes code reuse, enforces
consistent behavior (e.g., standardized error types), and enhances the overall
robustness and security of the framework. It is designed to have no circular
dependencies on the higher-level agent, provider, or tool modules.
"""

from .errors import (
    AllosError,
    ConfigurationError,
    FileOperationError,
    PermissionError,
    ProviderError,
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
)
from .file_utils import (
    DEFAULT_MAX_FILE_SIZE,
    is_safe_path,
    safe_read_file,
    safe_write_file,
)
from .logging import logger, setup_logging
from .token_counter import count_tokens, truncate_text_by_tokens

__all__ = [
    # errors
    "AllosError",
    "ConfigurationError",
    "ProviderError",
    "ToolError",
    "ToolNotFoundError",
    "ToolExecutionError",
    "PermissionError",
    "FileOperationError",
    # logging
    "logger",
    "setup_logging",
    # file_utils
    "is_safe_path",
    "safe_read_file",
    "safe_write_file",
    "DEFAULT_MAX_FILE_SIZE",
    # token_counter
    "count_tokens",
    "truncate_text_by_tokens",
]
