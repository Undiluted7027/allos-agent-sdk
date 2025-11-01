# allos/utils/__init__.py

"""
The `utils` module provides common utilities for the Allos SDK,
including error handling, logging, secure file operations, and token counting.
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
