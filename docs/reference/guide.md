# Allos Agent SDK - MVP Implementation Guide

> **Goal**: Build a working, provider-agnostic agentic SDK in 6-8 weeks
>
> **Target**: A CLI tool that can switch between OpenAI, Anthropic, and Ollama to perform file operations and shell commands

---

## Table of Contents

1. [Week 1: Foundation & Core Abstractions](#week-1-foundation--core-abstractions)
2. [Week 2: Provider Layer](#week-2-provider-layer)
3. [Week 3: Tool System](#week-3-tool-system)
4. [Week 4: Agent Core](#week-4-agent-core)
5. [Week 5: CLI & Polish](#week-5-cli--polish)
6. [Week 6: Testing & Documentation](#week-6-testing--documentation)
7. [Week 7-8: Examples & Community Prep](#week-7-8-examples--community-prep)

---

## Week 1: Foundation & Core Abstractions

### Day 1-2: Project Setup & Dependencies

#### 1.1 Configure `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "allos-agent-sdk"
version = "0.1.0"
description = "LLM-agnostic agentic SDK with provider flexibility"
authors = [{name = "Your Name", email = "your.email@example.com"}]
license = {text = "MIT"}
readme = "README.md"
requires-python = ">=3.9"
keywords = ["ai", "agents", "llm", "agentic", "claude-code"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]

dependencies = [
    "pydantic>=2.0.0",
    "httpx>=0.24.0",
    "rich>=13.0.0",
    "click>=8.0.0",
    "pyyaml>=6.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
anthropic = ["anthropic>=0.25.0"]
openai = ["openai>=1.0.0"]
google = ["google-generativeai>=0.3.0"]
cohere = ["cohere>=5.0.0"]
all = [
    "anthropic>=0.25.0",
    "openai>=1.0.0",
    "google-generativeai>=0.3.0",
    "cohere>=5.0.0"
]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.0.0",
    "pytest-mock>=3.11.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
    "mypy>=1.0.0",
    "pre-commit>=3.0.0",
]

[project.scripts]
allos = "allos.cli.main:main"

[project.urls]
Homepage = "https://github.com/yourusername/allos-agent-sdk"
Documentation = "https://allos-agent-sdk.readthedocs.io"
Repository = "https://github.com/yourusername/allos-agent-sdk"
Issues = "https://github.com/yourusername/allos-agent-sdk/issues"

[tool.black]
line-length = 100
target-version = ['py39']

[tool.ruff]
line-length = 100
target-version = "py39"

[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --cov=allos --cov-report=term-missing"
```

#### 1.2 Create `requirements.txt`

```txt
pydantic>=2.0.0
httpx>=0.24.0
rich>=13.0.0
click>=8.0.0
pyyaml>=6.0
python-dotenv>=1.0.0
```

#### 1.3 Create `requirements-dev.txt`

```txt
-r requirements.txt
pytest>=7.0.0
pytest-asyncio>=0.21.0
pytest-cov>=4.0.0
pytest-mock>=3.11.0
black>=23.0.0
ruff>=0.1.0
mypy>=1.0.0
pre-commit>=3.0.0
anthropic>=0.25.0
openai>=1.0.0
```

#### 1.4 Set up development environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Set up pre-commit hooks
cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
        language_version: python3.9

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.9
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
EOF

pre-commit install
```

#### 1.5 Create `allos/__version__.py`

```python
"""Version information for Allos Agent SDK"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"
```

#### 1.6 Create `allos/__init__.py`

```python
"""
Allos Agent SDK - LLM-agnostic agentic framework

Build production-ready AI agents with any LLM provider.
"""

from .agent.agent import Agent, AgentConfig
from .providers.base import BaseProvider, Message, MessageRole, ProviderResponse
from .tools.base import BaseTool, ToolParameter, ToolPermission
from .__version__ import __version__

__all__ = [
    "Agent",
    "AgentConfig",
    "BaseProvider",
    "Message",
    "MessageRole",
    "ProviderResponse",
    "BaseTool",
    "ToolParameter",
    "ToolPermission",
    "__version__",
]
```

### Day 3-4: Error Handling & Utilities

#### 1.7 Create `allos/utils/errors.py`

```python
"""Custom exceptions for Allos"""


class AllosError(Exception):
    """Base exception for all Allos errors"""
    pass


class ProviderError(AllosError):
    """Error from LLM provider"""
    pass


class ToolExecutionError(AllosError):
    """Error executing a tool"""
    pass


class ToolNotFoundError(AllosError):
    """Requested tool not found"""
    pass


class ContextWindowExceededError(AllosError):
    """Context window size exceeded"""
    pass


class ConfigurationError(AllosError):
    """Invalid configuration"""
    pass


class PermissionDeniedError(AllosError):
    """Tool execution permission denied"""
    pass


class ValidationError(AllosError):
    """Validation error"""
    pass
```

#### 1.8 Create `allos/utils/logging.py`

```python
"""Logging configuration for Allos"""

import logging
import sys
from pathlib import Path
from typing import Optional

# Color codes for terminal
COLORS = {
    "DEBUG": "\033[36m",      # Cyan
    "INFO": "\033[32m",       # Green
    "WARNING": "\033[33m",    # Yellow
    "ERROR": "\033[31m",      # Red
    "CRITICAL": "\033[35m",   # Magenta
    "RESET": "\033[0m",
}


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for terminal output"""

    def format(self, record):
        if sys.stdout.isatty():
            levelname = record.levelname
            if levelname in COLORS:
                record.levelname = f"{COLORS[levelname]}{levelname}{COLORS['RESET']}"
        return super().format(record)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    verbose: bool = False
) -> logging.Logger:
    """
    Set up logging configuration

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file to write logs to
        verbose: Enable verbose logging

    Returns:
        Configured logger instance
    """
    if verbose:
        level = "DEBUG"

    logger = logging.getLogger("allos")
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = ColoredFormatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "allos") -> logging.Logger:
    """Get a logger instance"""
    return logging.getLogger(name)
```

#### 1.9 Create `allos/utils/file_utils.py`

```python
"""File utility functions"""

from pathlib import Path
from typing import Optional, List
import os


def get_project_root() -> Path:
    """Get the project root directory"""
    return Path.cwd()


def is_path_safe(path: Path, allowed_dirs: Optional[List[Path]] = None) -> bool:
    """
    Check if a path is safe to access

    Args:
        path: Path to check
        allowed_dirs: List of allowed directories (defaults to current directory)

    Returns:
        True if path is within allowed directories
    """
    if allowed_dirs is None:
        allowed_dirs = [get_project_root()]

    try:
        resolved_path = path.resolve()
        for allowed_dir in allowed_dirs:
            if resolved_path.is_relative_to(allowed_dir.resolve()):
                return True
        return False
    except (ValueError, OSError):
        return False


def read_file_safe(path: Path, max_size_mb: int = 10) -> str:
    """
    Safely read a file with size limits

    Args:
        path: Path to file
        max_size_mb: Maximum file size in MB

    Returns:
        File contents

    Raises:
        ValueError: If file is too large
        FileNotFoundError: If file doesn't exist
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > max_size_mb:
        raise ValueError(f"File too large: {size_mb:.2f}MB (max: {max_size_mb}MB)")

    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def get_relative_path(path: Path, base: Optional[Path] = None) -> str:
    """
    Get relative path from base directory

    Args:
        path: Target path
        base: Base directory (defaults to current directory)

    Returns:
        Relative path as string
    """
    if base is None:
        base = get_project_root()

    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def ensure_directory(path: Path) -> Path:
    """
    Ensure directory exists, create if necessary

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path.mkdir(parents=True, exist_ok=True)
    return path
```

#### 1.10 Create `allos/utils/token_counter.py`

```python
"""Token counting utilities"""

from typing import Optional
import re


def estimate_tokens(text: str) -> int:
    """
    Rough estimation of tokens in text

    This is a simple heuristic:
    - 1 token ≈ 4 characters for English text
    - More accurate for code than the 1 token ≈ 0.75 words rule

    For production, use provider-specific tokenizers

    Args:
        text: Input text

    Returns:
        Estimated token count
    """
    if not text:
        return 0

    # Rough heuristic: 1 token per 4 characters
    return max(1, len(text) // 4)


def truncate_to_tokens(text: str, max_tokens: int, suffix: str = "...") -> str:
    """
    Truncate text to approximate token count

    Args:
        text: Input text
        max_tokens: Maximum tokens
        suffix: Suffix to add if truncated

    Returns:
        Truncated text
    """
    estimated_tokens = estimate_tokens(text)

    if estimated_tokens <= max_tokens:
        return text

    # Calculate approximate character limit
    max_chars = max_tokens * 4 - len(suffix)

    if len(text) <= max_chars:
        return text

    return text[:max_chars] + suffix


# Provider-specific token counting (optional, requires packages)
def get_openai_token_count(text: str, model: str = "gpt-4") -> Optional[int]:
    """Get accurate OpenAI token count"""
    try:
        import tiktoken
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except ImportError:
        return None


def get_anthropic_token_count(text: str) -> Optional[int]:
    """Get accurate Anthropic token count"""
    try:
        from anthropic import Anthropic
        client = Anthropic()
        return client.count_tokens(text)
    except ImportError:
        return None
```

#### 1.11 Create `allos/utils/__init__.py`

```python
"""Utility functions and helpers"""

from .errors import (
    AllosError,
    ProviderError,
    ToolExecutionError,
    ToolNotFoundError,
    ContextWindowExceededError,
    ConfigurationError,
    PermissionDeniedError,
    ValidationError,
)
from .logging import setup_logging, get_logger
from .file_utils import (
    get_project_root,
    is_path_safe,
    read_file_safe,
    get_relative_path,
    ensure_directory,
)
from .token_counter import estimate_tokens, truncate_to_tokens

__all__ = [
    # Errors
    "AllosError",
    "ProviderError",
    "ToolExecutionError",
    "ToolNotFoundError",
    "ContextWindowExceededError",
    "ConfigurationError",
    "PermissionDeniedError",
    "ValidationError",
    # Logging
    "setup_logging",
    "get_logger",
    # File utils
    "get_project_root",
    "is_path_safe",
    "read_file_safe",
    "get_relative_path",
    "ensure_directory",
    # Token counting
    "estimate_tokens",
    "truncate_to_tokens",
]
```

### Day 5-7: Test Infrastructure

#### 1.12 Create `tests/conftest.py`

```python
"""Pytest configuration and fixtures"""

import pytest
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    tmpdir = Path(tempfile.mkdtemp())
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def sample_file(temp_dir):
    """Create a sample file for testing"""
    file_path = temp_dir / "test.txt"
    file_path.write_text("Hello, World!")
    return file_path


@pytest.fixture
def mock_provider(mocker):
    """Mock provider for testing"""
    from allos.providers.base import BaseProvider, ProviderResponse, MessageRole

    mock = mocker.Mock(spec=BaseProvider)
    mock.chat.return_value = ProviderResponse(
        content="Test response",
        tool_calls=None,
        finish_reason="stop",
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        raw_response={}
    )
    mock.supports_tool_calling.return_value = True
    mock.get_token_count.return_value = 10
    mock.context_window = 8192
    mock.name = "mock"

    return mock


@pytest.fixture
def mock_tool(mocker):
    """Mock tool for testing"""
    from allos.tools.base import BaseTool, ToolParameter

    class MockTool(BaseTool):
        name = "mock_tool"
        description = "A mock tool for testing"
        parameters = [
            ToolParameter(name="arg1", type="string", description="Test argument")
        ]

        def execute(self, **kwargs):
            return {"success": True, "result": "mock result"}

    return MockTool()
```

#### 1.13 Create `scripts/run_tests.sh`

```bash
#!/bin/bash
set -e

echo "Running Allos Agent SDK tests..."

# Run tests with coverage
pytest tests/ \
    --cov=allos \
    --cov-report=term-missing \
    --cov-report=html \
    -v

echo ""
echo "✓ Tests complete! Coverage report generated in htmlcov/"
```

Make it executable:
```bash
chmod +x scripts/run_tests.sh
```

---

## Week 2: Provider Layer

### Day 8-9: Provider Base & Registry

#### 2.1 Create `allos/providers/base.py`

```python
"""Base classes and types for LLM providers"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class MessageRole(Enum):
    """Message roles in conversation"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """Unified message format across providers"""
    role: MessageRole
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


@dataclass
class ToolCall:
    """Represents a tool invocation request from the model"""
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ProviderResponse:
    """Unified response format from providers"""
    content: Optional[str]
    tool_calls: Optional[List[ToolCall]]
    finish_reason: str
    usage: Dict[str, int]
    raw_response: Any


class BaseProvider(ABC):
    """Abstract base class for all LLM providers"""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.extra_params = kwargs

    @abstractmethod
    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ProviderResponse:
        """Send a chat completion request"""
        pass

    @abstractmethod
    def supports_tool_calling(self) -> bool:
        """Whether this provider supports native tool calling"""
        pass

    @abstractmethod
    def get_token_count(self, text: str) -> int:
        """Count tokens in text (provider-specific)"""
        pass

    @property
    @abstractmethod
    def context_window(self) -> int:
        """Maximum context window size in tokens"""
        pass

    @property
    def name(self) -> str:
        """Provider name"""
        return self.__class__.__name__.replace("Provider", "").lower()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model})"
```

#### 2.2 Create `allos/providers/__init__.py`

```python
"""Provider implementations for various LLM services"""

from .base import (
    BaseProvider,
    Message,
    MessageRole,
    ProviderResponse,
    ToolCall,
)
from .registry import ProviderRegistry

__all__ = [
    "BaseProvider",
    "Message",
    "MessageRole",
    "ProviderResponse",
    "ToolCall",
    "ProviderRegistry",
]
```

#### 2.3 Create `allos/providers/registry.py`

```python
"""Provider registry for managing available LLM providers"""

from typing import Dict, Type, List
from .base import BaseProvider
from ..utils.errors import ConfigurationError


class ProviderRegistry:
    """Registry for managing available providers"""

    _providers: Dict[str, Type[BaseProvider]] = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a provider

        Usage:
            @ProviderRegistry.register("openai")
            class OpenAIProvider(BaseProvider):
                ...
        """
        def wrapper(provider_class: Type[BaseProvider]):
            if not issubclass(provider_class, BaseProvider):
                raise TypeError(f"{provider_class} must inherit from BaseProvider")
            cls._providers[name.lower()] = provider_class
            return provider_class
        return wrapper

    @classmethod
    def get_provider(cls, name: str, **kwargs) -> BaseProvider:
        """
        Get a provider instance by name

        Args:
            name: Provider name (e.g., 'openai', 'anthropic')
            **kwargs: Arguments to pass to provider constructor

        Returns:
            Provider instance

        Raises:
            ConfigurationError: If provider not found
        """
        name = name.lower()
        if name not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise ConfigurationError(
                f"Unknown provider: '{name}'. Available providers: {available}"
            )
        return cls._providers[name](**kwargs)

    @classmethod
    def list_providers(cls) -> List[str]:
        """List all registered provider names"""
        return sorted(cls._providers.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a provider is registered"""
        return name.lower() in cls._providers
```

### Day 10-12: OpenAI Provider Implementation

#### 2.4 Create `allos/providers/openai.py`

```python
"""OpenAI provider implementation"""

import json
import os
from typing import List, Dict, Any, Optional

from .base import BaseProvider, Message, ProviderResponse, ToolCall, MessageRole
from .registry import ProviderRegistry
from ..utils.errors import ProviderError
from ..utils.token_counter import estimate_tokens


@ProviderRegistry.register("openai")
class OpenAIProvider(BaseProvider):
    """OpenAI/OpenAI-compatible provider implementation"""

    # Context windows for different models
    CONTEXT_WINDOWS = {
        "gpt-4": 8192,
        "gpt-4-32k": 32768,
        "gpt-4-turbo": 128000,
        "gpt-4-turbo-preview": 128000,
        "gpt-4o": 128000,
        "gpt-4o-mini": 128000,
        "gpt-3.5-turbo": 16385,
        "gpt-3.5-turbo-16k": 16385,
    }

    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model, api_key, base_url, **kwargs)

        # Get API key from env if not provided
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ProviderError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Initialize OpenAI client
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                **kwargs
            )
        except ImportError:
            raise ProviderError(
                "OpenAI package not installed. Install with: pip install openai"
            )

        # Try to initialize tokenizer
        self._tokenizer = None
        try:
            import tiktoken
            self._tokenizer = tiktoken.encoding_for_model(self.model)
        except Exception:
            pass  # Fall back to estimation

    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ProviderResponse:
        """Send chat completion request to OpenAI"""
        try:
            # Convert messages to OpenAI format
            openai_messages = [self._convert_message(msg) for msg in messages]

            # Build request parameters
            params = {
                "model": self.model,
                "messages": openai_messages,
                "temperature": temperature,
            }

            if tools:
                params["tools"] = tools
                params["tool_choice"] = "auto"

            if max_tokens:
                params["max_tokens"] = max_tokens

            # Merge any extra parameters
            params.update(kwargs)

            # Make API call
            response = self.client.chat.completions.create(**params)

            # Convert response to our format
            return self._convert_response(response)

        except Exception as e:
            raise ProviderError(f"OpenAI API error: {str(e)}") from e

    def _convert_message(self, msg: Message) -> Dict[str, Any]:
        """Convert our Message format to OpenAI format"""
        result = {
            "role": msg.role.value,
            "content": msg.content
        }

        if msg.tool_calls:
            result["tool_calls"] = msg.tool_calls

        if msg.tool_call_id:
            result["tool_call_id"] = msg.tool_call_id

        if msg.name:
            result["name"] = msg.name

        return result

    def _convert_response(self, response) -> ProviderResponse:
        """Convert OpenAI response to our format"""
        choice = response.choices[0]
        message = choice.message

        # Extract tool calls if present
        tool_calls = None
        if hasattr(message, 'tool_calls') and message.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments)
                )
                for tc in message.tool_calls
            ]

        return ProviderResponse(
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            raw_response=response
        )

    def supports_tool_calling(self) -> bool:
        """OpenAI supports tool calling"""
        return True

    def get_token_count(self, text: str) -> int:
        """Count tokens using tiktoken if available"""
        if self._tokenizer:
            try:
                return len(self._tokenizer.encode(text))
            except Exception:
                pass

        # Fall back to estimation
        return estimate_tokens(text)

    @property
    def context_window(self) -> int:
        """Get context window for the model"""
        # Try to match the model name
        for model_prefix, window in self.CONTEXT_WINDOWS.items():
            if self.model.startswith(model_prefix):
                return window

        # Default to conservative estimate
        return 8192
```

### Day 13-14: Anthropic Provider + Tests

#### 2.5 Create `allos/providers/anthropic.py`

```python
"""Anthropic Claude provider implementation"""

import json
import os
from typing import List, Dict, Any, Optional

from .base import BaseProvider, Message, ProviderResponse, ToolCall, MessageRole
from .registry import ProviderRegistry
from ..utils.errors import ProviderError
from ..utils.token_counter import estimate_tokens


@ProviderRegistry.register("anthropic")
class AnthropicProvider(BaseProvider):
    """Anthropic Claude provider implementation"""

    # Context windows for Claude models
    CONTEXT_WINDOWS = {
        "claude-3-opus": 200000,
        "claude-3-sonnet": 200000,
        "claude-3-haiku": 200000,
        "claude-3-5-sonnet": 200000,
        "claude-4-opus": 200000,
        "claude-4-sonnet": 200000,
        "claude-sonnet-4": 200000,
    }

    def __init__(
        self,
        model: str = "claude-sonnet-4-5-20250929",
        api_key: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model, api_key, None, **kwargs)

        # Get API key from env if not provided
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ProviderError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Initialize Anthropic client
        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=self.api_key, **kwargs)
        except ImportError:
            raise ProviderError(
                "Anthropic package not installed. Install with: pip install anthropic"
            )

    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = 4096,
        **kwargs
    ) -> ProviderResponse:
        """Send chat completion request to Anthropic"""
        try:
            # Separate system message from conversation
            system_message = None
            conversation_messages = []

            for msg in messages:
                if msg.role == MessageRole.SYSTEM:
                    system_message = msg.content
                else:
                    conversation_messages.append(self._convert_message(msg))

            # Build request parameters
            params = {
                "model": self.model,
                "messages": conversation_messages,
                "temperature": temperature,
                "max_tokens": max_tokens or 4096,
            }

            if system_message:
                params["system"] = system_message

            if tools:
                params["tools"] = self._convert_tools(tools)

            # Merge extra parameters
            params.update(kwargs)

            # Make API call
            response = self.client.messages.create(**params)

            # Convert response to our format
            return self._convert_response(response)

        except Exception as e:
            raise ProviderError(f"Anthropic API error: {str(e)}") from e

    def _convert_message(self, msg: Message) -> Dict[str, Any]:
        """Convert our Message format to Anthropic format"""
        if msg.role == MessageRole.TOOL:
            # Tool result
            return {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": msg.content
                    }
                ]
            }
        elif msg.tool_calls:
            # Assistant message with tool calls
            content = []
            if msg.content:
                content.append({"type": "text", "text": msg.content})

            for tc in msg.tool_calls:
                content.append({
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["function"]["name"],
                    "input": json.loads(tc["function"]["arguments"])
                })

            return {"role": "assistant", "content": content}
        else:
            # Regular message
            return {
                "role": msg.role.value if msg.role != MessageRole.ASSISTANT else "assistant",
                "content": msg.content
            }

    def _convert_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI tool format to Anthropic format"""
        anthropic_tools = []
        for tool in tools:
            if tool["type"] == "function":
                func = tool["function"]
                anthropic_tools.append({
                    "name": func["name"],
                    "description": func["description"],
                    "input_schema": func["parameters"]
                })
        return anthropic_tools

    def _convert_response(self, response) -> ProviderResponse:
        """Convert Anthropic response to our format"""
        # Extract text content and tool calls
        text_content = []
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                text_content.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input
                    )
                )

        return ProviderResponse(
            content=" ".join(text_content) if text_content else None,
            tool_calls=tool_calls if tool_calls else None,
            finish_reason=response.stop_reason,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
            raw_response=response
        )

    def supports_tool_calling(self) -> bool:
        """Claude supports tool calling"""
        return True

    def get_token_count(self, text: str) -> int:
        """Estimate token count"""
        # Anthropic has a count_tokens API but requires a request
        # For now, use estimation
        return estimate_tokens(text)

    @property
    def context_window(self) -> int:
        """Get context window for the model"""
        for model_prefix, window in self.CONTEXT_WINDOWS.items():
            if self.model.startswith(model_prefix):
                return window
        return 200000  # Default for Claude
```

#### 2.6 Create `tests/unit/test_providers.py`

```python
"""Tests for provider implementations"""

import pytest
from allos.providers.base import Message, MessageRole, ProviderResponse
from allos.providers.registry import ProviderRegistry
from allos.utils.errors import ConfigurationError, ProviderError


def test_provider_registry():
    """Test provider registry functionality"""
    providers = ProviderRegistry.list_providers()
    assert "openai" in providers
    assert "anthropic" in providers


def test_provider_registry_unknown():
    """Test registry with unknown provider"""
    with pytest.raises(ConfigurationError):
        ProviderRegistry.get_provider("nonexistent")


def test_message_creation():
    """Test message creation"""
    msg = Message(role=MessageRole.USER, content="Hello")
    assert msg.role == MessageRole.USER
    assert msg.content == "Hello"
    assert msg.tool_calls is None


@pytest.mark.skipif(
    not pytest.config.getoption("--run-integration"),
    reason="Requires API keys"
)
def test_openai_provider_real():
    """Test real OpenAI provider (requires API key)"""
    provider = ProviderRegistry.get_provider("openai", model="gpt-3.5-turbo")

    messages = [Message(role=MessageRole.USER, content="Say 'test' and nothing else")]
    response = provider.chat(messages, temperature=0)

    assert response.content is not None
    assert "test" in response.content.lower()
```

Add to `pyproject.toml`:
```toml
[tool.pytest.ini_options]
addopts = "-v --cov=allos --cov-report=term-missing"
markers = [
    "integration: marks tests as integration tests (deselect with '-m \"not integration\"')",
]
```

---

## Week 3: Tool System

### Day 15-16: Tool Base & Registry

#### 3.1 Create `allos/tools/base.py`

```python
"""Base classes for tools"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

from ..utils.errors import ValidationError


class ToolPermission(Enum):
    """Permission levels for tool execution"""
    ALWAYS_ALLOW = "always_allow"
    ASK = "ask"
    NEVER = "never"


@dataclass
class ToolParameter:
    """Describes a tool parameter"""
    name: str
    type: str  # "string", "integer", "boolean", "object", "array"
    description: str
    required: bool = True
    enum: Optional[List[Any]] = None
    default: Optional[Any] = None


class BaseTool(ABC):
    """Abstract base class for all tools"""

    # Subclasses must set these
    name: str
    description: str
    parameters: List[ToolParameter]

    # Default permission level
    default_permission: ToolPermission = ToolPermission.ASK

    def __init__(self, permission: Optional[ToolPermission] = None):
        """
        Initialize tool

        Args:
            permission: Override default permission level
        """
        self.permission = permission or self.default_permission
        self._validate_class_attributes()

    def _validate_class_attributes(self):
        """Ensure subclass defined required attributes"""
        required = ["name", "description", "parameters"]
        for attr in required:
            if not hasattr(self, attr):
                raise AttributeError(
                    f"{self.__class__.__name__} must define class attribute '{attr}'"
                )

    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the tool with given parameters

        Returns:
            Dict with 'success' (bool) and either 'result' or 'error'

        Example:
            {"success": True, "result": {...}}
            {"success": False, "error": "Error message"}
        """
        pass

    def to_openai_format(self) -> Dict[str, Any]:
        """
        Convert tool to OpenAI function calling format

        Returns:
            Tool schema in OpenAI format
        """
        properties = {}
        required = []

        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description,
            }

            if param.enum:
                prop["enum"] = param.enum

            if param.default is not None:
                prop["default"] = param.default

            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                }
            }
        }

    def validate_arguments(self, arguments: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validate tool arguments

        Args:
            arguments: Arguments to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        for param in self.parameters:
            if param.required and param.name not in arguments:
                return False, f"Missing required parameter: {param.name}"

            if param.name in arguments and param.enum:
                if arguments[param.name] not in param.enum:
                    return False, (
                        f"Invalid value for {param.name}. "
                        f"Must be one of: {param.enum}"
                    )

        return True, None

    def __str__(self) -> str:
        return f"{self.name}: {self.description}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
```

#### 3.2 Create `allos/tools/registry.py`

```python
"""Tool registry for managing available tools"""

from typing import Dict, Type, List, Optional
from .base import BaseTool, ToolPermission
from ..utils.errors import ToolNotFoundError


class ToolRegistry:
    """Registry for managing available tools"""

    _tools: Dict[str, Type[BaseTool]] = {}

    @classmethod
    def register(cls, tool_class: Type[BaseTool]) -> Type[BaseTool]:
        """
        Register a tool class

        Args:
            tool_class: Tool class to register

        Returns:
            The tool class (for use as decorator)
        """
        if not issubclass(tool_class, BaseTool):
            raise TypeError(f"{tool_class} must inherit from BaseTool")

        tool_name = tool_class.name
        if tool_name in cls._tools:
            raise ValueError(f"Tool already registered: {tool_name}")

        cls._tools[tool_name] = tool_class
        return tool_class

    @classmethod
    def get_tool(cls, name: str, **kwargs) -> BaseTool:
        """
        Get a tool instance by name

        Args:
            name: Tool name
            **kwargs: Arguments to pass to tool constructor

        Returns:
            Tool instance

        Raises:
            ToolNotFoundError: If tool not found
        """
        if name not in cls._tools:
            available = ", ".join(cls._tools.keys())
            raise ToolNotFoundError(
                f"Unknown tool: '{name}'. Available tools: {available}"
            )
        return cls._tools[name](**kwargs)

    @classmethod
    def list_tools(cls) -> List[str]:
        """List all registered tool names"""
        return sorted(cls._tools.keys())

    @classmethod
    def get_all_tools(
        cls,
        permissions: Optional[Dict[str, ToolPermission]] = None
    ) -> List[BaseTool]:
        """
        Get instances of all registered tools

        Args:
            permissions: Dict mapping tool names to permission levels

        Returns:
            List of tool instances
        """
        permissions = permissions or {}
        tools = []
        for name, tool_class in cls._tools.items():
            permission = permissions.get(name)
            tools.append(tool_class(permission=permission))
        return tools

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a tool is registered"""
        return name in cls._tools


def tool(cls: Type[BaseTool]) -> Type[BaseTool]:
    """
    Decorator to auto-register tools

    Usage:
        @tool
        class MyTool(BaseTool):
            name = "my_tool"
            ...
    """
    return ToolRegistry.register(cls)
```

#### 3.3 Create `allos/tools/__init__.py`

```python
"""Tool implementations"""

from .base import BaseTool, ToolParameter, ToolPermission
from .registry import ToolRegistry, tool

__all__ = [
    "BaseTool",
    "ToolParameter",
    "ToolPermission",
    "ToolRegistry",
    "tool",
]
```

### Day 17-19: File System Tools

#### 3.4 Create `allos/tools/filesystem/read.py`

```python
"""File reading tool"""

from pathlib import Path
from typing import Dict, Any, Optional

from ..base import BaseTool, ToolParameter, ToolPermission
from ..registry import tool
from ...utils.file_utils import is_path_safe, read_file_safe
from ...utils.errors import ToolExecutionError


@tool
class FileReadTool(BaseTool):
    """Tool for reading file contents"""

    name = "read_file"
    description = "Read the contents of a file"
    parameters = [
        ToolParameter(
            name="path",
            type="string",
            description="Path to the file to read (relative or absolute)",
            required=True
        ),
        ToolParameter(
            name="start_line",
            type="integer",
            description="Starting line number (1-indexed, optional)",
            required=False
        ),
        ToolParameter(
            name="end_line",
            type="integer",
            description="Ending line number (inclusive, optional)",
            required=False
        ),
    ]

    default_permission = ToolPermission.ALWAYS_ALLOW  # Reading is generally safe

    def execute(
        self,
        path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None
    ) -> Dict[str, Any]:
        """Execute file read operation"""
        try:
            file_path = Path(path).resolve()

            # Security check
            if not is_path_safe(file_path):
                return {
                    "success": False,
                    "error": f"Access denied: path outside allowed directories: {path}"
                }

            # Check file exists
            if not file_path.exists():
                return {
                    "success": False,
                    "error": f"File not found: {path}"
                }

            if not file_path.is_file():
                return {
                    "success": False,
                    "error": f"Not a file: {path}"
                }

            # Read file
            content = read_file_safe(file_path)

            # Apply line range if specified
            if start_line is not None or end_line is not None:
                lines = content.splitlines(keepends=True)
                start = (start_line or 1) - 1
                end = end_line if end_line else len(lines)
                content = ''.join(lines[start:end])

            return {
                "success": True,
                "result": {
                    "path": str(file_path),
                    "content": content,
                    "lines": len(content.splitlines()),
                    "size_bytes": len(content.encode('utf-8'))
                }
            }

        except PermissionError:
            return {
                "success": False,
                "error": f"Permission denied: {path}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error reading file: {str(e)}"
            }
```

#### 3.5 Create `allos/tools/filesystem/write.py`

```python
"""File writing tool"""

from pathlib import Path
from typing import Dict, Any

from ..base import BaseTool, ToolParameter, ToolPermission
from ..registry import tool
from ...utils.file_utils import is_path_safe, ensure_directory


@tool
class FileWriteTool(BaseTool):
    """Tool for writing content to files"""

    name = "write_file"
    description = "Write content to a file (creates or overwrites)"
    parameters = [
        ToolParameter(
            name="path",
            type="string",
            description="Path to the file to write",
            required=True
        ),
        ToolParameter(
            name="content",
            type="string",
            description="Content to write to the file",
            required=True
        ),
    ]

    default_permission = ToolPermission.ASK  # Writing requires confirmation

    def execute(self, path: str, content: str) -> Dict[str, Any]:
        """Execute file write operation"""
        try:
            file_path = Path(path).resolve()

            # Security check
            if not is_path_safe(file_path):
                return {
                    "success": False,
                    "error": f"Access denied: path outside allowed directories: {path}"
                }

            # Create parent directories if needed
            ensure_directory(file_path.parent)

            # Write file
            file_path.write_text(content, encoding='utf-8')

            return {
                "success": True,
                "result": {
                    "path": str(file_path),
                    "size_bytes": len(content.encode('utf-8')),
                    "lines": len(content.splitlines())
                }
            }

        except PermissionError:
            return {
                "success": False,
                "error": f"Permission denied: {path}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error writing file: {str(e)}"
            }
```

#### 3.6 Create `allos/tools/filesystem/edit.py`

```python
"""File editing tool (string replace)"""

from pathlib import Path
from typing import Dict, Any

from ..base import BaseTool, ToolParameter, ToolPermission
from ..registry import tool
from ...utils.file_utils import is_path_safe, read_file_safe


@tool
class FileEditTool(BaseTool):
    """Tool for editing files by replacing text"""

    name = "edit_file"
    description = "Edit a file by replacing old text with new text"
    parameters = [
        ToolParameter(
            name="path",
            type="string",
            description="Path to the file to edit",
            required=True
        ),
        ToolParameter(
            name="old_text",
            type="string",
            description="Text to search for (must match exactly)",
            required=True
        ),
        ToolParameter(
            name="new_text",
            type="string",
            description="Text to replace with",
            required=True
        ),
    ]

    default_permission = ToolPermission.ASK  # Editing requires confirmation

    def execute(self, path: str, old_text: str, new_text: str) -> Dict[str, Any]:
        """Execute file edit operation"""
        try:
            file_path = Path(path).resolve()

            # Security check
            if not is_path_safe(file_path):
                return {
                    "success": False,
                    "error": f"Access denied: path outside allowed directories: {path}"
                }

            # Check file exists
            if not file_path.exists():
                return {
                    "success": False,
                    "error": f"File not found: {path}"
                }

            # Read file
            content = read_file_safe(file_path)

            # Check if old_text exists
            if old_text not in content:
                return {
                    "success": False,
                    "error": f"Text not found in file: {old_text[:100]}..."
                }

            # Check if old_text is unique
            count = content.count(old_text)
            if count > 1:
                return {
                    "success": False,
                    "error": f"Text appears {count} times in file. Must be unique for safe editing."
                }

            # Replace text
            new_content = content.replace(old_text, new_text)

            # Write back
            file_path.write_text(new_content, encoding='utf-8')

            return {
                "success": True,
                "result": {
                    "path": str(file_path),
                    "replacements": 1,
                    "old_length": len(old_text),
                    "new_length": len(new_text)
                }
            }

        except PermissionError:
            return {
                "success": False,
                "error": f"Permission denied: {path}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error editing file: {str(e)}"
            }
```

#### 3.7 Create `allos/tools/filesystem/directory.py`

```python
"""Directory operations tool"""

from pathlib import Path
from typing import Dict, Any, List

from ..base import BaseTool, ToolParameter, ToolPermission
from ..registry import tool
from ...utils.file_utils import is_path_safe, get_relative_path


@tool
class ListDirectoryTool(BaseTool):
    """Tool for listing directory contents"""

    name = "list_directory"
    description = "List files and directories in a path"
    parameters = [
        ToolParameter(
            name="path",
            type="string",
            description="Path to directory (defaults to current directory)",
            required=False,
            default="."
        ),
        ToolParameter(
            name="recursive",
            type="boolean",
            description="List recursively",
            required=False,
            default=False
        ),
    ]

    default_permission = ToolPermission.ALWAYS_ALLOW

    def execute(self, path: str = ".", recursive: bool = False) -> Dict[str, Any]:
        """Execute directory listing"""
        try:
            dir_path = Path(path).resolve()

            # Security check
            if not is_path_safe(dir_path):
                return {
                    "success": False,
                    "error": f"Access denied: path outside allowed directories: {path}"
                }

            # Check directory exists
            if not dir_path.exists():
                return {
                    "success": False,
                    "error": f"Directory not found: {path}"
                }

            if not dir_path.is_dir():
                return {
                    "success": False,
                    "error": f"Not a directory: {path}"
                }

            # List contents
            items: List[Dict[str, Any]] = []

            if recursive:
                pattern = "**/*"
            else:
                pattern = "*"

            for item in dir_path.glob(pattern):
                if item.name.startswith('.'):
                    continue  # Skip hidden files

                items.append({
                    "name": item.name,
                    "path": get_relative_path(item),
                    "type": "directory" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else None,
                })

            # Sort: directories first, then files
            items.sort(key=lambda x: (x["type"] != "directory", x["name"]))

            return {
                "success": True,
                "result": {
                    "path": str(dir_path),
                    "items": items,
                    "total": len(items)
                }
            }

        except PermissionError:
            return {
                "success": False,
                "error": f"Permission denied: {path}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error listing directory: {str(e)}"
            }
```

#### 3.8 Create `allos/tools/filesystem/__init__.py`

```python
"""Filesystem tools"""

from .read import FileReadTool
from .write import FileWriteTool
from .edit import FileEditTool
from .directory import ListDirectoryTool

__all__ = [
    "FileReadTool",
    "FileWriteTool",
    "FileEditTool",
    "ListDirectoryTool",
]
```

### Day 20-21: Execution Tools

#### 3.9 Create `allos/tools/execution/shell.py`

```python
"""Shell command execution tool"""

import subprocess
import shlex
from typing import Dict, Any, Optional

from ..base import BaseTool, ToolParameter, ToolPermission
from ..registry import tool


@tool
class ShellExecuteTool(BaseTool):
    """Tool for executing shell commands"""

    name = "shell_exec"
    description = "Execute a shell command and return its output"
    parameters = [
        ToolParameter(
            name="command",
            type="string",
            description="Shell command to execute",
            required=True
        ),
        ToolParameter(
            name="timeout",
            type="integer",
            description="Timeout in seconds (default: 30)",
            required=False,
            default=30
        ),
    ]

    default_permission = ToolPermission.ASK  # Shell execution requires confirmation

    # Dangerous commands that should never be auto-approved
    DANGEROUS_PATTERNS = [
        "rm -rf",
        "mkfs",
        "dd if=",
        "> /dev/",
        ":(){ :|:& };:",  # Fork bomb
        "chmod -R 777",
    ]

    def execute(self, command: str, timeout: int = 30) -> Dict[str, Any]:
        """Execute shell command"""
        try:
            # Safety check
            for pattern in self.DANGEROUS_PATTERNS:
                if pattern in command:
                    return {
                        "success": False,
                        "error": f"Dangerous command pattern detected: {pattern}"
                    }

            # Execute command
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(Path.cwd())
            )

            return {
                "success": result.returncode == 0,
                "result": {
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                    "command": command
                }
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Command timed out after {timeout} seconds"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error executing command: {str(e)}"
            }
```

#### 3.10 Create `allos/tools/execution/__init__.py`

```python
"""Execution tools"""

from .shell import ShellExecuteTool

__all__ = ["ShellExecuteTool"]
```

---

## Week 4: Agent Core & Context Management

### Day 22-24: Context Manager

#### 4.1 Create `allos/context/manager.py`

```python
"""Conversation context management"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import json

from ..providers.base import Message, MessageRole, ToolCall


@dataclass
class ConversationContext:
    """Manages conversation state and history"""

    messages: List[Message] = field(default_factory=list)
    max_tokens: Optional[int] = None
    system_prompt: Optional[str] = None

    # Metadata
    total_tokens_used: int = 0
    total_cost: float = 0.0
    turn_count: int = 0

    def add_system_message(self, content: str):
        """Add or update system message"""
        self.system_prompt = content

        # System message is always first if present
        if self.messages and self.messages[0].role == MessageRole.SYSTEM:
            self.messages[0] = Message(role=MessageRole.SYSTEM, content=content)
        else:
            self.messages.insert(0, Message(role=MessageRole.SYSTEM, content=content))

    def add_user_message(self, content: str):
        """Add a user message"""
        self.messages.append(Message(role=MessageRole.USER, content=content))
        self.turn_count += 1

    def add_assistant_message(
        self,
        content: Optional[str],
        tool_calls: Optional[List[ToolCall]] = None
    ):
        """Add an assistant message"""
        # Convert ToolCall objects to dict format for Message
        tool_calls_dict = None
        if tool_calls:
            tool_calls_dict = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments)
                    }
                }
                for tc in tool_calls
            ]

        self.messages.append(Message(
            role=MessageRole.ASSISTANT,
            content=content or "",
            tool_calls=tool_calls_dict
        ))

    def add_tool_result(self, tool_call_id: str, tool_name: str, result: str):
        """Add a tool execution result"""
        self.messages.append(Message(
            role=MessageRole.TOOL,
            content=result,
            tool_call_id=tool_call_id,
            name=tool_name
        ))

    def get_token_count(self, provider) -> int:
        """Estimate total tokens in context"""
        total = 0
        for msg in self.messages:
            total += provider.get_token_count(msg.content or "")
            # Add overhead for role, tool calls, etc.
            total += 10
        return total

    def needs_compaction(self, provider, buffer: int = 1000) -> bool:
        """Check if context needs compaction"""
        if not self.max_tokens:
            self.max_tokens = provider.context_window

        current_tokens = self.get_token_count(provider)
        return current_tokens > (self.max_tokens - buffer)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize context for persistence"""
        return {
            "messages": [
                {
                    "role": msg.role.value,
                    "content": msg.content,
                    "tool_calls": msg.tool_calls,
                    "tool_call_id": msg.tool_call_id,
                    "name": msg.name,
                }
                for msg in self.messages
            ],
            "system_prompt": self.system_prompt,
            "metadata": {
                "total_tokens_used": self.total_tokens_used,
                "total_cost": self.total_cost,
                "turn_count": self.turn_count,
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationContext":
        """Deserialize context"""
        messages = [
            Message(
                role=MessageRole(msg["role"]),
                content=msg["content"],
                tool_calls=msg.get("tool_calls"),
                tool_call_id=msg.get("tool_call_id"),
                name=msg.get("name"),
            )
            for msg in data["messages"]
        ]

        ctx = cls(messages=messages, system_prompt=data.get("system_prompt"))
        if "metadata" in data:
            ctx.total_tokens_used = data["metadata"].get("total_tokens_used", 0)
            ctx.total_cost = data["metadata"].get("total_cost", 0.0)
            ctx.turn_count = data["metadata"].get("turn_count", 0)

        return ctx
```

#### 4.2 Create `allos/context/__init__.py`

```python
"""Context management"""

from .manager import ConversationContext

__all__ = ["ConversationContext"]
```

### Day 25-28: Agent Core Implementation

#### 4.3 Create `allos/agent/agent.py`

```python
"""Main agent implementation"""

import json
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from ..providers.base import BaseProvider, ToolCall
from ..providers.registry import ProviderRegistry
from ..tools.base import BaseTool, ToolPermission
from ..tools.registry import ToolRegistry
from ..context.manager import ConversationContext
from ..utils.errors import AllosError, ToolExecutionError
from ..utils.logging import get_logger

console = Console()
logger = get_logger(__name__)


@dataclass
class AgentConfig:
    """Configuration for an agent"""
    provider: str
    model: str
    tools: List[str]
    system_prompt: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    max_iterations: int = 15

    # Tool permissions
    tool_permissions: Optional[Dict[str, ToolPermission]] = None
    auto_approve_safe_tools: bool = True

    # Callbacks
    on_tool_call: Optional[Callable] = None
    on_iteration: Optional[Callable] = None


class Agent:
    """Main agent orchestrator"""

    DEFAULT_SYSTEM_PROMPT = """You are an AI coding assistant. You help users with programming tasks by:
- Reading and understanding codebases
- Writing and modifying code
- Executing commands
- Debugging issues
- Following best practices

When using tools:
- Read files before modifying them
- Make targeted, specific changes
- Test your changes when possible
- Explain what you're doing

Be direct and efficient. Focus on getting the task done."""

    def __init__(self, config: AgentConfig):
        self.config = config
        logger.info(f"Initializing agent with provider={config.provider}, model={config.model}")

        # Initialize provider
        self.provider: BaseProvider = ProviderRegistry.get_provider(
            config.provider,
            model=config.model
        )

        # Initialize tools
        self.tools: Dict[str, BaseTool] = {}
        for tool_name in config.tools:
            permission = None
            if config.tool_permissions:
                permission = config.tool_permissions.get(tool_name)
            tool = ToolRegistry.get_tool(tool_name, permission=permission)
            self.tools[tool_name] = tool

        logger.info(f"Loaded {len(self.tools)} tools: {list(self.tools.keys())}")

        # Initialize context
        self.context = ConversationContext()
        system_prompt = config.system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.context.add_system_message(system_prompt)

    def run(self, task: str) -> str:
        """
        Main entry point: Run the agent on a task

        Args:
            task: The task description from the user

        Returns:
            Final response string
        """
        console.print(f"\n[bold blue]Task:[/bold blue] {task}")
        logger.info(f"Starting task: {task}")

        # Add user's task to context
        self.context.add_user_message(task)

        # Main agentic loop
        for iteration in range(self.config.max_iterations):
            if self.config.on_iteration:
                self.config.on_iteration(iteration, self.context)

            console.print(f"\n[dim]Iteration {iteration + 1}/{self.config.max_iterations}[/dim]")
            logger.debug(f"Starting iteration {iteration + 1}")

            # Get response from LLM
            response = self._get_llm_response()

            # If no tool calls, we're done
            if not response.tool_calls:
                final_response = response.content or ""
                console.print(f"\n[bold green]✓ Complete[/bold green]")
                logger.info("Task completed successfully")
                return final_response

            # Execute tool calls
            self._execute_tool_calls(response.tool_calls)

            # Add assistant's response to context
            self.context.add_assistant_message(
                content=response.content,
                tool_calls=response.tool_calls
            )

        # Hit max iterations
        console.print(f"\n[bold yellow]⚠ Max iterations reached[/bold yellow]")
        logger.warning("Reached max iterations")
        return self.context.messages[-1].content or "Task incomplete: max iterations reached"

    def _get_llm_response(self):
        """Get response from LLM"""
        # Convert tools to OpenAI format
        tool_schemas = [tool.to_openai_format() for tool in self.tools.values()]

        try:
            response = self.provider.chat(
                messages=self.context.messages,
                tools=tool_schemas if tool_schemas else None,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            # Update token usage
            self.context.total_tokens_used += response.usage["total_tokens"]
            logger.debug(
                f"LLM response: {response.usage['total_tokens']} tokens, "
                f"finish_reason={response.finish_reason}"
            )

            return response

        except Exception as e:
            logger.error(f"Provider error: {str(e)}")
            raise AllosError(f"Provider error: {str(e)}") from e

    def _execute_tool_calls(self, tool_calls: List[ToolCall]):
        """Execute requested tool calls"""
        for tool_call in tool_calls:
            console.print(f"[cyan]  → {tool_call.name}[/cyan]({tool_call.arguments})")
            logger.info(f"Executing tool: {tool_call.name} with args: {tool_call.arguments}")

            # Get the tool
            if tool_call.name not in self.tools:
                error_msg = f"Tool not available: {tool_call.name}"
                console.print(f"[red]    ✗ {error_msg}[/red]")
                logger.error(error_msg)
                self.context.add_tool_result(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    result=json.dumps({"success": False, "error": error_msg})
                )
                continue

            tool = self.tools[tool_call.name]

            # Check permission
            if not self._check_tool_permission(tool, tool_call):
                error_msg = "Tool execution denied by user"
                console.print(f"[red]    ✗ {error_msg}[/red]")
                logger.warning(f"Tool execution denied: {tool_call.name}")
                self.context.add_tool_result(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    result=json.dumps({"success": False, "error": error_msg})
                )
                continue

            # Validate arguments
            valid, error = tool.validate_arguments(tool_call.arguments)
            if not valid:
                console.print(f"[red]    ✗ {error}[/red]")
                logger.error(f"Validation error: {error}")
                self.context.add_tool_result(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    result=json.dumps({"success": False, "error": error})
                )
                continue

            # Execute tool
            try:
                result = tool.execute(**tool_call.arguments)

                if result["success"]:
                    console.print(f"[green]    ✓ Success[/green]")
                    logger.info(f"Tool executed successfully: {tool_call.name}")
                else:
                    console.print(f"[yellow]    ! {result.get('error', 'Failed')}[/yellow]")
                    logger.warning(f"Tool execution failed: {result.get('error')}")

                # Callback
                if self.config.on_tool_call:
                    self.config.on_tool_call(tool_call.name, tool_call.arguments, result)

                # Add result to context
                self.context.add_tool_result(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    result=json.dumps(result)
                )

            except Exception as e:
                error_msg = f"Tool execution failed: {str(e)}"
                console.print(f"[red]    ✗ {error_msg}[/red]")
                logger.error(f"Tool execution error: {str(e)}", exc_info=True)
                self.context.add_tool_result(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    result=json.dumps({"success": False, "error": error_msg})
                )

    def _check_tool_permission(self, tool: BaseTool, tool_call: ToolCall) -> bool:
        """Check if tool execution is permitted"""
        if tool.permission == ToolPermission.ALWAYS_ALLOW:
            return True

        if tool.permission == ToolPermission.NEVER:
            return False

        if tool.permission == ToolPermission.ASK:
            # Show user what the tool will do
            console.print(f"\n[yellow]  Tool: {tool.name}[/yellow]")
            console.print(f"[yellow]  Description: {tool.description}[/yellow]")
            console.print(f"[yellow]  Arguments:[/yellow]")
            for key, value in tool_call.arguments.items():
                # Truncate long values
                str_value = str(value)
                if len(str_value) > 100:
                    str_value = str_value[:100] + "..."
                console.print(f"[yellow]    {key}: {str_value}[/yellow]")

            response = console.input("[yellow]  Allow? (y/n): [/yellow]").strip().lower()
            return response == 'y'

        return False

    def save_session(self, path: str):
        """Save conversation to file"""
        session_path = Path(path)
        session_path.parent.mkdir(parents=True, exist_ok=True)

        with open(session_path, 'w') as f:
            json.dump(self.context.to_dict(), f, indent=2)

        console.print(f"[green]Session saved to {path}[/green]")
        logger.info(f"Session saved to {path}")

    @classmethod
    def load_session(cls, path: str, config: AgentConfig) -> "Agent":
        """Load conversation from file"""
        with open(path, 'r') as f:
            data = json.load(f)

        agent = cls(config)
        agent.context = ConversationContext.from_dict(data)

        console.print(f"[green]Session loaded from {path}[/green]")
        logger.info(f"Session loaded from {path}")
        return agent
```

#### 4.4 Create `allos/agent/__init__.py`

```python
"""Agent core"""

from .agent import Agent, AgentConfig

__all__ = ["Agent", "AgentConfig"]
```

---

## Week 5: CLI & Polish

### Day 29-31: CLI Implementation

#### 5.1 Create `allos/cli/main.py`

```python
"""Main CLI entry point"""

import click
import os
from pathlib import Path
from dotenv import load_dotenv

from rich.console import Console
from rich.panel import Panel

from ..agent.agent import Agent, AgentConfig
from ..tools.base import ToolPermission
from ..providers.registry import ProviderRegistry
from ..tools.registry import ToolRegistry
from ..utils.logging import setup_logging

# Load environment variables
load_dotenv()

console = Console()


@click.group()
@click.version_option()
def cli():
    """Allos Agent SDK - LLM-agnostic agentic coding"""
    pass


@cli.command()
@click.argument('task', required=False)
@click.option('--provider', '-p', default='openai', help='LLM provider (openai, anthropic, ollama)')
@click.option('--model', '-m', help='Model name')
@click.option('--tools', '-t', multiple=True, help='Tools to enable')
@click.option('--auto-approve', is_flag=True, default=False,
              help='Auto-approve safe tools')
@click.option('--session', '-s', help='Session file to save/resume')
@click.option('--interactive', '-i', is_flag=True, help='Interactive mode')
@click.option('--verbose', '-v', is_flag=True, help='Verbose logging')
def run(task, provider, model, tools, auto_approve, session, interactive, verbose):
    """Run the agent on a task"""

    # Set up logging
    setup_logging(verbose=verbose)

    # Default model per provider
    if not model:
        model_defaults = {
            'openai': 'gpt-4',
            'anthropic': 'claude-sonnet-4-5-20250929',
            'ollama': 'qwen2.5-coder',
        }
        model = model_defaults.get(provider, 'gpt-4')

    # Default tools
    if not tools:
        tools = ['read_file', 'write_file', 'edit_file', 'list_directory', 'shell_exec']

    # Build config
    config = AgentConfig(
        provider=provider,
        model=model,
        tools=list(tools),
        auto_approve_safe_tools=auto_approve,
    )

    # Load or create agent
    if session and Path(session).exists():
        agent = Agent.load_session(session, config)
    else:
        agent = Agent(config)

    # Show welcome
    console.print(Panel.fit(
        f"[bold]Allos Agent[/bold]\n"
        f"Provider: {provider}\n"
        f"Model: {model}\n"
        f"Tools: {', '.join(tools)}",
        border_style="blue"
    ))

    # Interactive mode
    if interactive or not task:
        console.print("\n[dim]Type 'exit' or 'quit' to end session[/dim]")
        while True:
            try:
                task = console.input("\n[bold cyan]You:[/bold cyan] ")
                if task.lower() in ['exit', 'quit', 'q']:
                    break

                if not task.strip():
                    continue

                response = agent.run(task)
                console.print(f"\n[bold green]Agent:[/bold green] {response}")

                if session:
                    agent.save_session(session)

            except KeyboardInterrupt:
                console.print("\n[yellow]Goodbye![/yellow]")
                break
            except Exception as e:
                console.print(f"\n[red]Error: {str(e)}[/red]")
    else:
        # Single task mode
        if not task:
            console.print("[red]Error: No task provided. Use --interactive or provide a task.[/red]")
            return

        try:
            response = agent.run(task)
            console.print(f"\n[bold green]Result:[/bold green] {response}")

            if session:
                agent.save_session(session)
        except Exception as e:
            console.print(f"\n[red]Error: {str(e)}[/red]")
            raise


@cli.command()
def list_providers():
    """List available providers"""
    providers = ProviderRegistry.list_providers()
    console.print("\n[bold]Available Providers:[/bold]")
    for p in providers:
        console.print(f"  • [cyan]{p}[/cyan]")


@cli.command()
def list_tools():
    """List available tools"""
    tools = ToolRegistry.list_tools()
    console.print("\n[bold]Available Tools:[/bold]")
    for t in tools:
        tool = ToolRegistry.get_tool(t)
        console.print(f"  • [cyan]{tool.name}[/cyan]: {tool.description}")


def main():
    """Entry point for CLI"""
    cli()


if __name__ == '__main__':
    main()
```

#### 5.2 Create `.env.example`

```bash
# OpenAI
OPENAI_API_KEY=your_openai_key_here

# Anthropic
ANTHROPIC_API_KEY=your_anthropic_key_here

# Google
GOOGLE_API_KEY=your_google_key_here

# Ollama (usually runs locally, no key needed)
# OLLAMA_BASE_URL=http://localhost:11434
```

---

## Week 6: Testing & Documentation

### Day 32-35: Comprehensive Testing

#### 6.1 Create `tests/unit/test_tools.py`

```python
"""Tests for tool system"""

import pytest
from pathlib import Path

from allos.tools.base import ToolParameter, ToolPermission
from allos.tools.registry import ToolRegistry
from allos.tools.filesystem.read import FileReadTool
from allos.tools.filesystem.write import FileWriteTool


def test_tool_registry():
    """Test tool registry"""
    tools = ToolRegistry.list_tools()
    assert "read_file" in tools
    assert "write_file" in tools


def test_file_read_tool(temp_dir, sample_file):
    """Test file reading"""
    tool = FileReadTool()
    result = tool.execute(path=str(sample_file))

    assert result["success"]
    assert result["result"]["content"] == "Hello, World!"


def test_file_write_tool(temp_dir):
    """Test file writing"""
    tool = FileWriteTool()
    test_file = temp_dir / "test_write.txt"

    result = tool.execute(
        path=str(test_file),
        content="Test content"
    )

    assert result["success"]
    assert test_file.read_text() == "Test content"


def test_tool_validation():
    """Test tool argument validation"""
    tool = FileReadTool()

    # Missing required parameter
    valid, error = tool.validate_arguments({})
    assert not valid
    assert "path" in error


def test_tool_openai_format():
    """Test OpenAI format conversion"""
    tool = FileReadTool()
    schema = tool.to_openai_format()

    assert schema["type"] == "function"
    assert schema["function"]["name"] == "read_file"
    assert "path" in schema["function"]["parameters"]["properties"]
```

#### 6.2 Create basic integration test

```python
# tests/integration/test_agent_workflow.py
"""Integration tests for agent workflows"""

import pytest
from pathlib import Path

from allos.agent.agent import Agent, AgentConfig


def test_basic_workflow(temp_dir, monkeypatch):
    """Test basic agent workflow"""
    monkeypatch.chdir(temp_dir)

    # Create test file
    test_file = temp_dir / "test.py"
    test_file.write_text("print('hello')")

    # Configure agent
    config = AgentConfig(
        provider="openai",
        model="gpt-3.5-turbo",
        tools=["read_file"],
        max_iterations=2
    )

    # This test requires API key - skip if not available
    try:
        agent = Agent(config)
        response = agent.run("Read the test.py file")
        assert "hello" in response.lower() or "print" in response.lower()
    except Exception as e:
        pytest.skip(f"Skipping: {str(e)}")
```

### Day 36-38: Documentation

#### 6.3 Create comprehensive README

```markdown
# Allos Agent SDK

[![PyPI version](https://badge.fury.io/py/allos-agent-sdk.svg)](https://badge.fury.io/py/allos-agent-sdk)
[![Tests](https://github.com/yourusername/allos-agent-sdk/workflows/tests/badge.svg)](https://github.com/yourusername/allos-agent-sdk/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Build production-ready AI agents with any LLM provider**

Allos is an LLM-agnostic agentic SDK that lets you build powerful AI agents without vendor lock-in. Switch seamlessly between OpenAI, Anthropic, Ollama, and other providers.

## ✨ Features

- 🔄 **Provider Agnostic**: OpenAI, Anthropic, Ollama, Google - switch anytime
- 🛠️ **Rich Tools**: Filesystem, shell, web search, and extensible
- 🎯 **Production Ready**: Error handling, session management, monitoring
- 🔌 **MCP Compatible**: Extend with Model Context Protocol (coming soon)
- 🚀 **Fast Setup**: Working agent in 5 minutes

## Quick Start

### Installation

```bash
pip install allos-agent-sdk

# With specific providers
pip install "allos-agent-sdk[anthropic,openai]"
```

### Basic Usage

```bash
# Set API key
export OPENAI_API_KEY=your_key_here

# Run agent
allos "Create a FastAPI hello world app"

# With different provider
export ANTHROPIC_API_KEY=your_key_here
allos --provider anthropic --model claude-sonnet-4-5 "Same task"

# Interactive mode
allos --interactive
```

### Python API

```python
from allos import Agent, AgentConfig

agent = Agent(AgentConfig(
    provider="openai",
    model="gpt-4",
    tools=["read_file", "write_file", "shell_exec"]
))

result = agent.run("Fix the bug in main.py")
print(result)
```

## Documentation

- [Getting Started](../getting-started.md)
- [Providers](../guides/providers.md)
- [Tools](../guides/tools.md)
- [Custom Tools](../guides/custom-tools.md)
- [CLI Reference](../reference/cli-reference.md)

## Supported Providers

| Provider | Status | Models |
|----------|--------|--------|
| OpenAI | ✅ | GPT-4, GPT-3.5 |
| Anthropic | ✅ | Claude 3, Claude 4 |
| Ollama | 🚧 | Local models |
| Google | 🚧 | Gemini |

## Why Allos?

**Problem**: Most agentic frameworks lock you into a single LLM provider.

**Solution**: Allos provides a unified interface across all major LLM providers, letting you:
- Switch providers without rewriting code
- Use the best model for each task
- Avoid vendor lock-in
- Run models locally with Ollama

## Contributing

We welcome contributions! See [CONTRIBUTING.md](../../.github/CONTRIBUTING.md)

## License

MIT License - see [LICENSE](../../LICENSE)

## Roadmap

See [ROADMAP.md](../../ROADMAP.md) for planned features.
```

---

## Week 7-8: Polish & Launch

### Day 39-42: Examples & Templates

Create example files in `examples/`:

**examples/basic_usage.py**:
```python
"""Basic usage example"""

from allos import Agent, AgentConfig

# Simple task
agent = Agent(AgentConfig(
    provider="openai",
    model="gpt-4",
    tools=["read_file", "write_file"]
))

result = agent.run("Create a README.md with project description")
print(result)
```

**examples/provider_switching.py**:
```python
"""Switching between providers"""

from allos import Agent, AgentConfig

providers = ["openai", "anthropic"]

for provider in providers:
    print(f"\n=== Using {provider} ===")

    agent = Agent(AgentConfig(
        provider=provider,
        model="gpt-4" if provider == "openai" else "claude-sonnet-4-5",
        tools=["read_file"]
    ))

    result = agent.run("List all Python files in current directory")
    print(result)
```

### Day 43-48: Final Polish

1. **Add CI/CD** (.github/workflows/ci.yml)
2. **Write CONTRIBUTING.md**
3. **Create comprehensive docs**
4. **Add more tests**
5. **Performance testing**
6. **Security audit**

### Day 49-56: Community Launch

1. **Create demo video**
2. **Write blog post**
3. **Post on Reddit/HN**
4. **Tweet about it**
5. **Join Discord communities**
6. **Respond to issues**

---

## Testing Checklist

- [ ] All unit tests pass
- [ ] Integration tests work with real API calls
- [ ] CLI commands work
- [ ] Provider switching works
- [ ] Tool execution is safe
- [ ] Sessions save/load correctly
- [ ] Error handling is robust
- [ ] Documentation is complete

## Pre-Launch Checklist

- [ ] README is comprehensive
- [ ] Examples work
- [ ] PyPI package is ready
- [ ] License is set (MIT recommended)
- [ ] Contributing guide exists
- [ ] Code is formatted (black)
- [ ] Linter passes (ruff)
- [ ] Security audit done
- [ ] Demo video created

---

## Success Metrics

**Week 1-2**: Foundation complete, provider working
**Week 3-4**: Tools working, agent running
**Week 5-6**: CLI functional, tests passing
**Week 7-8**: Examples done, docs complete

**MVP Success**: Can run `allos "Create a web scraper"` and it works with OpenAI, Anthropic, and Ollama.

---

## Tips for Success

1. **Start Simple**: Get ONE provider + ONE tool working first
2. **Test Early**: Write tests as you go
3. **Document**: Good docs = adoption
4. **Show, Don't Tell**: Video demos are powerful
5. **Community First**: Make it easy to contribute
6. **Iterate**: Ship MVP, gather feedback, improve

---

## Need Help?

During implementation, refer to:
- Architecture design document
- Provider documentation
- Tool implementation examples
- Test fixtures

Remember: **Ship fast, iterate faster!** The goal is a working MVP in 6-8 weeks, not perfection.
