# allos/providers/base.py
"""Base classes and data structures for LLM providers. This is just to satisfy type checkers and Linters for conftest and testing infrastructure."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class MessageRole(str, Enum):
    """Enumeration for the roles in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """Represents a single message in a conversation."""

    role: MessageRole
    content: str


@dataclass
class ToolCall:
    """Represents a tool call requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ProviderResponse:
    """Standardized response from a provider."""

    content: Optional[str] = None
    tool_calls: list[ToolCall] = field(default_factory=list)


class BaseProvider(ABC):
    """Abstract base class for all LLM providers."""

    @abstractmethod
    def chat(self, messages: list[Message]) -> ProviderResponse:
        """Sends a list of messages to the provider and gets a response."""
        raise NotImplementedError
