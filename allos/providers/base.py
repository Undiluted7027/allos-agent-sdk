# allos/providers/base.py

"""Base classes and data structures for all LLM providers.

This module defines the abstract interface that all provider implementations must follow,
ensuring they are interchangeable within the Allos ecosystem.
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .metadata import Metadata


class MessageRole(str, Enum):
    """Enumeration for the roles in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ToolCall:
    """Represents a tool call requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    """Represents a single message in a conversation.

    Attributes:
        role: The role of the message sender (system, user, assistant, or tool).
        content: The text content of the message. Can be None for tool calls.
        tool_calls: A list of tool calls requested by the assistant.
        tool_call_id: The ID of the tool call this message is a response to (for role='tool').
        thought_signatures: Maps tool_call_id to signature (Gemini 3)
    """

    role: MessageRole
    content: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    thought_signatures: Optional[Dict[str, bytes]] = None


@dataclass
class ProviderResponse:
    """Standardized response from a provider."""

    metadata: Metadata
    content: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    thought_signatures: Optional[Dict[str, bytes]] = None
    # metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderChunk:
    """Represents a single chunk of data yielded from a streaming provider.

    Only one of the fields should be populated per chunk.
    """

    content: Optional[str] = None
    tool_call_start: Optional[Dict[str, Any]] = (
        None  # e.g., {"id": "call_123", "name": "get_weather", "index": 0}
    )
    tool_call_delta: Optional[str] = None  # e.g., '{"location": "S'
    tool_call_done: Optional[ToolCall] = None  # The fully formed ToolCall object
    thought_signatures: Optional[Dict[str, bytes]] = (
        None  # Google Gemini 3 thought signatures
    )
    # usage: Optional[Dict[str, Any]] = None  # e.g., {"input_tokens": 10, ...}
    final_metadata: Optional[Metadata] = None
    error: Optional[str] = None


class BaseProvider(ABC):
    """Abstract base class for all LLM providers.

    All provider implementations must inherit from this class and implement
    the `chat` and `get_context_window` methods.
    """

    env_var: Optional[str] = None

    def __init__(self, model: str, **kwargs: Any):
        """Initializes the provider.

        Args:
            model: The specific model to use (e.g., 'gpt-4', 'claude-3-opus-20240229').
            **kwargs: Provider-specific arguments (e.g., api_key, base_url).
        """
        self.model = model
        self.provider_specific_kwargs = kwargs

    @classmethod
    def check_env_config(cls) -> Tuple[bool, str]:
        """Check if environment is properly configured for this provider.

        Override this method in subclasses that need complex env var logic
        (e.g., multiple valid configurations, OR/AND combinations).

        Returns:
            Tuple of (is_configured, display_message) where:
            - is_configured: True if the provider can be used
            - display_message: Human-readable status for CLI display
            Examples: "OPENAI_API_KEY (Set)", "GOOGLE_API_KEY (Not Set)",
                       "Vertex AI (PROJECT=Set, LOCATION=Set)"
        """
        if cls.env_var is None:
            return (True, "N/A")
        if cls.env_var in os.environ:
            return (True, f"{cls.env_var} (Set)")
        return (False, f"{cls.env_var} (Not Set)")

    @abstractmethod
    def chat(self, messages: List[Message], **kwargs: Any) -> ProviderResponse:
        """Sends a list of messages to the LLM and gets a response.

        This method must be implemented by all subclasses.

        Args:
            messages: A list of Message objects representing the conversation history.
            **kwargs: Additional provider-specific parameters for the API call
                      (e.g., temperature, max_tokens, tools).

        Returns:
            A ProviderResponse object containing the LLM's reply.
        """
        raise NotImplementedError

    @abstractmethod
    def stream_chat(
        self, messages: List[Message], **kwargs: Any
    ) -> Iterator[ProviderChunk]:
        """Sends a list of messages to the LLM and streams the response.

        Args:
            messages: A list of Message objects representing the conversation history.
            **kwargs: Additional provider-specific parameters for the API call.

        Yields:
            ProviderChunk: An iterator of chunks representing the streaming response.
        """
        raise NotImplementedError

    @abstractmethod
    def get_context_window(self) -> int:
        """Retrieves the context window size for the provider's configured model.

        This method must be implemented by subclasses to report the maximum
        number of tokens the model can handle in a single context.

        Returns:
            An integer representing the maximum number of tokens for the model's
            context window.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        """Provides a formal, unambiguous string representation of the provider instance.

        Returns:
            A string in the format 'ClassName(model='model_name')'.
        """
        return f"{self.__class__.__name__}(model='{self.model}')"
