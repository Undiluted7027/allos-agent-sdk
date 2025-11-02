# allos/context/manager.py

"""
Manages the conversation context, which includes the history of messages.
"""

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from ..providers.base import Message, MessageRole, ToolCall


@dataclass
class ConversationContext:
    """
    A container for the messages in a conversation, serving as the agent's memory.

    This class provides methods to add messages, and to serialize/deserialize
    the entire conversation history for session management.
    """

    messages: List[Message] = field(default_factory=list)
    provider_state: Dict[str, Any] = field(default_factory=dict)

    def add_system_message(self, content: str) -> None:
        """Adds a system message to the context."""
        self.messages.append(Message(role=MessageRole.SYSTEM, content=content))

    def add_user_message(self, content: str) -> None:
        """Adds a user message to the context."""
        self.messages.append(Message(role=MessageRole.USER, content=content))

    def add_assistant_message(
        self, content: Optional[str], tool_calls: Optional[List[ToolCall]] = None
    ) -> None:
        """Adds an assistant message, which may contain text and/or tool calls."""
        self.messages.append(
            Message(
                role=MessageRole.ASSISTANT,
                content=content,
                tool_calls=tool_calls or [],
            )
        )

    def add_tool_result_message(self, tool_call_id: str, content: str) -> None:
        """Adds the result of a tool execution to the context."""
        self.messages.append(
            Message(
                role=MessageRole.TOOL,
                tool_call_id=tool_call_id,
                content=content,
            )
        )

    def get_provider_state(self, key: str, default: Any = None) -> Any:
        """Get provider-specific state."""
        return self.provider_state.get(key, default)

    def set_provider_state(self, key: str, value: Any) -> None:
        """Set provider-specific state."""
        self.provider_state[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the entire conversation context to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationContext":
        """Deserializes a dictionary into a ConversationContext instance."""
        context = cls()
        for msg_data in data.get("messages", []):
            # Convert role string back to MessageRole enum
            msg_data["role"] = MessageRole(msg_data["role"])

            # Reconstruct ToolCall objects if they exist
            tool_calls_data = msg_data.get("tool_calls", [])
            if tool_calls_data:
                msg_data["tool_calls"] = [ToolCall(**tc) for tc in tool_calls_data]

            context.messages.append(Message(**msg_data))

        # Load provider state (but don't restore response_id - it's invalid)
        context.provider_state = data.get("provider_state", {})
        # Clear response_id on load since it's from a previous session
        context.provider_state.pop("response_id", None)
        return context

    def to_json(self, **kwargs) -> str:
        """Serializes the context to a JSON string."""
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_json(cls, json_str: str) -> "ConversationContext":
        """Deserializes a JSON string into a ConversationContext instance."""
        return cls.from_dict(json.loads(json_str))

    @property
    def is_empty(self) -> bool:
        """Returns True if the context has no messages."""
        return not self.messages

    def __len__(self) -> int:
        """Returns the number of messages in the context."""
        return len(self.messages)
