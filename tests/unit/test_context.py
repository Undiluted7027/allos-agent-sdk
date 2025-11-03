# tests/unit/test_context.py


from allos.context.manager import ConversationContext
from allos.providers.base import MessageRole, ToolCall


class TestConversationContext:
    """Tests for the ConversationContext class."""

    def test_initialization(self):
        """Test that a new context is empty."""
        context = ConversationContext()
        assert context.is_empty
        assert len(context) == 0
        assert context.messages == []

    def test_add_messages(self):
        """Test adding messages of each role."""
        context = ConversationContext()

        context.add_system_message("You are a helpful assistant.")
        assert len(context) == 1
        assert context.messages[-1].role == MessageRole.SYSTEM

        context.add_user_message("Hello!")
        assert len(context) == 2
        assert context.messages[-1].role == MessageRole.USER

        context.add_assistant_message("Hi there!")
        assert len(context) == 3
        assert context.messages[-1].role == MessageRole.ASSISTANT
        assert context.messages[-1].content == "Hi there!"

        tool_calls = [ToolCall("1", "tool_name", {"arg": "value"})]
        context.add_assistant_message(None, tool_calls=tool_calls)
        assert len(context) == 4
        assert context.messages[-1].role == MessageRole.ASSISTANT
        assert context.messages[-1].tool_calls[0].name == "tool_name"

        context.add_tool_result_message("1", '{"status": "ok"}')
        assert len(context) == 5
        assert context.messages[-1].role == MessageRole.TOOL
        assert context.messages[-1].tool_call_id == "1"

    def test_serialization_to_dict(self):
        """Test serializing the context to a dictionary."""
        context = ConversationContext()
        context.add_system_message("System prompt")
        context.add_user_message("User message")
        tool_calls = [ToolCall("id123", "search", {"query": "allos sdk"})]
        context.add_assistant_message(None, tool_calls=tool_calls)

        data = context.to_dict()

        assert isinstance(data, dict)
        assert "messages" in data
        assert len(data["messages"]) == 3

        # Check that enums and objects are converted to primitives
        assert data["messages"][0]["role"] == "system"
        assert isinstance(data["messages"][2]["tool_calls"][0], dict)
        assert data["messages"][2]["tool_calls"][0]["name"] == "search"

    def test_deserialization_from_dict(self):
        """Test deserializing the context from a dictionary."""
        raw_data = {
            "messages": [
                {
                    "role": "system",
                    "content": "System prompt",
                    "tool_calls": [],
                    "tool_call_id": None,
                },
                {
                    "role": "user",
                    "content": "User message",
                    "tool_calls": [],
                    "tool_call_id": None,
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "id123",
                            "name": "search",
                            "arguments": {"query": "allos sdk"},
                        }
                    ],
                    "tool_call_id": None,
                },
            ]
        }

        context = ConversationContext.from_dict(raw_data)

        assert isinstance(context, ConversationContext)
        assert len(context) == 3

        # Check that primitives are converted back to objects
        assert context.messages[0].role == MessageRole.SYSTEM
        assert isinstance(context.messages[2].tool_calls[0], ToolCall)
        assert context.messages[2].tool_calls[0].name == "search"
        assert context.messages[2].tool_calls[0].arguments == {"query": "allos sdk"}

    def test_json_round_trip(self):
        """Test a full serialization to and from JSON."""
        original_context = ConversationContext()
        original_context.add_system_message("System prompt")
        original_context.add_user_message("User message")
        tool_calls = [ToolCall("id123", "search", {"query": "allos sdk"})]
        original_context.add_assistant_message("Thinking...", tool_calls=tool_calls)
        original_context.add_tool_result_message("id123", '{"results": "found"}')

        # Serialize to JSON
        json_str = original_context.to_json()
        assert isinstance(json_str, str)

        # Deserialize back from JSON
        rehydrated_context = ConversationContext.from_json(json_str)

        # The rehydrated context should be identical to the original
        assert isinstance(rehydrated_context, ConversationContext)
        assert original_context.messages == rehydrated_context.messages
