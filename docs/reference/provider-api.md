# Provider API Reference

This document provides a technical reference for the classes and data structures that form the provider abstraction layer in the Allos SDK.

## `BaseProvider`

This is the abstract base class that all provider implementations must inherit from.

**`allos.providers.base.BaseProvider(model: str, **kwargs)`**

### Abstract Methods

#### `chat(messages: List[Message], **kwargs) -> ProviderResponse`
This method must be implemented by all providers. It takes a list of `Message` objects and returns a `ProviderResponse`. Provider-specific options (like `tools` or `temperature`) can be passed as keyword arguments.

#### `get_context_window() -> int`
This method must be implemented by all providers. It should return an integer representing the maximum context window size (in tokens) for the configured model.

---

## Data Classes

These dataclasses are used to ensure a consistent data structure across all providers.

### `Message`

Represents a single turn in a conversation.

**`allos.providers.base.Message`**
- `role: MessageRole` - The role of the message sender.
- `content: Optional[str]` - The text content of the message.
- `tool_calls: List[ToolCall]` - A list of tool calls requested by the assistant in this turn.
- `tool_call_id: Optional[str]` - For `TOOL` role messages, the ID of the tool call this is a response to.

### `MessageRole`

An Enum for the different roles in a conversation.

**`allos.providers.base.MessageRole`**
- `SYSTEM: "system"`
- `USER: "user"`
- `ASSISTANT: "assistant"`
- `TOOL: "tool"`

### `ToolCall`

Represents a tool call requested by an LLM.

**`allos.providers.base.ToolCall`**
- `id: str` - A unique identifier for this specific tool call.
- `name: str` - The name of the tool to be called.
- `arguments: dict[str, Any]` - A dictionary of the parsed arguments for the tool.

### `ProviderResponse`

The standardized object returned by every provider's `.chat()` method.

**`allos.providers.base.ProviderResponse`**
- `content: Optional[str]` - The text content of the model's response.
- `tool_calls: List[ToolCall]` - A list of any tool calls requested by the model.
- `metadata: dict[str, Any]` - A dictionary containing provider-specific information and processing metrics from the API call.
