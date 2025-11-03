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
- `tool_call_id: Optional[str]` - For `TOOL` role messages, this holds the correlation ID of the tool call this message is a result for. **This should match the `id` of the corresponding `ToolCall` object.**

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
- `id: str` - The **correlation ID** for the tool call. This is the primary identifier used to link a tool call request from the assistant to a tool result message from the user. For providers like OpenAI's Responses API, this corresponds to the `call_id` (e.g., `"call_..."`).
- `name: str` - The name of the tool to be called.
- `arguments: Dict[str, Any]` - A dictionary of the parsed arguments for the tool.

> [!NOTE] A note on IDs and call_id
Different providers have different ways of identifying tool calls. The Allos SDK abstracts this into a single `id` field on the `ToolCall` object.
>
>   -   **For most providers (like Anthropic):** The `id` is a straightforward unique identifier for the tool use block.
>    -   **For OpenAI's Responses API:** The API uses two types of IDs:
        -   An **item ID** (e.g., `"fc_..."`) that uniquely identifies the `function_call` object in the response list.
        -   A **correlation ID** or `call_id` (e.g., `"call_..."`) that is used to link the call to its eventual result.
>
>    **The `ToolCall.id` in the Allos SDK always stores the correlation ID (`call_id`).** Our `OpenAIProvider` handles the translation and management of the item IDs (`fc_...`) internally, so you only need to work with the correlation ID when building custom tools or inspecting the agent's context.

### `ProviderResponse`

The standardized object returned by every provider's `.chat()` method.

**`allos.providers.base.ProviderResponse`**
- `content: Optional[str]` - The text content of the model's response.
- `tool_calls: List[ToolCall]` - A list of any tool calls requested by the model.
- `metadata: dict[str, Any]` - A dictionary containing provider-specific information and processing metrics from the API call.
