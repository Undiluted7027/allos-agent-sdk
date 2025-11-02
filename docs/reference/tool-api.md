# Tool API Reference

This document provides a technical reference for the classes and data structures that form the tool abstraction layer in the Allos SDK.

## `BaseTool`

The abstract base class for all tool implementations.

**`allos.tools.base.BaseTool`**

### Class Attributes
- `name: str`: The unique, machine-readable name for the tool.
- `description: str`: A detailed description of what the tool does, for the LLM to understand.
- `parameters: List[ToolParameter]`: A list of `ToolParameter` objects defining the tool's inputs.
- `permission: ToolPermission`: The default security permission level for the tool.

### Abstract Methods
#### `execute(**kwargs: Any) -> Dict[str, Any]`
The core logic of the tool. It must be implemented by all subclasses. It receives arguments as a dictionary and must return a JSON-serializable dictionary.

### Methods
#### `to_provider_format(provider: str) -> Dict[str, Any]`
Converts the tool's definition into the format required by a specific LLM provider (e.g., "openai" or "anthropic").

---

## `@tool` Decorator

A class decorator that automatically registers any class inheriting from `BaseTool` with the `ToolRegistry`.

**`allos.tools.registry.tool(cls: Type[BaseTool]) -> Type[BaseTool]`**

---

## Data Classes & Enums

### `ToolParameter`
Defines an input parameter for a tool.

**`allos.tools.base.ToolParameter`**
- `name: str`
- `type: str` (e.g., "string", "integer", "boolean")
- `description: str`
- `required: bool`

### `ToolPermission`
An Enum defining the security levels for tool execution.

**`allos.tools.base.ToolPermission`**
- `ALWAYS_ALLOW: "always_allow"`
- `ALWAYS_DENY: "always_deny"`
- `ASK_USER: "ask_user"`

### `ToolCall` (from [Provider API](./provider-api.md))

When your agent runs, it will receive `ToolCall` objects when the LLM decides to use a tool. It's important to understand the `id` field.

**`allos.providers.base.ToolCall`**
- `id: str` - The **correlation ID** for the tool call. This ID is essential for linking the tool request to the tool result. When you add a tool result back to the agent's context, you must use this ID.
- `name: str` - The name of the tool.
- `arguments: Dict[str, Any]` - The arguments for the tool.

For more technical details on how this ID maps to different provider-specific IDs (like OpenAI's `call_id`), please see the [Provider API Reference](./provider-api.md).
