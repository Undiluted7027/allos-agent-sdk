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
