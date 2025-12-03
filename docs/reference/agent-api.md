# Agent API Reference

This document provides a technical reference for the core `Agent` class, its configuration, and the context management classes.

## `AgentConfig`

A dataclass used to configure an `Agent` instance.

**`allos.agent.AgentConfig`**
- `provider_name: str`: The name of the LLM provider to use (e.g., `"openai"`, `"anthropic"`, `"chat_completions"`, `"groq"`).
- `model: str`: The specific model name for the chosen provider (e.g., `"gpt-4o"`, `"llama-3.1-8b-instant"`).
- `tool_names: List[str]`: A list of names of the tools the agent is allowed to use. These names must correspond to tools registered in the `ToolRegistry`.
- `max_iterations: int`: The maximum number of LLM-tool interaction loops the agent can perform before stopping. Defaults to `10`.
- `max_tokens: Optional[int]`: The maximum number of tokens the model is allowed to generate. Defaults to `None`.
- `no_tools: bool`: If `True`, the agent will be initialized without any tools, even if `tool_names` is provided. Useful for chat-only interactions or constrained models. Defaults to `False`.
- `base_url: Optional[str]`: Override the API endpoint URL. Useful for connecting to custom OpenAI-compatible servers.
- `api_key: Optional[str]`: Explicitly provide an API key. This overrides any environment variables. **Security Note:** This field is excluded from the string representation (`repr`) and is removed before session serialization to prevent leaks.

---

## `Agent`

The main class for orchestrating agentic behavior.

**`allos.agent.Agent(config: AgentConfig, context: Optional[ConversationContext] = None)`**

### Methods

#### `run(prompt: str) -> str`
Starts the agentic loop with a given user prompt. The agent will interact with its provider and tools until it generates a final text-based answer or reaches `max_iterations`.
- **Arguments:**
    - `prompt` (str): The high-level task or question for the agent.
- **Returns:**
    - `str`: The final text response from the agent.
- **Raises:**
    - `AllosError`: If `max_iterations` is reached.

#### `save_session(filepath: str | Path) -> None`
Serializes the agent's `AgentConfig` and its entire `ConversationContext` to a JSON file.
> [!NOTE] Security
> The `api_key` field is automatically stripped from the configuration before saving.

- **Arguments:**
    - `filepath` (str | Path): The path to save the session file to.
- **Raises:**
    - `AllosError`: If saving fails due to an `IOError` or `TypeError`.

#### `load_session(filepath: str | Path) -> "Agent"`
A class method that deserializes an agent session from a JSON file and returns a new, fully configured `Agent` instance with the restored conversation history.
- **Arguments:**
    - `filepath` (str | Path): The path to the session file.
- **Returns:**
    - `Agent`: A new `Agent` instance.
- **Raises:**
    - `AllosError`: If loading fails due to `FileNotFoundError`, `json.JSONDecodeError`, or `KeyError`.

---

## `ConversationContext`

A class that holds the state of a conversation. It is managed internally by the `Agent` but can also be instantiated and passed to an agent for more advanced control.

**`allos.context.ConversationContext`**

### Methods
- `add_system_message(content: str)`
- `add_user_message(content: str)`
- `add_assistant_message(content: Optional[str], tool_calls: Optional[List[ToolCall]] = None)`
- `add_tool_result_message(tool_call_id: str, content: str)`

### Serialization
- `to_dict() -> Dict[str, Any]`
- `from_dict(data: Dict[str, Any]) -> "ConversationContext"`
- `to_json() -> str`
- `from_json(json_str: str) -> "ConversationContext"`

For details on the `Message` and `ToolCall` objects stored within the context, please see the [Provider API Reference](./provider-api.md).
