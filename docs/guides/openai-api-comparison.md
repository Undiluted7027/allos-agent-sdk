# Guide: OpenAI Responses API vs. Chat Completions API

Allos supports two distinct ways to interact with OpenAI-compatible models. Understanding the difference helps you choose the right provider configuration for your needs.

## At a Glance

| Feature | **Responses API** (`openai`) | **Chat Completions API** (`chat_completions`) |
| :--- | :--- | :--- |
| **Target Service** | Official OpenAI (`api.openai.com`) | OpenAI-Compatible Ecosystem (Together, Groq, LocalAI) |
| **Tool Calling** | Native, Multi-turn optimization | Standard Function Calling |
| **State** | Stateful (Server-side context possible) | Stateless (Client manages full history) |
| **Input Format** | List of "Items" (Text, FunctionCall, Result) | List of "Messages" (System, User, Assistant, Tool) |
| **Recommended For** | `gpt-5`, `gpt-4-turbo` on OpenAI | Open-source models, Local models, Legacy OpenAI |

## Deep Dive

### 1. The Modern Path: Responses API (`provider="openai"`)

This is OpenAI's latest API standard. It treats conversation history as a flat list of "items" rather than rigid message roles.

**Advantages:**
*   **Better Tooling:** The model handles multi-step tool use more robustly.
*   **Future Proof:** Aligns with OpenAI's roadmap for agents.
*   **Simplicity:** Allos handles the complex mapping of `call_id` and item types for you.

**Usage:**
```bash
allos --provider openai --model gpt-4o "..."
```

### 2. The Compatibility Path: Chat Completions API (`provider="chat_completions"`)

This uses the legacy `/v1/chat/completions` endpoint standard. Because this standard was open and widely adopted, it is the *de facto* language of the LLM ecosystem.

**Advantages:**
*   **Universal:** Works with almost every LLM provider (Groq, Together, Mistral, DeepSeek).
*   **Local Models:** Works with Ollama, vLLM, and LM Studio.
*   **Portable:** Easy to switch endpoints just by changing the `base_url`.

**Usage:**
```bash
# Using an alias (automatically sets base_url)
allos --provider groq --model llama-3.1-8b-instant "..."

# Manual configuration
allos --provider chat_completions --base_url http://localhost:1234/v1 "..."
```

## Technical Implementation in Allos

The Allos SDK abstracts these differences away from you. You interact with a unified `Message` object, and the SDK translates it:

*   **Responses Adapter:** Converts `Message(role=TOOL)` into `{"type": "function_call_output", "call_id": ...}`.
*   **Chat Completions Adapter:** Converts `Message(role=TOOL)` into `{"role": "tool", "tool_call_id": ...}`.

## Recommendation

*   **Use `openai`** if you are paying for official OpenAI models. It offers the richest feature set. See implementation details in [openai provider](../providers/openai.md).
*   **Use `chat_completions` (or aliases)** for everything else. It is the universal key to the open-source and local AI world. See implementation details in [chat completions provider](../providers/chat-completions.md).
