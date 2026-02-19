# Using LLM Providers

The core feature of the Allos SDK is its provider-agnostic design. This guide shows you how to initialize and switch between different LLM providers like OpenAI, Anthropic, Cohere, Ollama, Google (Gemini/Vertex AI) and more using a unified interface.

## The `ProviderRegistry`

The `ProviderRegistry` is your single entry point for accessing any supported LLM provider. You don't need to import provider-specific classes; you just ask the registry for what you need by name.

```python
from allos.providers import ProviderRegistry

# Get an instance of the OpenAI provider
openai_provider = ProviderRegistry.get_provider("openai", model="gpt-4o")

# Get an instance of the Anthropic provider
anthropic_provider = ProviderRegistry.get_provider(
    "anthropic", model="claude-sonnet-4-5-20250929"
)

# Get an instance of the Google provider (Gemini API)
google_provider = ProviderRegistry.get_provider(
    "google", model="gemini-2.0-flash"
)

# Get an instance of the Google provider (Vertex AI)
vertex_provider = ProviderRegistry.get_provider(
    "google",
    model="gemini-2.0-flash",
    vertexai=True,
    project="my-gcp-project"
)

# Get an instance of the Cohere provider (native)
cohere_provider = ProviderRegistry.get_provider(
    "cohere", model="command-r7b-12-2024"
)
```

The first argument is the provider's unique name (e.g., `"openai"`), and all other keyword arguments are passed directly to the provider's constructor. The `model` argument is always required.

## Supported Providers

| Provider | Name | Best For | Python Version |
|----------|------|----------|----------------|
| OpenAI | `openai` | GPT models, production-ready | 3.9+ |
| Anthropic | `anthropic` | Claude models, reasoning | 3.9+ |
| Cohere | `cohere` | Native Cohere chat + tool calling | 3.9+ |
| Google Gemini | `google` | Gemini models, multimodal | **3.10+** |
| Ollama | `ollama` | Local models, privacy | 3.9+ |
| Groq | `groq` | Fast inference | 3.9+ |
| Chat Completions | `chat_completions` | Any OpenAI-compatible API | 3.9+ |

For detailed information on each provider, see:
- [OpenAI Provider](../providers/openai.md)
- [Anthropic Provider](../providers/anthropic.md)
- [Cohere Provider](../providers/cohere.md)
- [Google Provider](../providers/google.md)
- [Ollama Provider](../providers/ollama.md)
- [Chat Completions Provider](../providers/chat-completions.md)

## Universal Compatibility (OpenAI-Compatible APIs)

Allos supports **any** provider that is compatible with the OpenAI Chat Completions API (e.g., Together AI, Groq, Mistral, LocalAI).

We provide intelligent aliases for popular services that automatically configure the correct API endpoints.

```python
# Use Groq (Fast Inference)
# Auto-detects GROQ_API_KEY environment variable
groq_provider = ProviderRegistry.get_provider("groq", model="llama-3.1-8b-instant")

# Use Together AI
# Auto-detects TOGETHER_API_KEY environment variable
together_provider = ProviderRegistry.get_provider("together", model="meta-llama/Llama-3-70b-chat-hf")
```

### Manual Configuration

If you are using a custom endpoint (like a private vLLM instance) that isn't in our alias list, you can use the generic `chat_completions` provider and specify the `base_url`.

```python
custom_provider = ProviderRegistry.get_provider(
    "chat_completions",
    model="my-local-model",
    base_url="http://localhost:8000/v1",
    api_key="sk-internal-key" # Optional if env var is set
)
```

For more details on compatible providers, see the [Chat Completions Provider](../providers/chat-completions.md) reference.

## The `.chat()` Method

All provider instances returned by the registry share the same public interface. The primary method is `.chat()`, which takes a list of `Message` objects.

```python
from allos.providers import Message, MessageRole

messages = [
    Message(role=MessageRole.USER, content="Tell me a joke about programming.")
]

# The call is identical for both providers
openai_response = openai_provider.chat(messages)
anthropic_response = anthropic_provider.chat(messages)

print(f"OpenAI says: {openai_response.content}")
print(f"Anthropic says: {anthropic_response.content}")
```

## Provider-Specific Configuration

You can pass provider-specific arguments during initialization. For example, if you need to connect to an OpenAI-compatible API endpoint (like a local LLM server):

```python
custom_provider = ProviderRegistry.get_provider(
    "openai",
    model="local-model",
    base_url="http://localhost:8080/v1"
)
```

For more details on each provider, see the dedicated documentation pages:
- [OpenAI](./../providers/openai.md)
- [Anthropic](./../providers/anthropic.md)
- [ChatCompletions](./../providers/chat-completions.md)
- [Ollama (Native & Compatibility)](./../providers/ollama.md)

## Local Models with Ollama

For completely private, local AI execution, Allos provides a native Ollama provider:

```python
# Native Ollama provider with tool calling support
ollama_provider = ProviderRegistry.get_provider("ollama", model="llama3.1")

# Or use the OpenAI-compatible endpoint
ollama_compat = ProviderRegistry.get_provider("ollama_compat", model="mistral")
```

> [!NOTE]
> The `model` parameter is required for the `ollama` provider. See the [Ollama Provider](./../providers/ollama.md) documentation for supported models and tool calling capabilities.

> [!IMPORTANT]
> **Ollama-Exclusive Features**: The native `ollama` provider includes advanced features not available in other providers:
> - **Model Warm-Up Detection**: Automatic detection and logging of model loading time (10-30s on first request)
> - **Streaming Retry Logic**: Automatic retry with exponential backoff for interrupted streams
> - **Dynamic Capability Detection**: Runtime detection of model context windows and tool support
>
> These features are unique to Ollama and enhance reliability for local model usage. See the [Ollama Provider documentation](./../providers/ollama.md) for details.

> [!TIP]
> Use `allos --list-ollama-models` to see all locally available models with their context windows and tool support status.
>
> For a comprehensive guide comparing native vs compat providers, see **[examples/ollama_usage.py](../../examples/ollama_usage.py)**.

## The `ProviderResponse` Object

The `.chat()` method always returns a standardized `ProviderResponse` object, regardless of the provider used.

```python
from allos.providers import ProviderResponse

response = openai_provider.chat(messages)

# response is a ProviderResponse object with these attributes:
# response.content: The string content of the model's text response.
# response.tool_calls: A list of requested tool calls.
# response.metadata: Metadata object containing metadata of response.

print(response.metadata)
```

This ensures that the rest of your application can handle LLM outputs in a consistent and predictable way.
