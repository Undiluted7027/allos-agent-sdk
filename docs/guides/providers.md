# Using LLM Providers

The core feature of the Allos SDK is its provider-agnostic design. This guide shows you how to initialize and switch between different LLM providers like OpenAI, Anthropic, Ollama, and more using a unified interface.

## The `ProviderRegistry`

The `ProviderRegistry` is your single entry point for accessing any supported LLM provider. You don't need to import provider-specific classes; you just ask the registry for what you need by name.

```python
from allos.providers import ProviderRegistry

# Get an instance of the OpenAI provider
openai_provider = ProviderRegistry.get_provider("openai", model="gpt-4o")

# Get an instance of the Anthropic provider
anthropic_provider = ProviderRegistry.get_provider(
    "anthropic", model="claude-3-5-sonnet-20240620"
)
```

The first argument is the provider's unique name (e.g., `"openai"`), and all other keyword arguments are passed directly to the provider's constructor. The `model` argument is always required.

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
- [Ollama with Chat Completions](./../providers/ollama.md)

## The `ProviderResponse` Object

The `.chat()` method always returns a standardized `ProviderResponse` object, regardless of the provider used.

```python
from allos.providers import ProviderResponse

response = openai_provider.chat(messages)

# response is a ProviderResponse object with these attributes:
# response.content: The string content of the model's text response.
# response.tool_calls: A list of requested tool calls (for Phase 3).
# response.metadata: A dictionary with provider-specific call details.

print(response.metadata)
```

This ensures that the rest of your application can handle LLM outputs in a consistent and predictable way.
