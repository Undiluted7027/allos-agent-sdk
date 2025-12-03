# OpenAI Provider

The OpenAI provider allows you to interact with models from OpenAI, including the GPT-5 and GPT-4 families.

It uses OpenAI's modern **Responses API** (`/v1/responses`) to provide advanced capabilities like native multi-turn tool calling and stateful interactions.

## Configuration

To use this provider, you must set the `OPENAI_API_KEY` environment variable.

```env
# In your .env file
OPENAI_API_KEY="sk-..."
```

## Usage

Use the `ProviderRegistry` to get an instance of the OpenAI provider.

```python
from allos.providers import ProviderRegistry

openai_provider = ProviderRegistry.get_provider("openai", model="gpt-4o")
```

> [!TIP]
> For a detailed comparison between this provider and the compatible provider, see the **[API Comparison Guide](../guides/openai-api-comparison.md)**.

### OpenAI-Compatible APIs

While the `openai` provider handles the official API, many 3rd-party services (like Together AI, Groq, or LocalAI) use the older **Chat Completions API**.

For these services, we recommend using the **[Chat Completions Provider](./chat-completions.md)** (`provider="chat_completions"` or a specific alias like `groq`).

However, if you have an endpoint that strictly implements the new Responses API, you can configure the `openai` provider manually:

```python
custom_provider = ProviderRegistry.get_provider(
    "openai",
    model="custom-model",
    base_url="https://my-responses-api-clone.com/v1"
)
```

## Supported Models

The provider is compatible with all models available through the OpenAI Chat Completions and Responses API. Common models include:

- `gpt-5`
- `gpt-4`
- `gpt-4o`
- `gpt-4-turbo`
- `gpt-3.5-turbo`

For a full list, refer to the [official OpenAI documentation](https://platform.openai.com/docs/models).
