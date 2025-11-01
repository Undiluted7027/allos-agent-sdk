# OpenAI Provider

The OpenAI provider allows you to interact with models from OpenAI, including the GPT-4 and GPT-3.5 families.

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

### OpenAI-Compatible APIs

You can also use this provider to connect to any API that is compatible with the OpenAI specification (e.g., local models served with LiteLLM or vLLM). To do this, pass the `base_url` argument during initialization.

```python
local_provider = ProviderRegistry.get_provider(
    "openai",
    model="your-local-model-name",
    base_url="http://localhost:8080/v1",
    api_key="not-needed-for-local" # Often optional for local servers
)
```

## Supported Models

The provider is compatible with all models available through the OpenAI Chat Completions and Responses API. Common models include:

- `gpt-4o`
- `gpt-4-turbo`
- `gpt-3.5-turbo`

For a full list, refer to the [official OpenAI documentation](https://platform.openai.com/docs/models).
