# Anthropic Provider

The Anthropic provider allows you to interact with the Claude family of models.

## Configuration

To use this provider, you must set the `ANTHROPIC_API_KEY` environment variable.

```env
# In your .env file
ANTHROPIC_API_KEY="sk-ant-..."
```

## Usage

Use the `ProviderRegistry` to get an instance of the Anthropic provider.

```python
from allos.providers import ProviderRegistry

anthropic_provider = ProviderRegistry.get_provider(
    "anthropic", model="claude-3-5-sonnet-20240620"
)
```

## Supported Models

The provider is compatible with all models available through the Anthropic Messages API. Common models include:

- `claude-3-5-sonnet-20240620`
- `claude-3-opus-20240229`
- `claude-3-sonnet-20240229`
- `claude-3-haiku-20240307`

For a full list, refer to the [official Anthropic documentation](https://docs.anthropic.com/claude/reference/models).
