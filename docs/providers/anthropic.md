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
from allos import Agent, AgentConfig

config = AgentConfig(
    provider_name="anthropic",
    model="claude-sonnet-4-5-20250929"
)

agent = Agent(config)
```

## Supported Models

The provider is compatible with all models available through the Anthropic Messages API. Common models include:

- `claude-opus-4-6`
- `claude-sonnet-4-5-20250929`
- `claude-haiku-4-5-20251001`
- `claude-opus-4-5-20251101`

and others...

For a full list, refer to the [official Anthropic documentation](https://docs.anthropic.com/claude/reference/models).
