# Cohere Provider

The Cohere provider lets you use Cohere's native chat/tool-calling APIs through the unified Allos interface.

## Configuration

Set your API key:

```env
COHERE_API_KEY="your_cohere_api_key"
```

Install the provider extra:

```bash
uv pip install "allos-agent-sdk[cohere]"
```

## Usage

### CLI

```bash
# Simple chat
allos "In one sentence, explain embeddings." --provider cohere --model command-r7b-12-2024

# Streaming
allos --stream "Write a short haiku about testing." --provider cohere --model command-r7b-12-2024
```

> [!IMPORTANT]
> For `cohere`, provide `--model` explicitly in CLI usage.

### Agent API

```python
from allos import Agent, AgentConfig

agent = Agent(
    AgentConfig(
        provider_name="cohere",
        model="command-r7b-12-2024",
        tool_names=["read_file", "write_file"],
    )
)

result = agent.run("Create notes.txt with three bullet points about observability.")
print(result)
```

### Direct Provider Usage

```python
from allos.providers import Message, MessageRole, ProviderRegistry

provider = ProviderRegistry.get_provider("cohere", model="command-r7b-12-2024")
response = provider.chat(
    [Message(role=MessageRole.USER, content="What is retrieval-augmented generation?")],
    temperature=0,
)
print(response.content)
```

## Tools and Streaming

- Tool calling is supported through the native Cohere provider.
- Streaming is supported through `provider.stream_chat(...)` and `agent.stream_run(...)`.
- All responses are normalized into Allos `ProviderResponse` / `ProviderChunk` shapes with standard metadata.

## Native vs Compatibility Endpoint

- Use **`cohere`** for the native Cohere provider.
- Use **`cohere_compat`** when you specifically want the OpenAI Chat Completions compatibility endpoint via `chat_completions`.

For compatibility-mode details, see [Chat Completions Provider](./chat-completions.md).

## References

- [Cohere Models](https://docs.cohere.com/docs/models)
- [Cohere API Docs](https://docs.cohere.com/reference/chat)
- [Providers Guide](../guides/providers.md)
