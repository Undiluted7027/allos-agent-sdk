# Ollama Provider

Allos supports running local Large Language Models (LLMs) using [Ollama](https://ollama.com). This enables completely private, offline AI agent execution where your data never leaves your machine.

## Two Ways to Use Ollama

Allos provides **two methods** to connect to Ollama:

| Provider | Description | Best For |
|----------|-------------|----------|
| `ollama` | **Native provider** using the `ollama` Python library | Full feature support, tool calling, model verification |
| `ollama_compat` | Compatibility mode via OpenAI-compatible endpoint | Quick setup, simpler use cases |

We recommend the **native `ollama` provider** for most use cases.

> [!IMPORTANT]
> The `--model` parameter is **required** when using the `ollama` provider. There is no default model - you must specify which local model to use.

## Prerequisites

1. **Install Ollama**: Follow instructions at [ollama.com](https://ollama.com).
2. **Pull a Model**: Run `ollama pull llama3.1` (or your preferred model) in your terminal.
3. **Start the Server**: Ensure `ollama serve` is running (default port is 11434).

## Native Provider (Recommended)

The native provider uses the `ollama` Python library for optimal performance and full feature support, including native tool calling.

### CLI Usage

```bash
# Basic usage with native provider
allos "Why is the sky blue?" --provider ollama --model llama3.1

# With streaming
allos --stream "Write a poem about AI" --provider ollama --model llama3.1

# Disable tools for simpler queries
allos "Explain quantum computing" --provider ollama --model mistral --no-tools
```

### Python API

```python
from allos import Agent, AgentConfig

config = AgentConfig(
    provider_name="ollama",
    model="llama3.1",
    tool_names=["read_file", "write_file"],
)

agent = Agent(config)
response = agent.run("Read the file 'config.yaml' and summarize it.")
print(response)
```

### Using the Provider Directly

```python
from allos.providers import ProviderRegistry, Message, MessageRole

# Get the native Ollama provider
provider = ProviderRegistry.get_provider("ollama", model="llama3.1")

messages = [
    Message(role=MessageRole.USER, content="What is the capital of France?")
]

response = provider.chat(messages)
print(response.content)
```

### Custom Host Configuration

By default, the provider connects to `http://localhost:11434`. You can override this:

```python
# Via environment variable
# export OLLAMA_HOST=http://192.168.1.100:11434

# Or via code
provider = ProviderRegistry.get_provider(
    "ollama",
    model="llama3.1",
    host="http://192.168.1.100:11434"
)
```

## Supported Models

The native provider includes built-in support for popular Ollama models with their context window sizes:

| Model | Context Window | Tool Calling |
|-------|---------------|--------------|
| `llama3` | 8,192 | No |
| `llama3.1` | 128,000 | **Yes** |
| `llama3.2` | 131,072 | **Yes** |
| `mistral` | 32,768 | **Yes** |
| `mixtral` | 32,768 | **Yes** |
| `qwen2` | 32,768 | **Yes** |
| `qwen2.5` | 32,768 | **Yes** |
| `qwen2.5-coder` | 32,768 | **Yes** |
| `qwen3` | 40,960 | **Yes** |
| `codellama` | 16,384 | No |
| `gemma` | 8,192 | No |
| `gemma2` | 8,192 | **Yes** |
| `deepseek-coder` | 16,384 | No |

> [!TIP]
> For agentic workflows with tool calling, use **llama3.1**, **llama3.2**, **qwen2**, **qwen2.5**, or **qwen3**.

### Dynamic Capability Detection

The Ollama provider automatically detects each model's actual capabilities (context window and tool support) by querying the Ollama server. The table above serves as a fallback when capability detection is unavailable.

To see the actual capabilities of your locally installed models, use the CLI:

```bash
allos --list-ollama-models
```

This displays a table showing each model's size, context window, tool support, and modification date.

### Model Verification

The native provider automatically verifies that your requested model is available locally. If the model isn't found, you'll receive a helpful error message:

```
ProviderError: Model 'llama3.1' not available locally. Please run `ollama pull llama3.1`.
```

## Advanced Options

The native provider supports Ollama-specific generation parameters:

```python
response = provider.chat(
    messages,
    temperature=0.7,      # Creativity (0.0-1.0)
    top_p=0.9,            # Nucleus sampling
    top_k=40,             # Top-k sampling
    seed=42,              # Reproducible outputs
    num_ctx=4096,         # Context window override
    num_predict=512,      # Max tokens to generate
    repeat_penalty=1.1,   # Penalize repetition
)
```

### Full List of Supported Options

- `temperature`, `top_p`, `top_k`, `seed`
- `num_ctx`, `num_predict`
- `repeat_penalty`, `repeat_last_n`
- `mirostat`, `mirostat_eta`, `mirostat_tau`
- `tfs_z`, `stop`

## Compatibility Mode (Alternative)

If you prefer a simpler setup or encounter issues with the native provider, you can use the OpenAI-compatible endpoint:

### CLI Usage

```bash
allos "Why is the sky blue?" --provider ollama_compat --model mistral:latest
```

### Python API

```python
config = AgentConfig(
    provider_name="ollama_compat",
    model="mistral:latest",
    # The default base_url is http://localhost:11434/v1
    # base_url="http://192.168.1.100:11434/v1"  # Custom server
)

agent = Agent(config)
response = agent.run("Hello from local AI!")
```

> [!NOTE]
> The compatibility mode has limited tool calling support compared to the native provider.

## Troubleshooting

### "Ollama not running" Error

Ensure the Ollama server is running:
```bash
ollama serve
```

### Model Not Found

Pull the model first:
```bash
ollama pull llama3.1
```

### Tool Calling Not Working

Not all models support tool calling. If you pass tools to a model that doesn't support them, you'll see a warning:

```
WARNING: Model 'gemma' does not support native tool calling. Tools will be ignored
and the agent may not perform well for tasks requiring tool use. Consider using a
tool-capable model like 'llama3.1', 'qwen2', or 'qwen3:8b', or use --no-tools flag.
```

Use one of the supported models for tool calling:
- `llama3.1`, `llama3.2`
- `qwen2`, `qwen2.5`, `qwen2.5-coder`, `qwen3`
- `mistral`, `mixtral`
- `gemma2`

Or disable tools with `--no-tools` for models that don't support them.

### Connection to Custom Host

Set the `OLLAMA_HOST` environment variable:
```bash
export OLLAMA_HOST=http://your-server:11434
```
