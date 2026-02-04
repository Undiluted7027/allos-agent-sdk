# Ollama Provider

Allos supports running local Large Language Models (LLMs) using [Ollama](https://ollama.com). This enables completely private, offline AI agent execution where your data never leaves your machine.

## Two Ways to Use Ollama

Allos provides **two methods** to connect to Ollama:

| Provider | Description | Best For |
|----------|-------------|----------|
| `ollama` | **Native provider** using the `ollama` Python library | Full feature support, tool calling, model verification |
| `ollama_compat` | Compatibility mode via OpenAI-compatible endpoint | Quick setup, simpler use cases |

We recommend the **native `ollama` provider** for most use cases.

> [!TIP]
> See **[examples/ollama_usage.py](../../examples/ollama_usage.py)** for a comprehensive guide comparing both providers with streaming, tool calling, and decision criteria.

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

### Model Capability Matrix

This comprehensive comparison helps you choose the right model for your use case:

| Model Family | Best For | Tool Calling | Context Window | Typical Size | Performance |
|--------------|----------|--------------|----------------|--------------|-------------|
| **llama3.1** | General-purpose agents, long context | ✅ Yes | 128K | 4.7-8GB | Fast |
| **llama3.2** | Latest features, long context | ✅ Yes | 131K | 2-7GB | Fast |
| **qwen3** | Multilingual, balanced performance | ✅ Yes | 41K | 4.9GB | Very Fast |
| **qwen2.5** | Code generation, technical tasks | ✅ Yes | 33K | 3.8-7.6GB | Very Fast |
| **qwen2.5-coder** | Advanced code tasks | ✅ Yes | 33K | 4.5-8.5GB | Fast |
| **mistral** | Fast inference, good reasoning | ✅ Yes | 33K | 4.1GB | Very Fast |
| **mixtral** | High quality, MOE architecture | ✅ Yes | 33K | 26GB | Medium |
| **gemma2** | Efficient, good performance | ✅ Yes | 8K | 5.4-9GB | Fast |
| **llama3** | Legacy, simple tasks | ❌ No | 8K | 4.7GB | Fast |
| **codellama** | Code-only, no agent support | ❌ No | 16K | 3.8-13GB | Fast |
| **gemma** | Legacy, chat-only | ❌ No | 8K | 5GB | Fast |
| **deepseek-coder** | Code-only, no agent support | ❌ No | 16K | 6.7-33GB | Medium |

### Tool Calling Support by Model

> [!IMPORTANT]
> Tool calling is **essential** for using models as agents. Models without tool calling support can only work in chat-only mode (`--no-tools`).

#### ✅ Models with Native Tool Calling Support

These models support the Ollama tool calling API and work seamlessly with the Allos agent:

**Meta Llama Family:**
- `llama3.1` - Best all-around choice with 128K context
- `llama3.2` - Latest version with 131K context
- `llama3.2:1b`, `llama3.2:3b` - Lightweight versions for faster inference

**Qwen Family (Recommended for Code):**
- `qwen3` - Latest, excellent multilingual support
- `qwen3:8b`, `qwen3:14b` - Various sizes available
- `qwen2.5` - Stable, reliable choice
- `qwen2.5-coder` - Specialized for code generation
- `qwen2` - Legacy but stable

**Mistral Family:**
- `mistral` - Fast inference, good reasoning
- `mixtral` - Mixture of Experts, high quality (requires more resources)
- `mistral-nemo` - Efficient 12B parameter model

**Google Gemma:**
- `gemma2` - Efficient with good performance
- `gemma2:2b`, `gemma2:9b`, `gemma2:27b` - Multiple sizes

#### ❌ Models WITHOUT Tool Calling Support

These models can only be used in chat-only mode. Pass `--no-tools` or set `no_tools=True` in config:

**Legacy Models:**
- `llama3` - Use `llama3.1` or `llama3.2` instead
- `gemma` - Use `gemma2` instead

**Code-Specialized Models:**
- `codellama` - Designed for code completion, not agents
- `deepseek-coder` - Code-focused, no tool API

### Model Selection Guide

Choose your model based on your requirements:

#### For Production Agents
**Recommended:** `llama3.1`, `qwen3`, or `qwen2.5`
- Native tool calling support
- Large context windows (32K-128K)
- Good balance of quality and speed
- Well-tested in production

#### For Code Generation Tasks
**Recommended:** `qwen2.5-coder` or `qwen3`
- Specialized code understanding
- Native tool calling for file operations
- Fast inference

#### For Resource-Constrained Systems
**Recommended:** `qwen3:8b`, `llama3.2:3b`, or `gemma2:2b`
- Smaller model sizes (2-5GB)
- Still support tool calling
- Faster inference on limited hardware

#### For Maximum Context Length
**Recommended:** `llama3.2` (131K) or `llama3.1` (128K)
- Longest context windows available
- Handle large documents or long conversations
- Full tool calling support

#### For Simple Chat (No Agent Features)
**Acceptable:** `llama3`, `codellama`, `gemma`, `deepseek-coder`
- Use with `--no-tools` flag
- Limited to text responses
- No file operations or shell commands

### Performance Characteristics

Based on typical 8-16GB RAM systems:

| Model | Load Time (First Request) | Inference Speed | Memory Usage |
|-------|---------------------------|-----------------|--------------|
| `qwen3:8b` | 10-15s | ~30 tokens/s | 4.9GB |
| `llama3.2:3b` | 8-12s | ~40 tokens/s | 2GB |
| `llama3.1` | 12-18s | ~25 tokens/s | 4.7GB |
| `mistral` | 10-15s | ~35 tokens/s | 4.1GB |
| `qwen2.5-coder` | 12-18s | ~28 tokens/s | 4.5GB |
| `gemma2:9b` | 12-16s | ~25 tokens/s | 5.4GB |
| `mixtral` | 20-30s | ~15 tokens/s | 26GB |

> [!NOTE]
> Performance varies based on:
> - Hardware (GPU vs CPU, VRAM/RAM available)
> - Quantization level (Q4, Q5, Q8)
> - System load and other running processes

### Dynamic Capability Detection

The Ollama provider automatically detects each model's actual capabilities (context window and tool support) by querying the Ollama server. The table above serves as a fallback when capability detection is unavailable.

To see the actual capabilities of your locally installed models, use the CLI:

```bash
allos --list-ollama-models
```

This displays a table showing each model's size, context window, tool support, and modification date.

**Example Output:**
```
           Ollama Models (http://localhost:11434)
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━┓
┃ Model            ┃   Size ┃ Context ┃ Tools ┃ Modified   ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━┩
│ llama3.1:latest  │ 4.9 GB │    131K │ Yes   │ 2025-11-19 │
│ qwen3:8b         │ 5.2 GB │     40K │ Yes   │ 2026-01-29 │
│ gemma:latest     │ 5.0 GB │      8K │ No    │ 2025-10-15 │
└──────────────────┴────────┴─────────┴───────┴────────────┘
```

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

## Model Warm-Up Detection

> [!IMPORTANT]
> Model warm-up detection is a **unique feature of the Ollama provider** and is not available for other providers.

When you make the first request to an Ollama model, the model must be loaded into memory (GPU/CPU VRAM). This can take **10-30 seconds** depending on model size and hardware. Subsequent requests are much faster as the model stays loaded in memory.

### How It Works

The native Ollama provider automatically detects warm-up events and:
1. **Tracks** the first request to each model+host combination
2. **Measures** if the first request takes ≥10 seconds
3. **Logs** an informative message to help you understand the delay
4. **Exposes** warm-up information in the response metadata

### Accessing Warm-Up Information

#### Via Provider Response

```python
from allos.providers import ProviderRegistry, Message, MessageRole

provider = ProviderRegistry.get_provider("ollama", model="llama3.1")
messages = [Message(role=MessageRole.USER, content="Hello")]

response = provider.chat(messages)

# Check if warm-up was detected
if response.metadata.provider_specific.ollama and response.metadata.provider_specific.ollama.warm_up:
    duration = response.metadata.provider_specific.ollama.warm_up_duration_seconds
    print(f"⚠️  Model warm-up detected: {duration:.1f}s")
    print("   The model was loading into memory.")
    print("   Subsequent requests will be much faster.")
else:
    print("✓ Model already loaded (fast response)")
```

#### Via Agent Metadata

```python
from allos import Agent, AgentConfig

config = AgentConfig(provider_name="ollama", model="llama3.1")
agent = Agent(config)

response = agent.run("Hello, world!")

# Access metadata from the last run
meta = agent.last_run_metadata
if meta and meta.provider_specific.ollama:
    if meta.provider_specific.ollama.warm_up:
        print(f"Warm-up detected: {meta.provider_specific.ollama.warm_up_duration_seconds:.1f}s")
```

### Console Logging

When warm-up is detected, the SDK automatically logs an informative message:

```
INFO: Model 'llama3.1' first request took 12.5s (loading into memory).
      Subsequent requests will be faster.
```

### Why This Matters

Understanding warm-up helps you:
- **Diagnose** why the first request is slow
- **Optimize** workflows by pre-loading models
- **Set expectations** for users in production systems
- **Monitor** model loading times in logs

For more details on warm-up metadata, see the [Metadata Reference](../reference/metadata.md#providerspecificollama).

---

## Streaming Retry Logic

> [!IMPORTANT]
> Automatic streaming retry is a **unique feature of the Ollama provider** and is not available for other providers.

The native Ollama provider includes intelligent retry logic for streaming requests. If a stream is interrupted due to network issues or connection resets, it will automatically retry with exponential backoff.

### Retry Behavior

| Retry Type | When It Retries | When It Fails Fast |
|------------|-----------------|-------------------|
| **Network Errors** | Connection resets, timeouts, network failures | API errors, invalid requests |
| **Backoff Strategy** | Exponential: 1s → 2s → 4s → 8s... | Immediate failure for non-retryable errors |
| **Max Attempts** | 3 retries by default (configurable) | N/A |

### Configuration

You can configure retry behavior when creating the provider:

```python
from allos.providers.ollama import OllamaProvider

# Configure custom retry settings
provider = OllamaProvider(
    model="llama3.1",
    max_stream_retries=5,  # Allow up to 5 retry attempts (default: 3)
)

# Use the provider
for chunk in provider.stream_chat(messages):
    if chunk.content:
        print(chunk.content, end="")
    if chunk.error:
        print(f"\nStream error: {chunk.error}")
        break
```

### How It Works

1. **Connection Monitoring**: The provider monitors the streaming connection
2. **Failure Detection**: Detects connection resets, timeouts, and network errors
3. **Automatic Retry**: Retries the request with exponential backoff
4. **Error Reporting**: Returns error chunks for non-retryable failures

### Example: Handling Retries

```python
from allos import Agent, AgentConfig

config = AgentConfig(
    provider_name="ollama",
    model="llama3.1",
)

agent = Agent(config)

# Stream with automatic retry on connection failures
for chunk in agent.stream_run("Write a poem about AI"):
    if chunk.content:
        print(chunk.content, end="", flush=True)
    if chunk.error:
        print(f"\n❌ Stream failed: {chunk.error}")
        break
```

### Why This Matters

Streaming retry makes Ollama more reliable for:
- **Unstable Networks**: Auto-recovers from temporary network hiccups
- **Long-Running Streams**: Handles interruptions in lengthy responses
- **Production Systems**: Reduces manual error handling code
- **User Experience**: Seamless recovery without user intervention

### Retry Logs

When retries occur, you'll see log messages:

```
WARNING: Ollama stream attempt 1 failed: Connection reset. Retrying in 1.0s...
WARNING: Ollama stream attempt 2 failed: Connection reset. Retrying in 2.0s...
INFO: Ollama stream succeeded on attempt 3.
```

---

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

## Examples

Allos provides several examples demonstrating Ollama usage:

- **[examples/ollama_usage.py](../../examples/ollama_usage.py)** - **Comprehensive guide** covering:
  - Basic usage (native vs compat)
  - Provider comparison
  - Streaming examples
  - Tool calling capabilities
  - Decision guide for choosing providers

- **[examples/local_models.py](../../examples/local_models.py)** - Privacy-focused example using native provider with tools
- **[examples/ollama_compat.py](../../examples/ollama_compat.py)** - Simple compat provider usage
- **[examples/local_vs_cloud.py](../../examples/local_vs_cloud.py)** - Performance comparison between local and cloud providers

Run any example with:
```bash
python examples/ollama_usage.py
```

## Best Practices

1. **Choose the right provider**:
   - Use `ollama` for production with tools
   - Use `ollama_compat` for simple queries

2. **Model selection**:
   - Use tool-capable models (llama3.1, qwen2, mistral) for agents
   - Disable tools with `--no-tools` for non-supporting models

3. **Performance**:
   - Expect 10-30s warm-up on first request (model loading)
   - Keep models loaded by making regular requests
   - Local models are free but slower than cloud APIs
   - Use smaller models for faster responses
   - Consider quantized versions for better performance

4. **Reliability**:
   - Use native provider for automatic streaming retry
   - Configure `max_stream_retries` for unstable networks
   - Monitor warm-up logs to diagnose slow first requests

5. **Privacy**:
   - Local models keep all data on your machine
   - Perfect for sensitive data or offline work
   - No API costs or rate limits

## Ollama-Specific Features

The native Ollama provider includes features not available in other providers:

| Feature | Description | Documentation |
|---------|-------------|---------------|
| **Model Warm-Up Detection** | Automatic detection and logging of model loading time | [See above](#model-warm-up-detection) |
| **Streaming Retry Logic** | Automatic retry with exponential backoff for streaming | [See above](#streaming-retry-logic) |
| **Dynamic Capability Detection** | Automatic detection of model context windows and tool support | [CLI: `allos --list-ollama-models`](#dynamic-capability-detection) |
| **Model Verification** | Validates models are available before making requests | [See above](#model-verification) |
| **Connection Pooling** | Efficient HTTP connection reuse for better performance | Automatic |

> [!NOTE]
> These features are exclusive to the native `ollama` provider. The `ollama_compat` provider uses the OpenAI client and does not include these enhancements.
