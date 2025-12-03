# Chat Completions Provider

The `ChatCompletionsProvider` is a universal adapter designed to work with **any** API that is compatible with the OpenAI Chat Completions specification (`/v1/chat/completions`).

This provider powers the integration for services like Together AI, Groq, Mistral, DeepSeek, and local servers like vLLM.

## Supported Providers & Aliases

Allos includes an Intelligent Registry that automatically configures the correct settings for popular providers. You can use these aliases directly without manual URL configuration.

| Alias | Service | Required Env Var |
|-------|---------|------------------|
| `together` | Together AI | `TOGETHER_API_KEY` |
| `groq` | Groq | `GROQ_API_KEY` |
| `mistral` | Mistral AI | `MISTRAL_API_KEY` |
| `deepseek` | DeepSeek | `DEEPSEEK_API_KEY` |
| `cohere` | Cohere | `COHERE_API_KEY` |
| `openrouter` | OpenRouter | `OPENROUTER_API_KEY` |
| `portkey` | Portkey | `PORTKEY_API_KEY` |
| `ollama_compat` | Ollama | `No environment variable` |

## Usage

### Using Aliases (Recommended)

Simply set the environment variable and use the alias name.

**CLI:**
```bash
export GROQ_API_KEY="gsk_..."
allos "Explain LLM models" --provider groq --model llama-3.1-8b-instant
```

**Python API:**
```python
from allos import Agent, AgentConfig

config = AgentConfig(
    provider_name="groq",
    model="llama-3.1-8b-instant",
    # api_key is auto-detected from GROQ_API_KEY
)
agent = Agent(config)
```

### Manual Configuration (Custom Endpoints)

You can connect to **any** OpenAI-Chat-Completions compatible endpoint (e.g., a private vLLM instance or a new provider not yet aliased) by using the generic `chat_completions` provider name and specifying the `base_url`.

**CLI:**
```bash
allos "Hello" \
  --provider chat_completions \
  --model "my-local-model" \
  --base-url "http://localhost:8000/v1" \
  --api-key "sk-proj-..."
```

**Python API:**
```python
config = AgentConfig(
    provider_name="chat_completions",
    model="my-finetuned-model",
    base_url="https://internal-api.corp.com/v1",
    api_key="sk-internal-key"
)
agent = Agent(config)
```

## Supported Models

Because this provider is a universal adapter, **Allos supports ANY model hosted by ANY provider that adheres to the OpenAI Chat Completions API specification.**

This includes thousands of open-source models (Llama, Mistral, Qwen, Gemma) hosted on various platforms. Below are common models for the built-in aliases:

*   [**Groq:**](https://console.groq.com/docs/models) `llama-3.1-8b-instant`, `llama-3.1-70b-versatile`, `mixtral-8x7b-32768`
*   [**Together AI:**](https://docs.together.ai/docs/serverless-models#chat-models) `meta-llama/Llama-3-70b-chat-hf`, `mistralai/Mixtral-8x7B-Instruct-v0.1`, `Qwen/Qwen1.5-72B-Chat`
*   [**Mistral API:**](https://docs.mistral.ai/getting-started/models) `mistral-small-latest`, `mistral-medium-latest`, `mistral-large-latest`
*   [**Cohere:**](https://docs.cohere.com/docs/models) `command-r`, `command-r-plus`
*   [**DeepSeek:**](https://api-docs.deepseek.com/quick_start/pricing) `deepseek-chat`, `deepseek-coder`
*   [**OpenRouter:**](https://docs.openrouter.ai/models) `meta-llama/Llama-3-70b-chat-hf`, `mistralai/Mixtral-8x7B-Instruct-v0.1`, `Qwen/Qwen1.5-72B-Chat`
*   [**Portkey:**](https://docs.portkey.ai/models) `meta-llama/Llama-3-70b-chat-hf`, `mistralai/Mixtral-8x7B-Instruct-v0.1`, `Qwen/Qwen1.5-72B-Chat` [**Uses Model Catalog**](https://portkey.ai/docs/virtual_key_old/product/model-catalog)
*   [**Ollama:**](https://ollama.com/search) `llama3.1:latest`, `qwen2.5-coder:7b`, `mistral:latest`

Please refer to your specific provider's documentation for their full list of model IDs.

## Features & Limitations

This provider implements the standard OpenAI Chat Completions contract.

*   **Tools:** Supports standard function calling (`tools` parameter with `function` type).
*   **Legacy Models:** Ideal for older OpenAI models or providers that have not yet adopted the newer Responses API.
*   **Streaming:** Fully supported (*synchronous iteration currently*).

> [!TIP]
> Unsure which OpenAI provider to use? Read our **[Comparison Guide](../guides/openai-api-comparison.md)**.
