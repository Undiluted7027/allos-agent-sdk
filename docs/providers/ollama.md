# Ollama Provider

Allos supports running local Large Language Models (LLMs) using [Ollama](https://ollama.com).

Currently, support is provided via the **Compatibility Layer**. A native Ollama provider (utilizing the `ollama` Python library for advanced features like model pulling) is planned for the next release phase.

## Usage (Compatibility Mode)

You can use Ollama immediately by using the `ollama_compat` provider alias. This connects to your local Ollama server using the OpenAI-compatible endpoint.

### Prerequisites

1.  **Install Ollama**: Follow instructions at [ollama.com](https://ollama.com).
2.  **Pull a Model**: Run `ollama pull mistral` (or your preferred model) in your terminal.
3.  **Start the Server**: Ensure `ollama serve` is running (default port is 11434).

### CLI Usage

```bash
# Basic usage
allos "Why is the sky blue?" --provider ollama_compat --model mistral:latest

# Using a specific model (e.g., Llama 3)
allos "Write a poem" --provider ollama_compat --model llama3
```

> [!TIP]
> Smaller local models (like Mistral 7B) often struggle with complex tool definitions. If you encounter errors or hallucinations, try running with the **`--no-tools`** flag to disable tool calling:
>
> `allos "Explain quantum computing" --provider ollama_compat --model mistral --no-tools`

### Python API

```python
from allos import Agent, AgentConfig

config = AgentConfig(
    provider_name="ollama_compat",
    model="mistral:latest",
    # The default base_url is http://localhost:11434/v1
    # You can override it if your server is elsewhere:
    # base_url="http://192.168.1.100:11434/v1"
)

agent = Agent(config)
response = agent.run("Hello from local AI!")
print(response)
```

> [!NOTE] Documentation In Progress
This page is currently under development and will be populated as the corresponding feature is implemented in the Allos SDK. <br><br>
Our development is guided by our public roadmap. Please see the [Post MVP Roadmap](./../../POST_MVP_ROADMAP.md) for near-term features and the [Full Roadmap](./../../ROADMAP.md) for our long-term vision.<br><br>
Thank you for your interest and patience!
