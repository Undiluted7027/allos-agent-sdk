# Configuration Guide

Allos is designed to work out-of-the-box with minimal setup, but it offers powerful configuration options for advanced users.

You can configure Allos using **Environment Variables** (best for secrets) and **CLI Flags** (best for runtime overrides).

## Environment Variables

We follow a standard naming convention for API keys. Set these in your shell or a `.env` file.

| Provider | Variable Name |
|----------|---------------|
| **OpenAI** | `OPENAI_API_KEY` |
| **Anthropic** | `ANTHROPIC_API_KEY` |
| **Together AI** | `TOGETHER_API_KEY` |
| **Groq** | `GROQ_API_KEY` |
| **Mistral** | `MISTRAL_API_KEY` |
| **DeepSeek** | `DEEPSEEK_API_KEY` |
| **Cohere** | `COHERE_API_KEY` |
| **OpenRouter** | `OPENROUTER_API_KEY` |
| **Portkey** | `PORTKEY_API_KEY` |

You can check which variables are correctly detected by running:
```bash
allos --active-providers
```

## CLI Configuration Flags

When using the `allos` command, you can override any default setting using flags. These flags take precedence over environment variables and session files.

### Provider Settings
*   `--provider, -p <name>`: Select the LLM provider (e.g., `openai`, `groq`).
*   `--model, -m <name>`: Select the specific model ID.
*   `--base-url <url>`: Override the API endpoint URL (useful for local models or proxies).
*   `--api-key <key>`: Explicitly pass an API key (overrides env vars).

### Agent Behavior
*   `--max-tokens <int>`: Limit the number of tokens the model generates.
*   `--no-tools`: Disable all tools. Useful for "chat-only" mode or for models that hallucinate tool calls.
*   `--auto-approve`: skip permission prompts for tool execution. **Use with caution.**

### Session Management
*   `--session, -s <file.json>`: Load agent state from a file. The state includes conversation history. New state is saved back to this file after the run.

### Example: Fully Configured Run

```bash
allos "Analyze this log file" \
  --provider chat_completions \
  --base-url "http://localhost:8080/v1" \
  --model "local-model" \
  --max-tokens 500 \
  --no-tools \
  --session "analysis.json"
```

> [!NOTE] Coming Soon: File-Based Configuration
Support for YAML/JSON configuration files (.allos/config.yaml) is planned for Phase 2.5 of the [roadmap](../../POST_MVP_ROADMAP.md). This will allow for persistent, project-level settings beyond environment variables.
