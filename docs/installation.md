# Installation

This guide will walk you through installing the Allos Agent SDK. We recommend using `uv` for the fastest and most reliable experience.

## Prerequisites

- Python 3.9+
- We recommend installing `uv`, a next-generation Python package manager.

### Installing `uv`

If you don't have `uv` installed, you can install it with one of the following commands:

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (Powershell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

For other installation methods, see the [official `uv` documentation](https://docs.astral.sh/uv/getting-started/installation/).


## Installing the Allos SDK

You can install the core SDK and add optional dependencies for the specific LLM providers you want to use.

### Basic Installation

This installs the core SDK without any specific provider libraries.

```bash
uv pip install allos-agent-sdk
```

### Installation with Providers

To use a specific provider, you can install it as an "extra".

> [!NOTE] Universal Compatibility
> Installing the `openai` extra enables the **Chat Completions Provider**, which allows you to connect to **Together AI, Groq, Mistral, DeepSeek, LocalAI, vLLM**, and any other OpenAI-compatible service.

This is the recommended approach.

```bash
# Install with support for OpenAI
uv pip install "allos-agent-sdk[openai]"

# Install with support for Anthropic
uv pip install "allos-agent-sdk[anthropic]"

# Install with support for all currently available providers
uv pip install "allos-agent-sdk[all]"
```

### Development Installation

If you plan to contribute to the Allos SDK, please see the [Development Setup Guide](./contributing/development.md).
