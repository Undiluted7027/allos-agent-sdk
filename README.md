# Allos Agent SDK

<div align="center">

**🚀 The LLM-Agnostic Agentic Framework**

*Build powerful AI agents without vendor lock-in*

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Post MVP Phase 2 Active](https://img.shields.io/badge/status-Phase%202%20Active-blue.svg)](./POST_MVP_ROADMAP.md)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](./.github/CONTRIBUTING.md)
[![codecov](https://codecov.io/gh/Undiluted7027/allos-agent-sdk/graph/badge.svg?token=DUZU3HSZDL)](https://codecov.io/gh/Undiluted7027/allos-agent-sdk)
[![PyPI version](https://img.shields.io/pypi/v/allos-agent-sdk.svg)](https://pypi.org/project/allos-agent-sdk/)
[![GitHub release](https://img.shields.io/github/v/release/Undiluted7027/allos-agent-sdk)](https://github.com/Undiluted7027/allos-agent-sdk/releases)
[![Build](https://github.com/Undiluted7027/allos-agent-sdk/actions/workflows/ci.yml/badge.svg)](https://github.com/Undiluted7027/allos-agent-sdk/actions/workflows/ci.yml)



[Documentation](./docs) • [Roadmap](./ROADMAP.md) • [Contributing](./.github/CONTRIBUTING.md)

</div>

---

## 🎯 What is Allos?

Allos is an open-source, provider-agnostic agentic SDK that gives you the power to build production-ready AI agents that work with **any LLM provider**. Inspired by Anthropic's Claude Code, Allos delivers the same outstanding capabilities without locking you into a single ecosystem.

**The Problem**: Most agentic frameworks force you to choose between vendors, making it expensive and risky to switch models.

**The Solution**: Allos provides a one interface across OpenAI, Anthropic, Ollama, Google (Gemini/Vertex AI), Cohere, Groq, Mistral, and more—so you can use the best model for each task without rewriting your code.

## ✨ Key Features

### 🔄 **Provider Agnostic**
Connect to **10+ providers** out of the box. Switch seamlessly between OpenAI and Anthropic natively, or connect to Groq, Together AI, Mistral, Deepseek, and local models via our compatibility layer. Use GPT-5 for one task, Claude for another, or run models locally—all with the same code.

### 🛠️ **Rich Tool Ecosystem**
Built-in tools for:
- 📁 File operations (read, write, edit)
- 💻 Shell command execution
- 🌐 Web search and fetching *(coming soon)*
- 🔌 MCP (Model Context Protocol) extensibility *(coming soon)*

### 🎛️ **Advanced Capabilities**
- ⚡ **Context Management**: Automatic context window optimization
- 🔐 **Fine-grained Permissions**: Control what your agent can and cannot do
- 💾 **Session Management**: Save and resume conversations
- 📊 **Production Ready**: Built-in error handling, logging, and monitoring
- 🎨 **Extensible**: Easy to add custom tools and providers

### 🚀 **Developer Experience**
```bash
# Create your own Claude Code in 5 minutes
uv pip install allos-agent-sdk
export OPENAI_API_KEY=your_key

# Check what providers are ready to use
allos --active-providers
allos "Create a REST API for a todo app"
```

## 🆚 Why Allos?

| Feature | Allos | Anthropic Agent SDK | LangChain Agents |
|---------|-------|-------------------|------------------|
| Provider Agnostic | ✅ | ❌ (Anthropic only) | ⚠️ (Complex) |
| Local Models Support | ✅ | ❌ | ⚠️ |
| Simple API | ✅ | ✅ | ❌ |
| Built-in Tools | ✅ | ✅ | ⚠️ |
| MCP Support | 🚧 | ✅ | ❌ |
| Production Ready | ✅ | ✅ | ⚠️ |
| Open Source | ✅ MIT | ⚠️ Limited | ✅ |

## 🚀 Quick Start

See the full workflow in action by running our CLI demo script:
```bash
bash <(curl -s https://raw.githubusercontent.com/Undiluted7027/allos-agent-sdk/main/examples/cli_workflow.sh)
```

> [!NOTE] Python Support
> Core SDK supports Python 3.9+.
>
> Google provider support requires Python 3.10+.

### Installation

We recommend using `uv`, a fast Python package manager.

```bash
# Basic installation
uv pip install allos-agent-sdk

# With specific providers
uv pip install "allos-agent-sdk[openai]"
uv pip install "allos-agent-sdk[anthropic]"
uv pip install "allos-agent-sdk[cohere]"
uv pip install "allos-agent-sdk[all]"  # All providers

# With Google/Gemini support (requires Python 3.10+)
uv pip install "allos-agent-sdk[google]"
```

### CLI Usage

The `allos` CLI is the quickest way to use the agent.

```bash
# Set your API key (or use a .env file)
export OPENAI_API_KEY="your_key_here"

# Run a single task
allos "Create a FastAPI hello world app in a file named main.py and then run it."

# Run in streaming mode to see output instantly
allos --stream "Write a long story about space exploration."

# Start an interactive session for a conversation
allos -i
# >>> Create a file named 'app.py' with a simple Flask app.
# >>> Now, add a route to it that returns the current time.

# Switch providers and save your session
export ANTHROPIC_API_KEY="your_key_here"
allos -p anthropic -s my_project.json "Refactor the 'app.py' file to be more modular."
```

### Python API

```python
from allos import Agent, AgentConfig

# Configure for Together AI (using the Universal Adapter)
agent = Agent(
    AgentConfig(
        provider_name="together",
        model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        tool_names=["read_file", "write_file"],
    )
)

# Run the agent
result = agent.run("Fix the bug in main.py and add tests")
print(result)

# Access detailed metrics
print(f"Cost: ${agent.last_run_metadata.usage.estimated_cost.total_usd}")
```

### Provider Switching Example

```python
# Start with OpenAI
agent_openai = Agent(AgentConfig(
    provider_name="openai",
    model="gpt-4",
    tool_names=["read_file", "write_file"]
))

# Switch to Anthropic for complex reasoning
agent_claude = Agent(AgentConfig(
    provider_name="anthropic",
    model="claude-sonnet-4-5",
    tool_names=["read_file", "write_file"]
))

# Switch to Groq for fast responses
agent_groq = Agent(AgentConfig(
    provider_name="groq",
    model="llama-3.1-8b-instant",
))

# Or use local models with native Ollama support
agent_local = Agent(AgentConfig(
    provider_name="ollama",
    model="llama3.1",
    tool_names=["read_file", "write_file"]
))

# Same interface, different providers!
result = agent_openai.run("Create a FastAPI app")
```

### Custom Tools

```python
from allos.tools import BaseTool, tool, ToolParameter

@tool
class DatabaseQueryTool(BaseTool):
    name = "query_database"
    description = "Execute SQL queries"
    parameters = [
        ToolParameter(
            name="query",
            type="string",
            description="SQL query to execute",
            required=True
        )
    ]

    def execute(self, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        query = kwargs.get("query")
        if not query:
            return {"success": False, "error": "Query parameter is required."}
        # Your implementation
        # In a real scenario, you would connect to a DB.
        # result = your_db.execute(query)
        # For this example, we'll return a mock result.
        return {"status": "success", "result": f"Query '{query}' executed."}

# Use it
agent = Agent(AgentConfig(
    provider_name="openai",
    model="gpt-4o",
    tool_names=["query_database", "read_file"]
))
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      CLI Layer                          │
│              (User-friendly interface)                  │
└─────────────────────────┬───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│                   Agent Core                            │
│        (Orchestration & Agentic Loop)                   │
└─────┬──────────────────┬──────────────────┬────────────-┘
      │                  │                  │
┌─────▼────────┐  ┌──────▼───────┐  ┌───────▼──────┐
│  Providers   │  │    Tools     │  │   Context    │
│              │  │              │  │              │
│ • OpenAI     │  │ • FileSystem │  │ • History    │
│ • Anthropic  │  │ • Shell      │  │ • Compactor  │
│ • Groq       │  │ • Web        │  │ • Cache      │
│ • Mistral    │  │ • Custom     │  │ • Manager    │
│ • Ollama     │  │              │  │              │
| • Google     |  |              |  |              |
└──────────────┘  └──────────────┘  └──────────────┘
```

### Core Components

1. **Provider Layer**: Unified interface for all LLM providers
2. **Tool System**: Extensible toolkit with built-in and custom tools
3. **Agent Core**: Main agentic loop with planning and execution
4. **Context Manager**: Automatic context window optimization
5. **CLI**: User-friendly command-line interface

## 📊 Provider Support

Allos supports a massive range of models through native integrations and a universal compatibility layer.

| Provider | Status | Models |
|----------|--------|--------|
| **OpenAI** | ✅ Ready | GPT-4o, GPT-5, GPT-5.2 |
| **Anthropic** | ✅ Ready | Claude 4.5 Sonnet, Haiku |
| **Ollama** | ✅ Ready | Native support - Llama, Mistral, etc. |
| **Groq** | ✅ Ready | Llama 3, Mixtral |
| **Mistral** | ✅ Ready | Mistral Large, Small |
| **Together AI** | ✅ Ready | Llama 3, Qwen, DeepSeek |
| **DeepSeek** | ✅ Ready | DeepSeek Chat/Coder |
| **Cohere (compat)** | ✅ Ready | Command R+ |
| **Cohere native** | ✅ Ready | Command R+, Command A |
| **Ollama (compat)** | ✅ Ready | OpenAI-compatible endpoint mode |
| **Ollama native** | ✅ Ready | Native Ollama support |
| **LocalAI / vLLM** | ✅ Ready | Custom endpoints via `chat_completions` |
| **Google** | ✅ Ready | Gemini-3.0 Pro, Gemini-2.5 Flash |

## 🛠️ Built-in Tools

| Tool | Description | Permission |
|------|-------------|------------|
| `read_file` | Read file contents | Always Allow |
| `write_file` | Write/create files | Ask User |
| `edit_file` | Edit files (string replace) | Ask User |
| `list_directory` | List directory contents | Always Allow |
| `shell_exec` | Execute shell commands | Ask User |
| `web_search` | Search the web | 📋 Planned |
| `web_fetch` | Fetch web page content | 📋 Planned |

## 🎯 Use Cases

### Coding Agents
```python
# SRE Agent - Diagnose and fix production issues (Web Search COMING SOON!)
sre_agent = Agent(AgentConfig(
    provider_name="anthropic",
    model="claude-4-opus",
    tool_names=["read_file", "shell_exec"]
))
sre_agent.run("Investigate why the API latency spiked at 3pm")

# Code Review Agent
review_agent = Agent(AgentConfig(
    provider_name="openai",
    model="gpt-4",
    tool_names=["read_file", "write_file"]
))
review_agent.run("Review PR #123 for security issues and best practices")
```

### Business Automation
```python
# Data Analysis Agent
data_agent = Agent(AgentConfig(
    provider_name="openai",
    model="gpt-4",
    tool_names=["read_file", "write_file", "query_database"]
))
data_agent.run("Analyze Q4 sales data and create a summary report")

# Content Creation Agent (Web Search COMING SOON!)
content_agent = Agent(AgentConfig(
    provider_name="anthropic",
    model="claude-sonnet-4-5",
    tool_names=["web_search", "read_file", "write_file"]
))
content_agent.run("Research AI trends and write a blog post")
```

## 📚 Documentation

- **[Getting Started](./docs/getting-started.md)** - Installation and first steps
- **[Quickstart Guide](./docs/guides/quickstart.md)** - 5-minute tutorial
- **[Providers](./docs/guides/providers.md)** - Provider configuration
- **[Chat Completions](./docs/providers/chat-completions.md)** - Universal compatibility guide
- **[Tools](./docs/guides/tools.md)** - Using built-in tools
- **[Custom Tools](./docs/guides/custom-tools.md)** - Creating your own tools
- **[CLI Reference](./docs/reference/cli-reference.md)** - Command-line options
- **[API Reference](./docs/reference/agent-api.md)** - Python API documentation
<!-- - **[Architecture](./docs/reference/initial-architecture-design.md)** - System design -->

## 🗺️ Roadmap

### ✅ Phase 1: MVP (Completed)
- Core architecture, OpenAI/Anthropic support, CLI, Tools.

### ✅ Phase 2: Feature 2.0 (Completed)
- [x] **Universal Compatibility**: Support for Groq, Mistral, Together, etc.
- [x] **Streaming**: Real-time token streaming.
- [x] **Observability**: Standardized metadata schema.
- [x] **CLI DX**: Active providers check, config flags.

### 🏗️ Phase 2: Upcoming
- [x] **Native Ollama**: Advanced local model support with tool calling.
- [x] Google Gemini support
- [ ] Web search and fetch tools
- [ ] Advanced context management
- [ ] Plugin system
- [ ] Configuration files (YAML/JSON)
- [ ] Session management improvements

### 🔮 Phase 3: Advanced Capabilities
- [ ] MCP (Model Context Protocol) support
- [ ] Subagents and delegation
- [ ] Pydantic AI integration
- [ ] Smolagents compatibility
- [ ] Multi-modal support
- [ ] Advanced monitoring and observability
- [ ] Cloud deployment support

See [POST_MVP_ROADMAP.md](./POST_MVP_ROADMAP.md) for detailed timeline.

## 🚧 Known Limitations

While Allos has matured significantly beyond the MVP, some advanced features are still in development:

-   **Limited Context Management:** Basic context window checks are implemented, but advanced context compaction and summarization for very long conversations are planned.
-   **No Async Support:** The core `Agent` and `Tool` classes are synchronous. An async-first version is planned for a future release.
-   **Additional Provider Support:** While we support 10+ providers, native support for Azure OpenAI, and AWS Bedrock are planned for future releases.
-   **No Web Tools:** Built-in tools for web search (`web_search`) and fetching URLs (`web_fetch`) are planned but not yet implemented.
-   **Basic Error Recovery:** While the agent can recover from tool execution errors (like permission denied), it does not yet have sophisticated strategies for retrying failed API calls or self-correcting flawed plans.

Please see our full [ROADMAP.md](./ROADMAP.md) for more details on our plans for these and other features.

## 🚦 Current Status

**🟢 Stable Release (v0.0.1)**

All major features for the MVP and Phase 2.0 are implemented and tested.
- ✅ **Providers:** 10+ providers supported.
- ✅ **Streaming:** Full streaming support.
- ✅ **Tools:** Secure filesystem and shell tools included.
- ✅ **Agent Core:** Robust agentic loop with permissions.
- ✅ **CLI:** Polished CLI with interactive mode.

🎉 **MVP Released**

We welcome early contributors! See [Contributing](#-contributing) below.

## 🤝 Contributing

We're building Allos in the open and would love your help! Whether you're:

- 🐛 **Reporting bugs**
- 💡 **Suggesting features**
- 📖 **Improving documentation**
- 🔧 **Submitting PRs**
- ⭐ **Starring the repo** (helps a lot!)

All contributions are welcome! See [CONTRIBUTING.md](./.github/CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/Undiluted7027/allos-agent-sdk.git
cd allos-agent-sdk
```

For better DX, if you are on Linux/MacOS/WSL2, consider:
```bash
chmod +x scripts/setup_dev.sh
./scripts/setup_dev.sh
```

#### Python Environment

##### With `pip`

```bash
# Create virtual environment
python -m venv venv
# For: Mac OS/Linux
source venv/bin/activate
# On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[all, dev]"

# Make the test script executable
chmod +x scripts/run_tests.sh

# Run the default test suite (unit + e2e, no API keys required)
./scripts/run_tests.sh

# Run ONLY integration tests (requires API keys in a .env file)
uv run pytest --run-integration

# Format code
black allos tests
ruff check allos tests --fix
```

#### With `uv`

Ensure you have `uv` installed. Check out [UV Installation Instructions](https://docs.astral.sh/uv/getting-started/installation/) for more information.

```bash
# Create virtual environment
uv venv

# Activate environment
# For: MacOS/Linux
source .venv/bin/activate
# For: Windows (Powershell)
# .venv\Scripts\activate

# Install in development mode
uv pip install -e ".[dev]"

# Make the test script executable
chmod +x scripts/run_tests.sh

# Run the default test suite (unit + e2e, no API keys required)
./scripts/run_tests.sh

# Run ONLY integration tests (requires API keys in a .env file)
uv run pytest --run-integration

# Format code
black allos tests
ruff check allos tests --fix
```

## 🌟 Stargazers Hall of Fame

A huge thank you to our first 100 stargazers! You're helping build the future of AI agent development. 🚀

<div align="center">

<!-- STARGAZERS:START -->
*No stargazers yet. Be the first!* ⭐
<!-- STARGAZERS:END -->

</div>

**Not featured yet?** [⭐ Star us on GitHub](https://github.com/Undiluted7027/allos-agent-sdk) to join the Hall of Fame!

---

## 🌟 Why "Allos"?

**Allos** (Greek: ἄλλος) means "other" or "different" - representing our core philosophy of **choice and flexibility**. Just as the word implies alternatives and options, Allos gives you the freedom to choose any LLM provider without constraints.

## 📄 License

Allos is open source and available under the [MIT License](./LICENSE).

## 🙏 Acknowledgments

Inspired by:
- [Anthropic's Claude Code](https://www.claude.com/product/claude-code) - For showing what's possible with agentic coding
- [LangChain](https://github.com/langchain-ai/langchain) - For pioneering LLM frameworks
- [AutoGPT](https://github.com/Significant-Gravitas/AutoGPT) - For autonomous agent patterns

## 📬 Contact & Community

- **GitHub Issues**: [Report bugs or request features](https://github.com/Undiluted7027/allos-agent-sdk/issues)
- **Discussions**: [Join the conversation](https://github.com/Undiluted7027/allos-agent-sdk/discussions)
- **Twitter**: [@allos_sdk](https://twitter.com/allos_sdk) *(coming soon)*
- **Discord**: [Join our community](https://discord.gg/allos) *(coming soon)*

---

<div align="center">

**Built with ❤️ by the open source community**

[⭐ Star us on GitHub](https://github.com/Undiluted7027/allos-agent-sdk) • [🐦 Follow on X](https://x.com/saminem2079)

</div>
