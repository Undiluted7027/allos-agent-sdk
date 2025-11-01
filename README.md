# Allos Agent SDK

<div align="center">

**🚀 The LLM-Agnostic Agentic Framework**

*Build powerful AI agents without vendor lock-in*

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: MVP Development](https://img.shields.io/badge/status-MVP%20Development-orange.svg)](./MVP_ROADMAP.md)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](./CONTRIBUTING.md)

[Documentation](./docs) • [Roadmap](./ROADMAP.md) • [Contributing](./CONTRIBUTING.md)

</div>

---

## 🎯 What is Allos?

Allos is an open-source, provider-agnostic agentic SDK that gives you the power to build production-ready AI agents that work with **any LLM provider**. Inspired by Anthropic's Claude Code, Allos delivers the same outstanding capabilities without locking you into a single ecosystem.

**The Problem**: Most agentic frameworks force you to choose between vendors, making it expensive and risky to switch models.

**The Solution**: Allos provides a unified interface across OpenAI, Anthropic, Ollama, Google, and more—so you can use the best model for each task without rewriting your code.

## ✨ Key Features

### 🔄 **Provider Agnostic**
Switch seamlessly between OpenAI, Anthropic, Ollama, and other LLM providers. Use GPT-4 for one task, Claude for another, or run models locally—all with the same code.

### 🛠️ **Rich Tool Ecosystem**
Built-in tools for:
- 📁 File operations (read, write, edit)
- 💻 Shell command execution
- 🌐 Web search and fetching
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
pip install allos-agent-sdk
export OPENAI_API_KEY=your_key
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

### Installation

```bash
# Basic installation
pip install allos-agent-sdk

# With specific providers
pip install "allos-agent-sdk[openai]"
pip install "allos-agent-sdk[anthropic]"
pip install "allos-agent-sdk[all]"  # All providers
```

### CLI Usage

```bash
# Set your API key
export OPENAI_API_KEY=your_key_here

# Run a task
allos "Create a FastAPI hello world app"

# Use a different provider
export ANTHROPIC_API_KEY=your_key_here
allos --provider anthropic --model claude-sonnet-4-5 "Same task"

# Interactive mode
allos --interactive

# With specific tools
allos --tools read_file --tools write_file "Refactor main.py"

# Save session for later
allos --session my-project.json "Start building a web scraper"
allos --session my-project.json "Continue where we left off"
```

### Python API

```python
from allos import Agent, AgentConfig

# Simple agent
agent = Agent(AgentConfig(
    provider="openai",
    model="gpt-4",
    tools=["read_file", "write_file", "shell_exec"]
))

result = agent.run("Fix the bug in main.py and add tests")
print(result)
```

### Provider Switching Example

```python
# Start with OpenAI
agent_openai = Agent(AgentConfig(
    provider="openai",
    model="gpt-4",
    tools=["read_file", "write_file"]
))

# Switch to Anthropic for complex reasoning
agent_claude = Agent(AgentConfig(
    provider="anthropic",
    model="claude-sonnet-4-5",
    tools=["read_file", "write_file"]
))

# Or use local models with Ollama
agent_local = Agent(AgentConfig(
    provider="ollama",
    model="qwen2.5-coder",
    tools=["read_file", "write_file"]
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

    def execute(self, query: str):
        # Your implementation
        result = your_db.execute(query)
        return {"success": True, "result": result}

# Use it
agent = Agent(AgentConfig(
    provider="openai",
    model="gpt-4",
    tools=["query_database", "read_file"]
))
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      CLI Layer                          │
│              (User-friendly interface)                  │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│                   Agent Core                            │
│        (Orchestration & Agentic Loop)                   │
└─────┬──────────────────┬──────────────────┬────────────-┘
      │                  │                  │
┌─────▼────────┐  ┌─────▼────────┐  ┌─────▼────────┐
│  Providers   │  │    Tools     │  │   Context    │
│              │  │              │  │              │
│ • OpenAI     │  │ • FileSystem │  │ • History    │
│ • Anthropic  │  │ • Shell      │  │ • Compactor  │
│ • Ollama     │  │ • Web        │  │ • Cache      │
│ • Google     │  │ • Custom     │  │ • Manager    │
└──────────────┘  └──────────────┘  └──────────────┘
```

### Core Components

1. **Provider Layer**: Unified interface for all LLM providers
2. **Tool System**: Extensible toolkit with built-in and custom tools
3. **Agent Core**: Main agentic loop with planning and execution
4. **Context Manager**: Automatic context window optimization
5. **CLI**: User-friendly command-line interface

## 📊 Provider Support

| Provider | Status | Models | Features |
|----------|--------|--------|----------|
| **OpenAI** | ✅ Ready | GPT-4, GPT-3.5-Turbo, GPT-4o | Tool calling, streaming |
| **Anthropic** | ✅ Ready | Claude 3, Claude 4 (Opus, Sonnet, Haiku) | Tool calling, streaming |
| **Ollama** | 🚧 Coming Soon | Llama, Mistral, Qwen, etc. | Local models |
| **Google** | 🚧 Coming Soon | Gemini Pro, Gemini Ultra | Tool calling |
| **Cohere** | 📋 Planned | Command R, Command R+ | Tool calling |
| **Custom** | ✅ Ready | Any OpenAI-compatible API | Extensible |

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
# SRE Agent - Diagnose and fix production issues
sre_agent = Agent(AgentConfig(
    provider="anthropic",
    model="claude-4-opus",
    tools=["read_file", "shell_exec", "web_search"]
))
sre_agent.run("Investigate why the API latency spiked at 3pm")

# Code Review Agent
review_agent = Agent(AgentConfig(
    provider="openai",
    model="gpt-4",
    tools=["read_file", "write_file"]
))
review_agent.run("Review PR #123 for security issues and best practices")
```

### Business Automation
```python
# Data Analysis Agent
data_agent = Agent(AgentConfig(
    provider="openai",
    model="gpt-4",
    tools=["read_file", "write_file", "query_database"]
))
data_agent.run("Analyze Q4 sales data and create a summary report")

# Content Creation Agent
content_agent = Agent(AgentConfig(
    provider="anthropic",
    model="claude-sonnet-4-5",
    tools=["web_search", "read_file", "write_file"]
))
content_agent.run("Research AI trends and write a blog post")
```

## 📚 Documentation

- **[Getting Started](./docs/getting-started.md)** - Installation and first steps
- **[Quickstart Guide](./docs/guides/quickstart.md)** - 5-minute tutorial
- **[Providers](./docs/guides/providers.md)** - Provider configuration
- **[Tools](./docs/guides/tools.md)** - Using built-in tools
- **[Custom Tools](./docs/guides/custom-tools.md)** - Creating your own tools
- **[CLI Reference](./docs/reference/cli-reference.md)** - Command-line options
- **[API Reference](./docs/reference/agent-api.md)** - Python API documentation
- **[Architecture](./docs/reference/initial-architecture-design.md)** - System design

## 🗺️ Roadmap

### ✅ Phase 1: MVP (Current)
- [x] Initial architecture design
- [x] Directory structure
- [ ] Provider layer (OpenAI, Anthropic)
- [ ] Tool system (filesystem, shell)
- [ ] Agent core with agentic loop
- [ ] CLI interface
- [ ] Basic tests
- [ ] Documentation

See [MVP_ROADMAP.md](./MVP_ROADMAP.md) for detailed MVP timeline.

### 🚧 Phase 2: Enhanced Features
- [ ] Ollama integration (local models)
- [ ] Google Gemini support
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

See [ROADMAP.md](./ROADMAP.md) for the complete roadmap.

## 🚦 Current Status

**🟠 MVP Development in Progress**

Allos is currently under active development. The MVP will include:
- ✅ Initial architecture designed
- ⏳ OpenAI and Anthropic providers
- ⏳ Essential file and shell tools
- ⏳ Basic agentic loop
- ⏳ CLI interface
- ⏳ Python API

**Expected MVP Release**: 6-8 weeks from project start

We welcome early contributors! See [Contributing](#-contributing) below.

## 🤝 Contributing

We're building Allos in the open and would love your help! Whether you're:

- 🐛 **Reporting bugs**
- 💡 **Suggesting features**
- 📖 **Improving documentation**
- 🔧 **Submitting PRs**
- ⭐ **Starring the repo** (helps a lot!)

All contributions are welcome! See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/Undiluted7027/allos-agent-sdk.git
cd allos-agent-sdk
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
pip install -e ".[dev]"

# Run tests
pytest

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

# Run tests
pytest

# Format code
black allos tests
ruff check allos tests --fix
```



## 🌟 Why "Allos"?

**Allos** (Greek: ἄλλος) means "other" or "different" - representing our core philosophy of **choice and flexibility**. Just as the word implies alternatives and options, Allos gives you the freedom to choose any LLM provider without constraints.

## 📄 License

Allos is open source and available under the [MIT License](./LICENSE).

## 🙏 Acknowledgments

Inspired by:
- [Anthropic's Claude Code](https://claude.ai/code) - For showing what's possible with agentic coding
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
