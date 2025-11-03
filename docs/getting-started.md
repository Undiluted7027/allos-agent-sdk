# Getting Started

Welcome to the Allos Agent SDK! This guide provides a conceptual overview of the two main ways to use Allos: the **Command-Line Interface (CLI)** for quick tasks and the **Python API** for building custom applications.

For a runnable code example, see our [5-Minute Quickstart](./guides/quickstart.md).

## 1. Installation & Setup

First, install the SDK and configure your API keys.

```bash
# Install the SDK with all providers
uv pip install "allos-agent-sdk[all]"

# Create a .env file for your keys
echo "OPENAI_API_KEY=your_key_here" > .env
```
For more details, see the [Installation Guide](./installation.md).

## 2. The CLI: Your Primary Interface

The `allos` CLI is the fastest way to get started. It gives you direct access to a pre-configured agent. You can run a single task by providing a prompt directly.

```bash
# The agent will use its tools (like `write_file` and `shell_exec`) to accomplish this task.
# It will ask for your permission before executing any sensitive commands.
allos "Create a python script named 'app.py' that prints 'Hello, Agent!' and then execute it."
```

For a conversational experience, use the interactive flag:
```bash
allos --interactive
```
For a complete list of commands and options, see the [CLI Reference](./reference/cli-reference.md).

## 3. The Python API: The underlying Engine

The CLI is a user-friendly wrapper around the core Python components of the SDK. You can use these components directly to build more complex, custom agentic applications.

### `AgentConfig` and `Agent`

These are the two main classes you'll work with. `AgentConfig` defines the agent's setup, and `Agent` is the engine that runs the task.

```python
from allos import Agent, AgentConfig

# 1. Define the agent's configuration
config = AgentConfig(
    provider_name="openai",
    model="gpt-4o",
    tool_names=["read_file", "write_file", "shell_exec"]
)

# 2. Create the agent
agent = Agent(config)

# 3. Run a task
result = agent.run("Create a simple 'hello world' script in a file named 'main.py'.")

print(result)
```

This Python code does the same thing as the CLI, but gives you programmatic control over the agent's lifecycle, its context, and its final output.

For more details on building with the Python API, see the [Agents Guide](./guides/agents.md).
