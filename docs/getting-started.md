# Getting Started

Welcome to the Allos Agent SDK! This guide provides a conceptual overview to get you started. For a runnable code example, see our [5-Minute Quickstart](./guides/quickstart.md).

## 1. Installation

First, make sure you've installed the SDK along with the providers you need. For example, to use OpenAI and Anthropic:

```bash
uv pip install "allos-agent-sdk[all]"
```
For more details, see the [Installation Guide](./installation.md).

## 2. Configure Your API Keys

The SDK automatically reads API keys from your environment variables. Before you can use a provider, you must set the appropriate key.

Create a `.env` file in your project's root directory:

```env
# .env
OPENAI_API_KEY="your_openai_api_key_here"
ANTHROPIC_API_KEY="your_anthropic_api_key_here"
```

Your Python script will need to load these variables. We recommend using the `python-dotenv` library.

## 3. The `ProviderRegistry`

The central entry point for interacting with LLMs in Allos is the `ProviderRegistry`. It acts as a factory that gives you a provider instance on demand.

You never need to import `OpenAIProvider` or `AnthropicProvider` directly. You simply ask the registry for the provider you want by name.

```python
from allos.providers import ProviderRegistry

# Get an instance of the OpenAI provider for the gpt-4o model
openai_provider = ProviderRegistry.get_provider("openai", model="gpt-4o")

# Get an instance of the Anthropic provider
anthropic_provider = ProviderRegistry.get_provider("anthropic", model="claude-3-5-sonnet-20240620")
```

## 4. Making an API Call

Once you have a provider instance, you can use the unified `.chat()` method to interact with the LLM. You'll pass it a list of `Message` objects.

```python
from allos.providers import Message, MessageRole

# Create a list of messages
messages = [
    Message(role=MessageRole.USER, content="Hello, what is your name?")
]

# Get a response from OpenAI
response = openai_provider.chat(messages)

print(response.content)
```

The interface is the **same** for every provider, which is the core power of Allos.

## 5. Configure and Create an Agent

Instead of using providers directly, you'll typically configure and create an `Agent` instance.

```python
from allos import Agent, AgentConfig

# Define the agent's configuration
config = AgentConfig(
    provider_name="openai",
    model="gpt-4o",
    tool_names=["read_file", "write_file", "shell_exec"]
)

# Create the agent
agent = Agent(config)
```
The `Agent` automatically initializes the correct provider and tools from the registries based on your configuration.

## 6. Run a Task

Once you have an agent, you can give it a high-level task using the `.run()` method.

```python
# The agent will use its LLM and tools to accomplish this task
result = agent.run("Create a python script named 'app.py' that prints 'Hello, Agent!' and then execute it.")
```
During the run, the agent will prompt you for permission before executing sensitive tools like `write_file` or `shell_exec`.

## 7. Save and Load Sessions

You can persist an agent's conversation history to a file and load it back later to continue a task.

```python
# Save the agent's state after the task
agent.save_session("my_project_session.json")

# Later, in a different script or session...
loaded_agent = Agent.load_session("my_project_session.json")

# Continue the conversation
result = loaded_agent.run("Now, modify the script to print 'Hello again!'")
```

Now, head to the [Quickstart Guide](./guides/quickstart.md) to run this yourself!
