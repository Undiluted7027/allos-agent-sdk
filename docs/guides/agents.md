# Building Your First Agent

The `Agent` class is the heart of the Allos SDK. It is the orchestrator that brings together an LLM provider and a set of tools to accomplish complex tasks. This guide will walk you through configuring, running, and managing your agent.

## 1. Configuring the Agent

Every agent starts with an `AgentConfig`. This simple dataclass tells the agent which LLM to use and what tools it has access to.

```python
from allos import Agent, AgentConfig

config = AgentConfig(
    # Specify the LLM provider and model
    provider_name="openai",
    model="gpt-4o",

    # List the names of the tools the agent can use
    tool_names=["read_file", "write_file", "shell_exec"]
)
```
The `tool_names` must correspond to the names of built-in or custom tools that have been registered with the `ToolRegistry`.

## 2. Creating and Running the Agent

Once you have a configuration, creating and running the agent is straightforward.

```python
from allos import Agent

# Create the agent instance
agent = Agent(config)

# Give the agent a high-level task
prompt = "Create a file named 'hello.txt' with the content 'Hello, World!'."

# Run the agent and get the final response
final_response = agent.run(prompt)

print(final_response)
```

During the `run()` method, the agent will perform its reasoning loop:
1.  **Plan:** It sends the current conversation history to the LLM.
2.  **Act:** If the LLM requests a tool, the agent asks for your permission (if required) and executes it.
3.  **Observe:** The agent adds the tool's result to the conversation history and loops back to the planning step.
4.  This continues until the LLM provides a final text answer without requesting any more tools.

## 3. Session Management: Saving and Loading

For tasks that span multiple sessions, you can save the agent's state (its configuration and entire conversation history) and load it back later.

```python
# After the first run, save the session
agent.save_session("my_task.json")

# ... later, in another script or at another time ...

# Load the agent from the file. No config needed!
loaded_agent = Agent.load_session("my_task.json")

# Continue the task. The agent remembers everything from the previous session.
follow_up_prompt = "Now, read the file 'hello.txt' and tell me its content."
response = loaded_agent.run(follow_up_prompt)

print(response)
```

This is the fundamental workflow for using the Allos `Agent`. For more details on the available tools and how to create your own, see the following guides:
- **[Using Built-in Tools](./tools.md)**
- **[Creating Custom Tools](./custom-tools.md)**
