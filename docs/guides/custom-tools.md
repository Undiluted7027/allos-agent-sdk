# Creating Custom Tools

One of the most powerful features of the Allos SDK is the ability to create your own custom tools. This allows you to connect the agent to any proprietary API, database, or specific function you need.

## The `BaseTool` Class

All custom tools must inherit from `allos.tools.BaseTool` and be decorated with `@tool`. You must define the `name`, `description`, `parameters`, and implement the `execute` method.

## A Complete, Runnable Example

Let's build a complete script that defines a custom tool and uses an agent to call it.

### 1. The `my_tools.py` File

First, create a file named `my_tools.py` to define our custom tool. This tool will simulate querying a database.

```python
# my_tools.py
from typing import Any, Dict

from allos.tools import BaseTool, tool, ToolParameter

# A mock database connection for this example
class MockDB:
    def execute(self, query: str) -> Dict:
        # A simple mock that only understands one query
        if "users" in query.lower() and "count" in query.lower():
            return {"status": "success", "result": [{"count": 150}]}
        return {"status": "error", "message": "Query not supported in this mock."}

db_connection = MockDB()

# The @tool decorator automatically registers the tool with the ToolRegistry
@tool
class DatabaseQueryTool(BaseTool):
    """A custom tool to query a database."""

    # The unique name the LLM will use to call the tool
    name: str = "query_database"

    # A clear description for the LLM to understand what the tool does
    description: str = "Executes a read-only SQL query against the company database to find information about users."

    # Define the parameters the tool accepts
    parameters: list[ToolParameter] = [
        ToolParameter(
            name="sql_query",
            type="string",
            description="The SQL query to execute.",
            required=True,
        )
    ]

    # Implement the core logic of the tool
    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        sql_query = kwargs.get("sql_query")
        if not sql_query:
            return {"status": "error", "message": "'sql_query' argument is required."}

        print(f"\n[Custom Tool] Executing database query: {sql_query}")
        result = db_connection.execute(sql_query)
        return result
```

### 2. The Agent Script

Now, create your main script, `run_custom_tool_agent.py`, in the same directory. This script will import your custom tool and use an agent to solve a task.

```python
# run_custom_tool_agent.py
from dotenv import load_dotenv
from allos import Agent, AgentConfig

# This import is the key! It makes our custom tool discoverable.
import my_tools

# Load API keys from your .env file
load_dotenv()

def main():
    # 1. Configure the Agent
    # We include our custom tool's name in the list of tools.
    config = AgentConfig(
        provider_name="openai",
        model="gpt-4o",
        tool_names=["query_database"]
    )

    # 2. Create the Agent
    agent = Agent(config)

    # 3. Define the task for the agent
    prompt = "I need to know how many users are in our database for the monthly report."

    # 4. Run the agent
    # The agent will see the prompt, understand that the `query_database` tool
    # is the right one for the job, generate the correct SQL query, and execute it.
    final_response = agent.run(prompt)

    print("\n--- Agent's Final Analysis ---")
    print(final_response)

if __name__ == "__main__":
    main()
```

### 3. Run the Example

Before running, make sure you have a `.env` file with your `OPENAI_API_KEY`.

Execute the agent script from your terminal:

```bash
python run_custom_tool_agent.py
```

### Expected Output

You will see the agent's thought process, including the call to your custom tool:

```text
...
╭────────────────────────────── Tool Call Requested ───────────────────────────────╮
│ Tool: query_database                                                             │
│ Arguments: {                                                                     │
│   "sql_query": "SELECT COUNT(*) FROM users;"                                     │
│ }                                                                                │
╰──────────────────────────────────────────────────────────────────────────────────╯

[Custom Tool] Executing database query: SELECT COUNT(*) FROM users;
╭────────────────────────── Tool Result: query_database ───────────────────────────╮
│ {                                                                                │
│   "status": "success",                                                           │
│   "result": [                                                                    │
│     {                                                                            │
│       "count": 150                                                               │
│     }                                                                            │
│   ]                                                                              │
│ }                                                                                │
╰──────────────────────────────────────────────────────────────────────────────────╯
🧠 Thinking...
╭──────────────────────────────── Final Response ────────────────────────────────╮
│ Agent: There are 150 users in the database.                                    │
╰────────────────────────────────────────────────────────────────────────────────╯

--- Agent's Final Analysis ---
There are 150 users in the database.
```

This demonstrates how easily you can extend the Allos Agent's capabilities by defining your own Python classes, allowing it to connect to any data source or API you need.
