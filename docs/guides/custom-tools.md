# Creating Custom Tools

One of the most powerful features of the Allos SDK is the ability to create your own custom tools. This allows you to connect the agent to any proprietary API, database, or specific function you need.

## The `BaseTool` Class

All custom tools must inherit from the `allos.tools.BaseTool` class and implement a few required attributes and methods.

## Example: A Custom Database Tool

Let's create a simple tool that can execute a read-only SQL query.

### 1. Define the Tool Class

Create a new file, for example `my_tools.py`, and define your tool.

```python
# my_tools.py
from typing import Any, Dict

from allos.tools import BaseTool, tool, ToolParameter

# A mock database connection for this example
class MockDB:
    def execute(self, query: str) -> Dict:
        if "users" in query.lower() and "count" in query.lower():
            return {"status": "success", "result": [{"count": 150}]}
        return {"status": "error", "message": "Query not supported in this mock."}

db_connection = MockDB()


# The @tool decorator automatically registers the tool with the ToolRegistry
@tool
class DatabaseQueryTool(BaseTool):
    # The unique name the LLM will use to call the tool
    name: str = "query_database"

    # A clear description for the LLM to understand what the tool does
    description: str = "Executes a read-only SQL query against the company database."

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

        print(f"Executing database query: {sql_query}")
        result = db_connection.execute(sql_query)
        return result

```

### 2. Make the Tool Discoverable

To make your custom tool available, you just need to import the file where it's defined somewhere in your application's entry point. The `@tool` decorator handles the registration automatically.

```python
# your_main_agent_script.py

# This import triggers the @tool decorator in my_tools.py
import my_tools

from allos.tools import ToolRegistry

# Your custom tool will now appear in the list!
print(ToolRegistry.list_tools())
```

### 3. Using the Custom Tool

Once the Agent Core is implemented in Phase 4, you will be able to provide your custom tool to an agent. The agent will then be able to intelligently use it to answer questions.

```python
# --- Conceptual Code (coming in Phase 4) ---
#
# from allos import Agent, AgentConfig
#
# config = AgentConfig(
#     provider="openai",
#     model="gpt-4o",
#     tools=["query_database", "read_file"] # List your custom tool by name
# )
#
# agent = Agent(config)
# result = agent.run("How many users are in the database?")
#
# print(result)
```

The agent would see the user's question, identify that `query_database` is the right tool, generate the appropriate SQL (`SELECT COUNT(*) FROM users;`), and execute your custom tool to get the answer.
