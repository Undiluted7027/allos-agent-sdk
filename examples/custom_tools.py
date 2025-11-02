# examples/custom_tools.py

"""
An example demonstrating how to create a custom tool and have an agent use it.

This script shows how to:
1. Define a custom tool by inheriting from `BaseTool`.
2. Configure an agent to use this custom tool.
3. Run the agent and watch it intelligently decide to use your tool to solve a problem.

To run this example:
1. Install dependencies: `uv pip install "allos-agent-sdk[openai]" python-dotenv`
2. Create a .env file and add your OPENAI_API_KEY.
3. Run the script: `python examples/custom_tools.py`
"""

from typing import Any, Dict

from dotenv import load_dotenv

from allos import Agent, AgentConfig
from allos.tools import BaseTool, ToolParameter, tool

# --- Step 1: Define a Custom Tool ---
#
# We'll create a simple tool to calculate the area of a rectangle.
# The `@tool` decorator handles all the registration work automatically.


@tool
class RectangleAreaTool(BaseTool):
    """A custom tool to calculate the area of a rectangle."""

    name: str = "calculate_rectangle_area"
    description: str = "Calculates the area of a rectangle given its length and width."
    parameters: list[ToolParameter] = [
        ToolParameter(
            name="length",
            type="number",
            description="The length of the rectangle.",
            required=True,
        ),
        ToolParameter(
            name="width",
            type="number",
            description="The width of the rectangle.",
            required=True,
        ),
    ]

    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        length = kwargs.get("length")
        width = kwargs.get("width")

        if not isinstance(length, (int, float)) or not isinstance(width, (int, float)):
            return {"status": "error", "message": "Length and width must be numbers."}

        area = length * width
        return {"status": "success", "area": area}


# --- Step 2: Make the Tool Discoverable ---
#
# The tool is now automatically registered because this file has been executed.
# No other steps are needed. We can immediately use the ToolRegistry to find it.


def main():
    """Main function to demonstrate an agent using the custom tool."""
    print("--- Demonstrating Agent with Custom Tool ---")

    # 2. Configure an agent to use our custom tool by its name.
    config = AgentConfig(
        provider_name="openai",
        model="gpt-4o",
        tool_names=["calculate_rectangle_area"],  # Add your custom tool here
    )
    agent = Agent(config)

    # 3. Give the agent a prompt that requires the tool.
    prompt = "I have a garden plot that is 12.5 meters long and 8 meters wide. What is its total area?"

    final_response = agent.run(prompt)

    print("\n--- Agent's Final Analysis ---")
    print(final_response)


if __name__ == "__main__":
    load_dotenv()
    main()
