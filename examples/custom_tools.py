# examples/custom_tools.py

"""
An example demonstrating how to quickly create and use a custom tool.

This script shows the three key steps:
1. Define a custom tool by inheriting from `BaseTool` and using the `@tool` decorator.
2. Ensure the tool is discoverable by importing the file it's defined in.
3. Get the tool from the ToolRegistry and execute it manually.

To run this example:
1. Ensure the core SDK is installed: `uv pip install allos-agent-sdk`
2. Run the script: `python examples/custom_tools.py`
"""

from typing import Any, Dict

from allos.tools import BaseTool, ToolParameter, ToolRegistry, tool

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
    """Main function to demonstrate using the custom tool."""

    print("--- Demonstrating Custom Tool Creation and Usage ---")

    # Verify that our custom tool is now in the registry alongside the built-in ones
    all_tools = ToolRegistry.list_tools()
    print(f"\nAll available tools: {all_tools}")
    assert "calculate_rectangle_area" in all_tools

    # --- Step 3: Use the Custom Tool Manually ---
    print("\n--- Getting and executing the custom tool ---")
    try:
        # Get an instance of our tool from the registry by its name
        area_tool = ToolRegistry.get_tool("calculate_rectangle_area")

        # Execute it with some arguments
        arguments = {"length": 10, "width": 5}
        print(f"Executing tool '{area_tool.name}' with arguments: {arguments}")
        result = area_tool.execute(**arguments)

        print("Result from tool execution:")
        print(result)
        assert result.get("area") == 50

        print("\nCustom tool works as expected!")

    except Exception as e:
        print(f"An error occurred: {e}")

    # This shows how an agent (coming in Phase 4) will eventually use the tool.
    print("\n--- Future Vision: How an Agent will use this ---")
    print(
        "In Phase 4, an agent could be asked: 'What is the area of a rectangle with length 10 and width 5?'"
    )
    print(
        "The agent would see our 'calculate_rectangle_area' tool, generate the arguments {'length': 10, 'width': 5}, and call it automatically."
    )


if __name__ == "__main__":
    main()
