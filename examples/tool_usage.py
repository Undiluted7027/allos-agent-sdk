# examples/tool_usage.py

"""
An example demonstrating how to use the built-in tools and the ToolRegistry.

This script shows the fundamental steps for working with tools:
1. List all available, built-in tools from the ToolRegistry.
2. Get an instance of a specific tool.
3. Execute the tool with the required parameters.

To run this example:
1. Ensure the core SDK is installed: `uv pip install allos-agent-sdk`
2. Run the script: `python examples/tool_usage.py`
"""

import os

from allos.tools import ToolRegistry
from allos.utils.errors import AllosError

# A dummy file to interact with
DUMMY_FILE_NAME = "example_file.txt"
DUMMY_FILE_CONTENT = (
    "Hello from the Allos SDK!\nThis is line two.\nThis is the final line."
)


def main():
    """Main function to demonstrate tool usage."""
    try:
        # --- Create a dummy file for the example ---
        print(f"Creating a temporary file: '{DUMMY_FILE_NAME}'")
        with open(DUMMY_FILE_NAME, "w") as f:
            f.write(DUMMY_FILE_CONTENT)

        # --- 1. List all available tools ---
        print("\n--- Listing all available tools ---")
        all_tool_names = ToolRegistry.list_tools()
        print(f"Registered tools: {all_tool_names}")
        assert "read_file" in all_tool_names
        assert "write_file" in all_tool_names
        assert "shell_exec" in all_tool_names

        # --- 2. Get and execute the 'read_file' tool ---
        print("\n--- Executing the 'read_file' tool ---")
        read_tool = ToolRegistry.get_tool("read_file")

        # Execute the tool by passing arguments as keywords
        result = read_tool.execute(path=DUMMY_FILE_NAME)

        print(f"Result of reading '{DUMMY_FILE_NAME}':")
        if result["status"] == "success":
            print("--- FILE CONTENT ---")
            print(result["content"])
            print("--------------------")
        else:
            print(f"Error: {result['message']}")

        # --- 3. Get and execute the 'shell_exec' tool ---
        print("\n--- Executing the 'shell_exec' tool ---")
        shell_tool = ToolRegistry.get_tool("shell_exec")

        # This command is safe and read-only
        command_to_run = f"wc -l {DUMMY_FILE_NAME}"  # Word count (lines)
        print(f"Running command: '{command_to_run}'")
        result = shell_tool.execute(command=command_to_run)

        print("Result of shell command:")
        if result["status"] == "success":
            print(f"  Return Code: {result['return_code']}")
            print(f"  STDOUT:\n{result['stdout']}")
            print(f"  STDERR:\n{result['stderr']}")
        else:
            print(f"Error: {result['message']}")

    except AllosError as e:
        print(f"\nAn SDK error occurred: {e}")
    finally:
        # --- Clean up the dummy file ---
        if os.path.exists(DUMMY_FILE_NAME):
            print(f"\nCleaning up temporary file: '{DUMMY_FILE_NAME}'")
            os.remove(DUMMY_FILE_NAME)


if __name__ == "__main__":
    main()
