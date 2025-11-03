# examples/tool_usage.py

"""
An example demonstrating how to use the built-in tools and the ToolRegistry directly.

This script shows the fundamental steps for working with tools manually:
1. List all available, built-in tools from the ToolRegistry.
2. Get an instance of a specific tool.
3. Execute the tool with the required parameters and inspect the result.

To run this example:
1. Ensure the core SDK is installed: `uv pip install allos-agent-sdk`
2. Run the script: `python examples/tool_usage.py`
"""

import os
import shutil
from pathlib import Path

from allos.tools import ToolRegistry
from allos.utils.errors import AllosError

# Use a specific workspace for this test to keep things clean
WORKSPACE_DIR = Path("./tool_usage_workspace")


def setup_workspace():
    if WORKSPACE_DIR.exists():
        shutil.rmtree(WORKSPACE_DIR)
    WORKSPACE_DIR.mkdir()
    os.chdir(WORKSPACE_DIR)


def cleanup():
    os.chdir("..")
    if WORKSPACE_DIR.exists():
        shutil.rmtree(WORKSPACE_DIR)


def main():
    """Main function to demonstrate manual tool usage."""
    setup_workspace()
    try:
        # --- Create a dummy file for the example ---
        dummy_file_name = "example_file.txt"
        dummy_file_content = "Hello from the Allos SDK!\nThis is line two."
        print(f"Creating a temporary file: '{dummy_file_name}'")
        with open(dummy_file_name, "w") as f:
            f.write(dummy_file_content)

        # --- 1. List all available tools ---
        print("\n--- Listing all available tools ---")
        all_tool_names = ToolRegistry.list_tools()
        print(f"Registered tools: {all_tool_names}")

        # --- 2. Get and execute the 'read_file' tool ---
        print("\n--- Executing the 'read_file' tool ---")
        read_tool = ToolRegistry.get_tool("read_file")
        result = read_tool.execute(
            path=dummy_file_name, start_line=2, end_line=2, inclusive=True
        )

        print(f"Result of reading line 2 from '{dummy_file_name}':")
        if result["status"] == "success":
            print(f"  Content: '{result['content']}'")
        else:
            print(f"  Error: {result['message']}")

        # --- 3. Get and execute the 'shell_exec' tool ---
        print("\n--- Executing the 'shell_exec' tool ---")
        shell_tool = ToolRegistry.get_tool("shell_exec")

        command_to_run = (
            f"ls -l {dummy_file_name}" if os.name != "nt" else f"dir {dummy_file_name}"
        )
        print(f"Running command: '{command_to_run}'")
        result = shell_tool.execute(command=command_to_run)

        print("Result of shell command:")
        if result["status"] == "success":
            print(f"  Return Code: {result['return_code']}")
            print(f"  STDOUT:\n{result['stdout'].strip()}")
        else:
            print(f"  Error: {result['message']}")

    except AllosError as e:
        print(f"\nAn SDK error occurred: {e}")
    finally:
        cleanup()


if __name__ == "__main__":
    main()
