# examples/tool_usage.py

"""
An example demonstrating how to use the built-in tools and the ToolRegistry directly,
without using the main Agent class.

This script shows how to:
1. List all available tools.
2. Get an instance of a specific tool.
3. Execute the tool with parameters and inspect the result.

To run this example:
1. Ensure the core SDK is installed: `uv pip install allos-agent-sdk`
2. Run the script: `python examples/tool_usage.py`
"""

import os
import platform
import shutil
from pathlib import Path

from allos.tools import ToolRegistry
from allos.utils.errors import AllosError

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
        dummy_file = "example.txt"
        with open(dummy_file, "w") as f:
            f.write("line 1\nline 2\n")

        print("--- 1. Listing all available tools ---")
        all_tool_names = ToolRegistry.list_tools()
        print(f"Registered tools: {all_tool_names}")

        print("\n--- 2. Executing the 'list_directory' tool ---")
        list_tool = ToolRegistry.get_tool("list_directory")
        result = list_tool.execute()
        print(f"Result of listing current directory: {result['contents']}")

        print("\n--- 3. Executing the 'shell_exec' tool ---")
        shell_tool = ToolRegistry.get_tool("shell_exec")
        command = (
            f"wc -l {dummy_file}"
            if platform.system() != "Windows"
            else f'find /c /v "" {dummy_file}'
        )
        print(f"Running command: '{command}'")
        result = shell_tool.execute(command=command)
        print(f"Result STDOUT:\n{result['stdout'].strip()}")

    except AllosError as e:
        print(f"\nAn SDK error occurred: {e}")
    finally:
        cleanup()


if __name__ == "__main__":
    main()
