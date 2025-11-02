# examples/code_review_agent.py

"""
An example of a Code Review Agent that can read files and provide feedback.

This agent uses the `read_file` and `list_directory` tools to inspect a simple
project and suggest improvements.

To run this example:
1. Install dependencies: `uv pip install "allos-agent-sdk[openai]" python-dotenv`
2. Create a .env file and add your OPENAI_API_KEY.
3. Run the script: `python examples/code_review_agent.py`
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from allos import Agent, AgentConfig

# --- Setup a dummy project for the agent to review ---
DUMMY_PROJECT_DIR = Path("./code_review_workspace")
DUMMY_FILE_CONTENT = """
def add(a, b):
    # A simple function to add two numbers
    result = a + b
    return result

def main():
    x = 5
    y = 10
    print(f"The result is: {add(x, y)}")

# No entry point protection
main()
"""


def setup_workspace():
    """Create a dummy project for the agent to review."""
    if DUMMY_PROJECT_DIR.exists():
        import shutil

        shutil.rmtree(DUMMY_PROJECT_DIR)
    DUMMY_PROJECT_DIR.mkdir()
    (DUMMY_PROJECT_DIR / "main.py").write_text(DUMMY_FILE_CONTENT)
    # The agent will operate inside this directory
    os.chdir(DUMMY_PROJECT_DIR)


def cleanup():
    """Clean up the dummy project."""
    os.chdir("..")
    if DUMMY_PROJECT_DIR.exists():
        import shutil

        shutil.rmtree(DUMMY_PROJECT_DIR)


def main():
    console = Console()
    setup_workspace()

    console.print(
        Panel(
            "Created a dummy project with `main.py` for the agent to review.",
            title="Setup",
        )
    )

    try:
        config = AgentConfig(
            provider_name="openai",
            model="gpt-4o",
            tool_names=["list_directory", "read_file"],
        )
        agent = Agent(config)

        prompt = (
            "Please review the code in the current project. "
            "Start by listing the files, then read `main.py` and provide feedback "
            "on code quality, style, and potential bugs."
        )

        final_response = agent.run(prompt)

        console.print("\n--- [bold green]Code Review Complete[/] ---")
        console.print(final_response)

    finally:
        cleanup()


if __name__ == "__main__":
    load_dotenv()
    main()
