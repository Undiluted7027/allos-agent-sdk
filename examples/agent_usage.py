# examples/agent_usage.py

"""
A complete, end-to-end example demonstrating the core capabilities of the Allos Agent.

This script showcases:
1.  Agent initialization with configuration.
2.  Execution of a multi-step task using filesystem tools (`write_file`, `edit_file`).
3.  The agent's permission-requesting mechanism.
4.  Saving the agent's state (session) to a file.
5.  Loading the agent from the session file.
6.  Continuing the task with the loaded agent, which retains the full conversation context.
7.  Using the `shell_exec` tool to run the final created script.

To run this example:
1. Install all dependencies: `uv pip install "allos-agent-sdk[all]" python-dotenv`
2. Create a .env file and add your OPENAI_API_KEY and/or ANTHROPIC_API_KEY.
3. Run the script: `python examples/agent_usage.py`
"""

import os
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from allos import Agent, AgentConfig
from allos.utils.errors import AllosError

# --- Configuration ---
# Change this to "anthropic" to test with Claude!
PROVIDER = "openai"
MODEL = "gpt-4o"
SESSION_FILE = "agent_session.json"
TEST_FILENAME = "hello_app.py"

# Use a specific workspace for this test to keep things clean
WORKSPACE_DIR = Path("./agent_workspace")


console = Console()


def setup_workspace():
    """Create and move into a clean workspace directory."""
    if WORKSPACE_DIR.exists():
        import shutil

        shutil.rmtree(WORKSPACE_DIR)
    WORKSPACE_DIR.mkdir(exist_ok=True)
    os.chdir(WORKSPACE_DIR)


def cleanup():
    """Cleans up created files and returns to the parent directory."""
    os.chdir("..")
    if WORKSPACE_DIR.exists():
        import shutil

        shutil.rmtree(WORKSPACE_DIR)
        console.print(f"\n🗑️  Cleaned up workspace: {WORKSPACE_DIR}")


def main():
    """Runs the end-to-end agent workflow test."""
    setup_workspace()

    try:
        console.print(
            Panel(
                f"🚀 [bold]Starting End-to-End Agent Demo[/]\nProvider: {PROVIDER}\nModel: {MODEL}",
                title="Setup",
            )
        )

        # === PART 1: Create a file and save the session ===
        console.print("\n--- [bold]Part 1: Initial Task & Save Session[/] ---")

        config = AgentConfig(
            provider_name=PROVIDER,
            model=MODEL,
            tool_names=["write_file", "edit_file", "read_file", "shell_exec"],
        )
        agent_part1 = Agent(config)

        prompt1 = f"Create a new Python file named '{TEST_FILENAME}' that contains a single function `greet()` which prints 'Hello from Allos!'."
        agent_part1.run(prompt1)

        assert Path(TEST_FILENAME).exists(), f"{TEST_FILENAME} was not created!"
        console.print(
            f"\n[green]✅ Verification: '{TEST_FILENAME}' created successfully.[/green]"
        )

        agent_part1.save_session(SESSION_FILE)
        assert Path(SESSION_FILE).exists(), f"{SESSION_FILE} was not created!"

        # === PART 2: Load the session and continue the task ===
        console.print("\n--- [bold]Part 2: Load Session & Continue Task[/] ---")

        agent_part2 = Agent.load_session(SESSION_FILE)
        console.print(
            f"[green]✅ Session loaded successfully from '{SESSION_FILE}'.[/green]"
        )

        prompt2 = f"Now, add a new line at the end of the file '{TEST_FILENAME}' that calls the `greet()` function."
        agent_part2.run(prompt2)

        content = Path(TEST_FILENAME).read_text()
        assert "greet()" in content, "The call to greet() was not added!"
        console.print(
            f"\n[green]✅ Verification: '{TEST_FILENAME}' was edited successfully.[/green]"
        )

        # === PART 3: Execute the final script ===
        console.print("\n--- [bold]Part 3: Execute Final Script with Shell Tool[/] ---")
        prompt3 = f"Execute the python script '{TEST_FILENAME}' and show me the output."
        agent_part2.run(prompt3)

    except (AllosError, AssertionError, Exception) as e:
        console.print(f"\n[bold red]❌ DEMO FAILED: {e}[/]")
    finally:
        cleanup()


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    main()
