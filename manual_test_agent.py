# manual_test_agent.py

import os
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from allos import Agent, AgentConfig
from allos.utils.errors import AllosError

# --- Configuration ---
PROVIDER = "openai"  # or "anthropic"
MODEL = "gpt-4o"  # or "claude-3-haiku-20240307"
SESSION_FILE = "agent_session.json"
TEST_FILENAME = "hello_app.py"

# Use a specific workspace for this test to keep things clean
WORKSPACE_DIR = Path("./agent_workspace")
WORKSPACE_DIR.mkdir(exist_ok=True)
os.chdir(WORKSPACE_DIR)  # Change current directory to the workspace

console = Console()


def cleanup():
    """Cleans up created files."""
    console.print("\n--- Cleaning up ---")
    if Path(SESSION_FILE).exists():
        os.remove(SESSION_FILE)
        console.print(f"🗑️  Deleted session file: {SESSION_FILE}")
    if Path(TEST_FILENAME).exists():
        os.remove(TEST_FILENAME)
        console.print(f"🗑️  Deleted test file: {TEST_FILENAME}")
    os.chdir("..")  # Return to original directory
    os.rmdir(WORKSPACE_DIR)
    console.print(f"🗑️  Deleted workspace: {WORKSPACE_DIR}")


def main():
    """Runs the end-to-end agent workflow test."""

    console.print(
        Panel(
            f"🚀 [bold]Starting End-to-End Agent Test[/]\nProvider: {PROVIDER}\nModel: {MODEL}",
            title="Setup",
        )
    )

    try:
        # --- PART 1: Create a file and save the session ---
        console.print("\n--- [bold]Part 1: Initial Task & Save Session[/] ---")

        # 1. Create the agent
        config = AgentConfig(
            provider_name=PROVIDER,
            model=MODEL,
            tool_names=["read_file", "write_file", "edit_file", "shell_exec"],
        )
        agent_part1 = Agent(config)

        # 2. Run the first task
        prompt1 = f"Create a new Python file named '{TEST_FILENAME}' that contains a single function `greet()` which prints 'Hello from Allos!'."

        # You will be prompted to approve the 'write_file' tool. Type 'y' and press Enter.
        agent_part1.run(prompt1)

        # 3. Verify the file was created
        assert Path(TEST_FILENAME).exists(), f"{TEST_FILENAME} was not created!"
        console.print(
            f"\n[green]✅ Verification: '{TEST_FILENAME}' created successfully.[/green]"
        )

        # 4. Save the session
        agent_part1.save_session(SESSION_FILE)
        assert Path(SESSION_FILE).exists(), f"{SESSION_FILE} was not created!"

        # --- PART 2: Load the session and continue the task ---
        console.print("\n--- [bold]Part 2: Load Session & Continue Task[/] ---")

        # 1. Load the agent from the session file
        agent_part2 = Agent.load_session(SESSION_FILE)
        console.print(
            f"[green]✅ Session loaded successfully from '{SESSION_FILE}'.[/green]"
        )
        console.print(
            f"Loaded agent has {len(agent_part2.context)} messages in its context."
        )
        assert len(agent_part2.context) > 0

        # 2. Run a follow-up task
        prompt2 = f"Now, read the file '{TEST_FILENAME}' and add a new line at the end of the file that calls the `greet()` function."

        # You will be prompted to approve the 'edit_file' tool. Type 'y' and press Enter.
        agent_part2.run(prompt2)

        # 3. Verify the file was edited
        content = Path(TEST_FILENAME).read_text()
        assert "greet()" in content, "The call to greet() was not added!"
        console.print(
            f"\n[green]✅ Verification: '{TEST_FILENAME}' was edited successfully.[/green]"
        )
        console.print("Final file content:")
        console.print(Panel(content, title=TEST_FILENAME, border_style="blue"))

        # --- PART 3: Execute the final script ---
        console.print("\n--- [bold]Part 3: Execute Final Script with Shell Tool[/] ---")
        prompt3 = f"Execute the python script '{TEST_FILENAME}' and show me the output."

        # You will be prompted to approve the 'shell_exec' tool. Type 'y' and press Enter.
        agent_part2.run(prompt3)

    except (AllosError, AssertionError, Exception) as e:
        console.print(f"\n[bold red]❌ TEST FAILED: {e}[/]")
    finally:
        cleanup()


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    main()
