# examples/local_models.py

"""
"The Private Vault"
Demonstrates interacting with local data using a local model via Ollama.
This script creates a "secret" file and asks a local model to summarize it,
proving that sensitive data doesn't need to leave your machine.
"""

import os
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from allos import Agent, AgentConfig

console = Console()

SECRET_FILE = "top_secret_plans.txt"
SECRET_CONTENT = """
PROJECT OMEGA
Status: Confidential
Launch Date: 2026-01-01
The objective is to build a Dyson sphere around the sun using self-replicating nanobots.
Phase 1: Mercury resource extraction.
"""


def main():
    console.print(
        Panel.fit(
            "[bold green]🔒 Allos Privacy Mode (Local AI: Might not respect instructions)[/]",
            border_style="green",
        )
    )

    # 1. Setup Data
    Path(SECRET_FILE).write_text(SECRET_CONTENT)
    console.print(f"[dim]Created local file: {SECRET_FILE}[/dim]")

    # 2. Configure Local Agent
    # We use 'ollama_compat' to talk to a local Ollama instance
    config = AgentConfig(
        provider_name="ollama_compat",
        model="llama3.1:latest",  # Requires: ollama pull llama3.1
        no_tools=False,  # Let's try to let it read the file (if model is capable)
        tool_names=["read_file", "list_directory"],
    )

    try:
        agent = Agent(config)

        # 3. The Task
        # We explicitly tell it to read the file. Note: Smaller local models might struggle
        # with tool calling. If it fails, try with no_tools=True and paste content in prompt.
        prompt = f"Read the file '{SECRET_FILE}' and summarize the secret project. Do not reveal the launch date."

        console.print(f"\n[bold]User:[/bold] {prompt}")
        console.print(
            "[dim](Sending to local llama3.1:latest... Data stays on device)[/dim]"
        )

        response = agent.run(prompt)

        console.print(
            Panel(response, title="Local Agent Response", border_style="green")
        )

    except Exception as e:
        console.print(f"\n[bold red]Connection Failed:[/bold] {e}")
        console.print("Ensure Ollama is running: [bold]ollama serve[/]")
        console.print("Ensure model is pulled: [bold]ollama pull llama3.1:latest[/]")

    finally:
        if os.path.exists(SECRET_FILE):
            os.remove(SECRET_FILE)


if __name__ == "__main__":
    main()
