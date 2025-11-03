# examples/sre_debugging.py
"""
An example of an SRE (Site Reliability Engineering) Agent that can diagnose issues.

This agent uses the `shell_exec` tool to inspect the local system.
**WARNING:** This script executes shell commands. It will ask for your permission
before running each command. Do not approve commands if you don't understand them.

To run this example:
1. Install dependencies: `uv pip install "allos-agent-sdk[openai]" python-dotenv`
2. Create a .env file and add your OPENAI_API_KEY.
3. Run the script: `python examples/sre_debugging.py`
"""

from dotenv import load_dotenv
from rich.console import Console

from allos import Agent, AgentConfig


def main():
    console = Console()
    console.print(
        "[bold yellow]⚠️ WARNING: This agent will request to run shell commands on your system.[/]"
    )

    config = AgentConfig(
        provider_name="openai", model="gpt-4o", tool_names=["shell_exec"]
    )
    agent = Agent(config)

    prompt = (
        "I need to know the current date and time on this system, and also list "
        "the top 5 running processes sorted by CPU usage. Use the available tools to find this information."
    )

    final_response = agent.run(prompt)

    console.print("\n--- [bold green]SRE Task Complete[/] ---")
    console.print(final_response)


if __name__ == "__main__":
    load_dotenv()
    main()
