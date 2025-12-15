# examples/provider_switching.py

"""A sophisticated example demonstrating "The Assembly Line" pattern.

Different models are used for what they are best at, sharing context via a Session.

1. Groq (Llama 3): High-speed Ideation.
2. Anthropic (Claude 3.5 Sonnet): High-IQ Architecture & Reasoning.
3. OpenAI (GPT-4o): Execution & Tool Usage.

Usage: python examples/provider_switching.py
"""

import os

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from allos import Agent, AgentConfig

load_dotenv()
console = Console()
SESSION_FILE = "assembly_line.json"


def run_stage(title, provider, model, prompt, color="blue", no_tools=False):
    """Execute a single stage of agent processing with a specified provider and model.

    This function creates an agent with the given configuration, loads or initializes
    context from a session file, runs the agent with the provided prompt, and displays
    the result in a formatted panel.

    Args:
        title (str): The name/title of the stage to display in output.
        provider (str): The provider name (e.g., 'openai', 'anthropic') for the agent.
        model (str): The model identifier to use within the specified provider.
        prompt (str): The input prompt to send to the agent.
        color (str, optional): The color to use for console output formatting. Defaults to "blue".
        no_tools (bool, optional): Whether to disable tool usage for this agent. Defaults to False.

    Returns:
        str or None: The agent's response text if successful, or None if the provider's
                     required environment variable is not found.

    Raises:
        None explicitly, but may raise exceptions from Agent initialization or execution.

    Side Effects:
        - Prints status messages and results to console with styled formatting.
        - Saves the agent session to SESSION_FILE after execution.
        - Loads existing context from SESSION_FILE if it exists.
    """
    # Check keys
    from allos.providers import ProviderRegistry

    env_var = ProviderRegistry.get_env_var_name(provider)
    if env_var and env_var not in os.environ:
        console.print(f"[yellow]⚠️ Skipping {title}: {env_var} not found.[/]")
        return None

    console.print(f"\n[bold {color}]╔══ STEP: {title} ({provider}/{model}) ══╗[/]")

    # Create config for this stage
    config = AgentConfig(provider_name=provider, model=model, no_tools=no_tools)

    # Load context if exists, otherwise new agent
    if os.path.exists(SESSION_FILE):
        loaded_agent = Agent.load_session(SESSION_FILE)
        # Create new agent with new config + old context
        agent = Agent(config=config, context=loaded_agent.context)
    else:
        agent = Agent(config)

    with console.status(f"[{color}]🤖 {model} is working...[/]", spinner="dots"):
        response = agent.run(prompt)
        agent.save_session(SESSION_FILE)

    console.print(
        Panel(Markdown(response), title=f"{title} Output", border_style=color)
    )
    return response


def main():
    """Execute a multi-stage AI assembly line workflow for generating a Python CLI tool.

    This function orchestrates a three-stage pipeline using different AI providers:
    1. Groq (Speed): Generates 3 ideas for a Python CLI tool
    2. Anthropic (Intelligence): Selects the best idea and creates a specification
    3. OpenAI (Execution): Implements the main.py code based on the specification
    The session file is cleaned up before and after execution to ensure a fresh start.
    Displays progress using formatted console panels with stage-specific styling.

    Returns:
        None
    """
    if os.path.exists(SESSION_FILE):
        os.remove(SESSION_FILE)

    console.print(
        Panel.fit("[bold white]🏭 The AI Assembly Line[/]", style="bold magenta")
    )

    # Stage 1: Speed (Groq)
    run_stage(
        "The Ideator",
        "groq",
        "llama-3.3-70b-versatile",
        "Generate 3 ideas for a Python CLI tool. Be brief.",
        color="cyan",
        no_tools=True,
    )

    # Stage 2: Intelligence (Anthropic)
    run_stage(
        "The Architect",
        "anthropic",
        "claude-3-7-sonnet-20250219",
        "Pick the best idea and write a spec.",
        color="magenta",
        no_tools=True,
    )

    # Stage 3: Execution (OpenAI)
    run_stage(
        "The Engineer",
        "openai",
        "gpt-4o",
        "Write the `main.py` code based on the spec.",
        color="green",
        no_tools=False,
    )

    if os.path.exists(SESSION_FILE):
        os.remove(SESSION_FILE)
    console.print("\n[bold green]✅ Assembly Line Complete![/]")


if __name__ == "__main__":
    main()
