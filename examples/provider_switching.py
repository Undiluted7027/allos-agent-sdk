# examples/provider_switching.py

"""
A sophisticated example demonstrating "The Assembly Line" pattern.
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
    # Check keys
    env_map = {
        "groq": "GROQ_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
    }
    if env_map.get(provider) not in os.environ:
        console.print(f"[yellow]⚠️ Skipping {title}: {env_map[provider]} not found.[/]")
        return None

    console.print(f"\n[bold {color}]╔══ STEP: {title} ({provider}/{model}) ══╗[/]")

    # Load previous state if available
    if os.path.exists(SESSION_FILE):
        agent = Agent.load_session(SESSION_FILE)
        # Reconfigure for new provider
        agent.config.provider_name = provider
        agent.config.model = model
        agent.config.no_tools = no_tools
        agent.config.base_url = None  # Reset potential custom URLs
        # Re-init internals
        agent.__init__(agent.config, agent.context)
    else:
        config = AgentConfig(provider_name=provider, model=model, no_tools=no_tools)
        agent = Agent(config)

    with console.status(f"[{color}]🤖 {model} is working...[/]", spinner="dots"):
        response = agent.run(prompt)
        agent.save_session(SESSION_FILE)

    console.print(
        Panel(Markdown(response), title=f"{title} Output", border_style=color)
    )
    return response


def main():
    if os.path.exists(SESSION_FILE):
        os.remove(SESSION_FILE)

    console.print(
        Panel.fit("[bold white]🏭 The AI Assembly Line[/]", style="bold magenta")
    )

    # Stage 1: Speed (Groq)
    # Fast iteration to generate raw ideas
    run_stage(
        "The Ideator",
        "groq",
        "llama-3.3-70b-versatile",
        "Generate 3 distinct ideas for a Python CLI tool that helps developers. Be brief.",
        color="cyan",
        no_tools=True,
    )

    # Stage 2: Intelligence (Anthropic)
    # High context reasoning to pick the best one and architect it
    run_stage(
        "The Architect",
        "anthropic",
        "claude-3-7-sonnet-20250219",
        "Review the ideas above. Pick the most useful one and write a detailed technical specification for it. Do not write code yet.",
        color="magenta",
        no_tools=True,
    )

    # Stage 3: Execution (OpenAI)
    # Robust tool use to write the file (Simulated here for visual clarity)
    run_stage(
        "The Engineer: This one takes a while",
        "openai",
        "gpt-5-nano-2025-08-07",
        "Based on the specification, write the `main.py` file for this tool. Use Markdown.",
        color="green",
        no_tools=False,
    )

    if os.path.exists(SESSION_FILE):
        os.remove(SESSION_FILE)
    console.print("\n[bold green]✅ Assembly Line Complete![/]")


if __name__ == "__main__":
    main()
