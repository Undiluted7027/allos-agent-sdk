# examples/provider_call_options.py

"""Demonstrate provider call options with Agent.run() and Agent.stream_run().

This example covers:
1. Default provider options via AgentConfig.provider_call_options
2. Per-run overrides via run(..., **kwargs)
3. Streaming overrides via stream_run(..., **kwargs)
4. Reserved key validation for 'messages' and 'tools'

Usage:
  uv pip install "allos-agent-sdk[openai]" python-dotenv
  export OPENAI_API_KEY="sk-..."
  python examples/provider_call_options.py
"""

import os

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from allos import Agent, AgentConfig
from allos.utils.errors import AllosError

console = Console()


def main() -> None:
    """Run provider call options examples."""
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        console.print(
            "[yellow]OPENAI_API_KEY not found. Skipping runtime options example.[/]"
        )
        return

    console.print(
        Panel.fit(
            "[bold]Agent Provider Call Options[/]",
            border_style="cyan",
        )
    )

    config = AgentConfig(
        provider_name="openai",
        model="gpt-4o",
        no_tools=True,
        max_tokens=120,
        provider_call_options={
            "temperature": 0.1,
            "top_p": 0.95,
        },
    )
    agent = Agent(config)

    console.print("\n[bold]1) run() with runtime override[/]")
    response = agent.run(
        "Summarize why deterministic sampling can be useful in one sentence.",
        temperature=0.7,  # Overrides config default temperature=0.1
    )
    console.print(Panel(response, title="run() response", border_style="green"))

    console.print("\n[bold]2) stream_run() with runtime override[/]")
    console.print("[green]stream:[/green] ", end="")
    for chunk in agent.stream_run(
        "Give one short tip for writing clear technical docs.", top_p=0.8
    ):
        if chunk.content:
            console.print(chunk.content, end="")
    console.print()

    console.print("\n[bold]3) Reserved-key validation[/]")
    for bad_key in ("tools", "messages"):
        try:
            agent.run("This should fail fast.", **{bad_key: "invalid"})  # type: ignore[arg-type]
        except AllosError as e:
            console.print(f"[yellow]Expected error for '{bad_key}':[/] {e}")


if __name__ == "__main__":
    main()
