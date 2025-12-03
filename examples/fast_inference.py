# examples/fast_inference.py

"""
A real-time benchmark comparing the performance of different providers.
Demonstrates the raw speed of Allos when connected to fast inference engines like Groq.
"""

import os
import time

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from allos import Agent, AgentConfig

load_dotenv()
console = Console()

# The heavy prompt
PROMPT = "Write a comprehensive history of the Internet, covering ARPANET to Web3. Approx 500 words."

# Candidates
PROVIDERS = [
    {"name": "OpenAI", "provider": "openai", "model": "gpt-4o"},
    {"name": "Anthropic", "provider": "anthropic", "model": "claude-3-haiku-20240307"},
    {"name": "Groq", "provider": "groq", "model": "llama-3.1-8b-instant"},
    {"name": "Mistral", "provider": "mistral", "model": "mistral-small-latest"},
]


def benchmark():
    table = Table(title="🏎️  Allos Inference Speedometer", expand=True)
    table.add_column("Provider", style="cyan")
    table.add_column("Model", style="dim")
    table.add_column("Time (s)", justify="right")
    table.add_column("Est. Tokens", justify="right")
    table.add_column("Speed (Tokens/s)", justify="right", style="bold")

    results = []

    for p in PROVIDERS:
        # Check env vars manually to skip cleanly
        # (This is simplified, the registry has smarter logic but we want to be explicit for the demo)
        key_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "groq": "GROQ_API_KEY",
            "mistral": "MISTRAL_API_KEY",
        }
        if key_map[p["provider"]] not in os.environ:
            continue

        config = AgentConfig(
            provider_name=p["provider"],
            model=p["model"],
            no_tools=True,  # Disable tools for pure generation speed
        )

        console.print(f"[dim]Testing {p['name']}...[/dim]")

        try:
            agent = Agent(config)
            start = time.time()
            response = agent.run(PROMPT)
            duration = time.time() - start

            # Estimate tokens (approx 4 chars per token)
            tokens = len(response) / 4
            tps = tokens / duration

            # Color code speed
            speed_style = "green" if tps > 100 else ("yellow" if tps > 50 else "white")

            table.add_row(
                p["name"],
                p["model"],
                f"{duration:.2f}",
                f"{int(tokens)}",
                f"[{speed_style}]{tps:.1f}[/]",
            )
            results.append((p["name"], tps))

        except Exception:
            table.add_row(p["name"], p["model"], "Error", "-", "-")

    console.print("\n")
    console.print(table)

    if results:
        winner = max(results, key=lambda x: x[1])
        console.print(
            f"\n🏆 Fastest Provider: [bold green]{winner[0]}[/] ({winner[1]:.1f} t/s)"
        )


if __name__ == "__main__":
    benchmark()
