# examples/fast_inference.py

"""A real-time benchmark comparing the performance of different providers.

Demonstrates the raw speed of Allos using internal Metadata metrics.
"""

import os

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from allos import Agent, AgentConfig
from allos.providers import ProviderRegistry

load_dotenv()
console = Console()

PROMPT = "Write a comprehensive history of the Internet, covering ARPANET to Web3. Approx 300 words."

# Candidates
PROVIDERS = [
    {"name": "OpenAI", "provider": "openai", "model": "gpt-4o"},
    {"name": "Anthropic", "provider": "anthropic", "model": "claude-3-haiku-20240307"},
    {"name": "Groq", "provider": "groq", "model": "llama-3.1-8b-instant"},
    {"name": "Mistral", "provider": "mistral", "model": "mistral-small-latest"},
]


def benchmark():
    """Run inference benchmarks across multiple AI providers and display results.

    Executes inference tests on configured providers/models, measures performance metrics,
    and displays a formatted table comparing:
    - Provider names
    - Model identifiers
    - Total execution time
    - Output token counts
    - Tokens per second (speed)
    Color-codes speed results based on throughput thresholds:
    - Green: > 100 tokens/second
    - Yellow: > 50 tokens/second
    - White: <= 50 tokens/second
    Skips providers with missing/invalid environment variables.
    Handles errors gracefully, displaying error messages in results table.
    After all benchmarks complete, displays the fastest provider with its throughput.

    Returns:
        None
    """
    table = Table(title="🏎️  Allos Inference Speedometer", expand=True)
    table.add_column("Provider", style="cyan")
    table.add_column("Model", style="dim")
    table.add_column("Total Time", justify="right")
    table.add_column("Output Tokens", justify="right")
    table.add_column("Speed (Tokens/s)", justify="right", style="bold")

    results = []

    for p in PROVIDERS:
        # Check env var validity using Registry logic
        env_var = ProviderRegistry.get_env_var_name(p["provider"])
        if env_var and env_var not in os.environ:
            continue

        config = AgentConfig(
            provider_name=p["provider"],
            model=p["model"],
            no_tools=True,
        )

        console.print(f"[dim]Testing {p['name']}...[/dim]")

        try:
            agent = Agent(config)
            agent.run(PROMPT)

            # Get precise metrics from the agent's last run metadata
            meta = agent.last_run_metadata
            if not meta:
                raise ValueError("No metadata returned")

            duration = meta.latency.total_duration_ms / 1000.0
            tokens = meta.usage.output_tokens
            tps = tokens / duration if duration > 0 else 0

            # Color code speed
            speed_style = "green" if tps > 100 else ("yellow" if tps > 50 else "white")

            table.add_row(
                p["name"],
                p["model"],
                f"{duration:.2f}s",
                f"{tokens}",
                f"[{speed_style}]{tps:.1f}[/]",
            )
            results.append((p["name"], tps))

        except Exception as e:
            table.add_row(p["name"], p["model"], "Error", "-", str(e)[:20])

    console.print("\n")
    console.print(table)

    if results:
        winner = max(results, key=lambda x: x[1])
        console.print(
            f"\n🏆 Fastest Provider: [bold green]{winner[0]}[/] ({winner[1]:.1f} t/s)"
        )


if __name__ == "__main__":
    benchmark()
