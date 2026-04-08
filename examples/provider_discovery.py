# examples/provider_discovery.py

"""Discover providers and environment readiness.

This script helps you inspect:
1. All currently available provider names from ProviderRegistry
2. Whether each provider is configured in the current environment
3. Which providers are native vs OpenAI-compatible aliases

Usage:
  python examples/provider_discovery.py
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from allos.providers import ProviderRegistry
from allos.providers.registry import OPENAI_COMPATIBLE_PROVIDERS

console = Console()


def _provider_kind(name: str) -> str:
    """Return display category for a provider name."""
    if name in OPENAI_COMPATIBLE_PROVIDERS:
        return "alias"
    return "native"


def main() -> None:
    """Print provider inventory and readiness."""
    providers = ProviderRegistry.list_providers()
    alias_count = len([p for p in providers if p in OPENAI_COMPATIBLE_PROVIDERS])
    native_count = len(providers) - alias_count

    table = Table(title="Allos Provider Discovery", expand=True)
    table.add_column("Provider", style="cyan", no_wrap=True)
    table.add_column("Kind", style="magenta")
    table.add_column("Configured", style="green")
    table.add_column("Details", style="dim")

    ready_count = 0
    for name in providers:
        is_ready, details = ProviderRegistry.check_provider_env(name)
        ready = "[green]yes[/]" if is_ready else "[red]no[/]"
        if is_ready:
            ready_count += 1
        table.add_row(name, _provider_kind(name), ready, details)

    console.print(table)
    console.print(
        "[bold]Summary:[/bold] "
        f"{len(providers)} total providers "
        f"({native_count} native, {alias_count} aliases), "
        f"{ready_count} configured."
    )
    console.print(
        "[dim]Tip: `allos --active-providers` and `allos --list-providers` "
        "show the same data from the CLI.[/dim]"
    )


if __name__ == "__main__":
    main()
