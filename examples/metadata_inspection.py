# examples/metadata_inspection.py

"""Inspect normalized and provider-specific metadata from Agent runs.

Usage:
  uv pip install "allos-agent-sdk[openai]" python-dotenv
  export OPENAI_API_KEY="sk-..."
  python examples/metadata_inspection.py
"""

import os
from typing import Any, Dict

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from allos import Agent, AgentConfig
from allos.providers import ProviderRegistry

console = Console()


def _extract_provider_specific(provider_specific_obj: Any) -> Dict[str, Any]:
    """Return only non-null provider-specific sections as a dict."""
    return {k: v for k, v in provider_specific_obj.items() if v is not None}


def main() -> None:
    """Run a short chat and print metadata details."""
    load_dotenv()

    provider_name = os.getenv("ALLOS_METADATA_PROVIDER", "openai")
    model = os.getenv("ALLOS_METADATA_MODEL", "gpt-4o")
    env_var = ProviderRegistry.get_env_var_name(provider_name)
    if env_var and not os.getenv(env_var):
        console.print(
            f"[yellow]{env_var} not set. Skipping metadata inspection for '{provider_name}'.[/]"
        )
        return

    agent = Agent(
        AgentConfig(
            provider_name=provider_name,
            model=model,
            no_tools=True,
        )
    )

    prompt = "Explain idempotency in distributed systems in two short bullets."
    response = agent.run(prompt)
    meta = agent.last_run_metadata
    if not meta:
        console.print("[red]No metadata found on agent.last_run_metadata[/red]")
        return

    provider_specific = _extract_provider_specific(
        meta.provider_specific.model_json_schema()
    )
    cost = (
        meta.usage.estimated_cost.total_usd
        if meta.usage.estimated_cost is not None
        else None
    )

    summary = {
        "provider": meta.model.provider,
        "model_id": meta.model.model_id,
        "input_tokens": meta.usage.input_tokens,
        "output_tokens": meta.usage.output_tokens,
        "total_tokens": meta.usage.total_tokens,
        "duration_ms": meta.latency.total_duration_ms,
        "ttft_ms": meta.latency.time_to_first_token_ms,
        "estimated_cost_usd": cost,
        "turns": meta.turns.total_turns,
        "tool_calls": meta.tools.total_tool_calls,
        "provider_specific": provider_specific,
    }

    console.print(Panel(response, title="Response", border_style="green"))
    console.print(Panel(str(summary), title="Metadata Summary", border_style="blue"))


if __name__ == "__main__":
    main()
