"""Use Cohere through the OpenAI-compatible endpoint (`cohere_compat`).

This example demonstrates compatibility-mode usage via the registry alias
`provider_name="cohere_compat"`.

Use this when you explicitly want Chat Completions compatibility behavior.
For native Cohere features and tool-calling flows, see `examples/cohere_usage.py`.

Usage:
  uv pip install "allos-agent-sdk[openai]" python-dotenv
  export COHERE_API_KEY="..."
  python examples/cohere_compat.py

Optional:
  export ALLOS_COHERE_COMPAT_MODEL="command-r7b-12-2024"
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from allos import Agent, AgentConfig

load_dotenv()
console = Console()


def main() -> None:
    """Run sync and streaming demos against Cohere compat endpoint."""
    if not os.getenv("COHERE_API_KEY"):
        console.print("[red]COHERE_API_KEY is not set.[/]")
        return

    model = os.getenv("ALLOS_COHERE_COMPAT_MODEL", "command-r7b-12-2024")
    console.print(
        Panel.fit(
            f"[bold]Cohere Compat Demo[/bold]\nprovider=cohere_compat\nmodel={model}",
            border_style="cyan",
        )
    )

    agent = Agent(
        AgentConfig(
            provider_name="cohere_compat",
            model=model,
            no_tools=True,
            provider_call_options={"temperature": 0, "max_tokens": 160},
        )
    )

    sync_response = agent.run(
        "Explain in one sentence what an API compatibility layer is."
    )
    console.print(Panel(sync_response, title="1) run()", border_style="green"))

    if agent.last_run_metadata:
        meta = agent.last_run_metadata
        console.print(
            f"[dim]provider={meta.model.provider} model={meta.model.model_id} "
            f"tokens={meta.usage.total_tokens} latency_ms={meta.latency.total_duration_ms}[/]"
        )

    console.print(Panel("2) stream_run()", border_style="blue"))
    final_meta = None
    for chunk in agent.stream_run("Write a short 2-line summary of deterministic decoding."):
        if chunk.content:
            console.print(chunk.content, end="")
        if chunk.error:
            console.print(f"\n[red]error:[/] {chunk.error}")
        if chunk.final_metadata:
            final_meta = chunk.final_metadata
    console.print()

    if final_meta:
        console.print(
            f"[dim]stream provider={final_meta.model.provider} "
            f"tokens={final_meta.usage.total_tokens} "
            f"ttft_ms={final_meta.latency.time_to_first_token_ms}[/]"
        )

    console.print(
        Panel.fit(
            "Completed Cohere compat demo.\n"
            "Tip: use `provider_name=\"cohere\"` for native Cohere behavior.",
            border_style="magenta",
        )
    )


if __name__ == "__main__":
    main()
