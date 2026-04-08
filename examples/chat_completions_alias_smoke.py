# examples/chat_completions_alias_smoke.py

"""Smoke-test any OpenAI-compatible alias provider.

This example is useful when validating one provider alias at a time without
changing code. It uses the regular Agent API with `no_tools=True`.

Usage:
  uv pip install "allos-agent-sdk[openai]" python-dotenv
  export ALLOS_ALIAS_PROVIDER="groq"
  export ALLOS_ALIAS_MODEL="llama-3.1-8b-instant"
  export GROQ_API_KEY="..."
  python examples/chat_completions_alias_smoke.py

Optional:
  export ALLOS_ALIAS_PROMPT="Explain idempotency in one sentence."
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from allos import Agent, AgentConfig
from allos.providers import ProviderRegistry
from allos.providers.registry import OPENAI_COMPATIBLE_PROVIDERS

console = Console()


DEFAULT_MODELS = {
    "groq": "llama-3.1-8b-instant",
    "together": "Qwen/Qwen2.5-7B-Instruct-Turbo",
    "mistral": "mistral-small-latest",
    "deepseek": "deepseek-chat",
    "openrouter": "openai/gpt-4o-mini",
    "portkey": "openai/gpt-4o-mini",
    "cohere_compat": "command-r7b-12-2024",
    "ollama_compat": "llama3.1:latest",
}


def main() -> None:
    """Run a single chat against a selected alias provider."""
    load_dotenv()

    provider_name = os.getenv("ALLOS_ALIAS_PROVIDER", "groq")
    if provider_name not in OPENAI_COMPATIBLE_PROVIDERS:
        console.print(
            f"[red]'{provider_name}' is not an OpenAI-compatible alias in this SDK build.[/]"
        )
        console.print(
            "[dim]Tip: use `python examples/provider_discovery.py` or "
            "`allos --list-providers` to inspect available names.[/]"
        )
        return

    model = os.getenv(
        "ALLOS_ALIAS_MODEL",
        DEFAULT_MODELS.get(provider_name, "gpt-4o-mini"),
    )
    prompt = os.getenv(
        "ALLOS_ALIAS_PROMPT",
        "Explain the purpose of an API gateway in one concise sentence.",
    )

    env_var = ProviderRegistry.get_env_var_name(provider_name)
    if env_var and not os.getenv(env_var):
        console.print(
            f"[yellow]{env_var} not set. Skipping alias smoke test for '{provider_name}'.[/]"
        )
        return

    console.print(
        Panel.fit(
            "[bold]Alias Smoke Test[/bold]\n"
            f"provider={provider_name}\n"
            f"model={model}",
            border_style="cyan",
        )
    )

    agent = Agent(
        AgentConfig(
            provider_name=provider_name,
            model=model,
            no_tools=True,
            provider_call_options={"temperature": 0, "max_tokens": 160},
        )
    )
    response = agent.run(prompt)

    console.print(Panel(response, title="Response", border_style="green"))

    if agent.last_run_metadata:
        meta = agent.last_run_metadata
        console.print(
            "[dim]"
            f"provider={meta.model.provider} "
            f"model={meta.model.model_id} "
            f"tokens={meta.usage.total_tokens} "
            f"latency_ms={meta.latency.total_duration_ms}"
            "[/]"
        )


if __name__ == "__main__":
    main()
