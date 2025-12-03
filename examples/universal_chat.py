# examples/universal_chat.py

"""
The Model Arena. Ask one question, get answers from everyone.
Demonstrates the power of the unified `Agent` interface.
"""

import os

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from allos import Agent, AgentConfig
from allos.providers import ProviderRegistry

load_dotenv()
console = Console()


def main():
    console.print(Panel.fit("[bold]⚔️  Allos Model Arena[/]", border_style="red"))

    prompt = console.input("\n[bold cyan]Enter a complex question for the arena: [/]")
    if not prompt:
        return

    # Define contenders
    contenders = [
        ("groq", "llama-3.3-70b-versatile", "Groq (Llama 3.3)"),
        ("openai", "gpt-4o", "OpenAI (GPT-4o)"),
        ("anthropic", "claude-3-7-sonnet-20250219", "Anthropic (Sonnet)"),
        ("mistral", "mistral-large-latest", "Mistral (Large)"),
    ]

    for provider, model, title in contenders:
        env_var = ProviderRegistry.get_env_var_name(provider)
        if env_var and env_var not in os.environ:
            continue

        console.print(f"\n[bold]Asking {title}...[/]")
        try:
            config = AgentConfig(provider_name=provider, model=model, no_tools=True)
            agent = Agent(config)
            response = agent.run(prompt)

            console.print(
                Panel(
                    Markdown(response), title=title, subtitle=model, border_style="blue"
                )
            )
        except Exception as e:
            console.print(f"[red]Error from {title}: {e}[/]")


if __name__ == "__main__":
    main()
