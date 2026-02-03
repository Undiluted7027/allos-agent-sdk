# examples/ollama_compat.py

"""The Compatibility Layer.

Demonstrates using Ollama via the OpenAI-compatible endpoint (ollama_compat).

This is an alternative to the native 'ollama' provider. Use this when:
- You want a simpler setup without the ollama Python library
- You're using a model that doesn't support native tool calling
- You prefer the OpenAI-compatible API interface

For full tool calling support, see examples/local_models.py which uses
the native 'ollama' provider.
"""

from rich.console import Console
from rich.panel import Panel

from allos import Agent, AgentConfig

console = Console()


def main():
    """Run the Ollama compatibility mode example.

    Demonstrates connecting to Ollama via its OpenAI-compatible endpoint
    and sending a simple query without tool support.
    """
    console.print(
        Panel.fit(
            "[bold cyan]🔄 Ollama Compatibility Mode (OpenAI-Compatible Endpoint)[/]",
            border_style="cyan",
        )
    )

    # Using ollama_compat connects to Ollama's OpenAI-compatible endpoint
    # Default: http://localhost:11434/v1
    config = AgentConfig(
        provider_name="ollama_compat",
        model="mistral:latest",  # Requires: ollama pull mistral
        no_tools=True,  # Disable tools for compatibility mode (simpler)
    )

    try:
        agent = Agent(config)

        prompt = "Explain the difference between compiled and interpreted languages in 2-3 sentences."

        console.print(f"\n[bold]User:[/bold] {prompt}")
        console.print(
            "[dim](Sending to Ollama via OpenAI-compatible endpoint at localhost:11434/v1)[/dim]"
        )

        response = agent.run(prompt)

        console.print(
            Panel(response, title="Ollama Compat Response", border_style="cyan")
        )

    except Exception as e:
        console.print(f"\n[bold red]Connection Failed:[/bold] {e}")
        console.print("Ensure Ollama is running: [bold]ollama serve[/]")
        console.print("Ensure model is pulled: [bold]ollama pull mistral[/]")


def with_custom_host():
    """Example: Connect to Ollama on a different machine."""
    console.print(
        Panel.fit(
            "[bold cyan]🌐 Ollama Compat with Custom Host[/]",
            border_style="cyan",
        )
    )

    # Connect to Ollama running on another machine
    config = AgentConfig(
        provider_name="ollama_compat",
        model="llama3",
        base_url="http://192.168.1.100:11434/v1",  # Custom host
        no_tools=True,
    )

    try:
        agent = Agent(config)
        response = agent.run("Hello from a remote connection!")
        console.print(
            Panel(response, title="Remote Ollama Response", border_style="cyan")
        )
    except Exception as e:
        console.print(f"[red]Failed to connect to remote Ollama: {e}[/]")


if __name__ == "__main__":
    main()
    # Uncomment to test custom host:
    # with_custom_host()
