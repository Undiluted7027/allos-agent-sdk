# examples/streaming_chunks.py

"""Inspect streaming chunk anatomy from a provider.

This example demonstrates provider-direct streaming and prints:
- content deltas
- tool call start/delta/done events (when emitted)
- final metadata chunk

Usage:
  uv pip install "allos-agent-sdk[openai]" python-dotenv
  export OPENAI_API_KEY="sk-..."
  python examples/streaming_chunks.py
"""

import os
from typing import Any, Dict

from dotenv import load_dotenv
from rich.console import Console

from allos.providers import Message, MessageRole, ProviderRegistry
from allos.tools import BaseTool, ToolParameter

console = Console()


class GetWeatherTool(BaseTool):
    """Simple tool schema used to trigger tool-calling streaming behavior."""

    name = "get_current_weather"
    description = "Get the current weather in a location."
    parameters = [
        ToolParameter(
            name="location",
            type="string",
            description="City and region, e.g., Boston, MA",
            required=True,
        )
    ]

    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        """Execute the tool and return a mock weather response.

        Args:
            **kwargs: Tool parameters (location, etc.).

        Returns:
            A dictionary with execution status and echoed parameters.
        """
        return {"ok": True, "echo": kwargs}


def main() -> None:
    """Run the streaming chunks demo."""
    load_dotenv()

    provider_name = os.getenv("ALLOS_STREAM_PROVIDER", "openai")
    model = os.getenv("ALLOS_STREAM_MODEL", "gpt-4o")
    env_var = ProviderRegistry.get_env_var_name(provider_name)
    if env_var and not os.getenv(env_var):
        console.print(
            f"[yellow]{env_var} not set. Skipping stream demo for provider '{provider_name}'.[/]"
        )
        return

    provider = ProviderRegistry.get_provider(provider_name, model=model)
    messages = [
        Message(
            role=MessageRole.USER,
            content=(
                "Use get_current_weather for Boston, MA. "
                "Then respond with one short sentence."
            ),
        )
    ]
    tools = [GetWeatherTool()]

    console.print(f"[bold]Streaming with {provider_name}/{model}[/bold]")
    for chunk in provider.stream_chat(messages, tools=tools, temperature=0):
        if chunk.error:
            console.print(f"\n[red]error:[/red] {chunk.error}")
            break
        if chunk.content:
            console.print(chunk.content, end="")
        if chunk.tool_call_start:
            console.print(f"\n[cyan]tool_start:[/cyan] {chunk.tool_call_start}")
        if chunk.tool_call_delta:
            console.print(f"\n[cyan]tool_delta:[/cyan] {chunk.tool_call_delta}")
        if chunk.tool_call_done:
            console.print(f"\n[green]tool_done:[/green] {chunk.tool_call_done}")
        if chunk.final_metadata:
            meta = chunk.final_metadata
            console.print(
                "\n[bold magenta]final_metadata:[/bold magenta] "
                f"provider={meta.model.provider}, model={meta.model.model_id}, "
                f"tokens={meta.usage.total_tokens}"
            )
    console.print()


if __name__ == "__main__":
    main()
