"""Comprehensive native Cohere usage for the Allos SDK.

This example demonstrates what native `provider_name="cohere"` can do:
1. Provider-direct chat
2. Provider-direct streaming
3. Provider-direct tool-calling
4. Agent.run() with Cohere
5. Agent.stream_run() with Cohere
6. Normalized metadata inspection

Usage:
  uv pip install "allos-agent-sdk[cohere]"
  export COHERE_API_KEY="..."
  python examples/cohere_usage.py

Optional:
  export ALLOS_COHERE_MODEL="command-r7b-12-2024"
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

from rich.console import Console
from rich.panel import Panel

from allos import Agent, AgentConfig
from allos.providers import Message, MessageRole, ProviderRegistry
from allos.providers.metadata import Metadata
from allos.tools.base import BaseTool, ToolParameter

console = Console()


class GetWeatherTool(BaseTool):
    """Simple tool schema used to trigger tool calling."""

    name = "get_current_weather"
    description = "Get weather for a location."
    parameters = [
        ToolParameter(
            name="location",
            type="string",
            description="City and state, e.g. Boston, MA",
            required=True,
        )
    ]

    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        """Return deterministic fixture output for demo roundtrip."""
        return {
            "location": kwargs.get("location", "unknown"),
            "temperature_c": 21,
            "condition": "Sunny",
            "source": "fixture",
        }


def print_metadata(metadata: Metadata, label: str) -> None:
    """Pretty-print a compact metadata summary."""
    console.print(
        Panel.fit(
            (
                f"[bold]{label}[/bold]\n"
                f"provider: {metadata.model.provider}\n"
                f"model: {metadata.model.model_id}\n"
                f"input_tokens: {metadata.usage.input_tokens}\n"
                f"output_tokens: {metadata.usage.output_tokens}\n"
                f"total_tokens: {metadata.usage.total_tokens}\n"
                f"latency_ms: {metadata.latency.total_duration_ms}\n"
                f"finish_reason: {metadata.quality_signals.finish_reason}"
            ),
            border_style="magenta",
        )
    )


def run_provider_chat(model: str) -> None:
    """Run a basic provider-direct chat call."""
    provider = ProviderRegistry.get_provider("cohere", model=model)
    messages = [
        Message(
            role=MessageRole.SYSTEM,
            content="You are concise. Return one short sentence.",
        ),
        Message(
            role=MessageRole.USER,
            content="What is Allos in one sentence?",
        ),
    ]

    response = provider.chat(messages, temperature=0, max_tokens=120)
    console.print(
        Panel(
            response.content or "(empty response)",
            title="1) Provider Chat",
            border_style="green",
        )
    )
    print_metadata(response.metadata, "Provider Chat Metadata")


def run_provider_stream(model: str) -> None:
    """Run provider-direct streaming and print chunk anatomy."""
    provider = ProviderRegistry.get_provider("cohere", model=model)
    messages = [
        Message(
            role=MessageRole.USER,
            content="Write a two-line summary of why streaming is useful.",
        )
    ]

    console.print(Panel("2) Provider Streaming", border_style="cyan"))
    final_metadata = None
    for chunk in provider.stream_chat(messages, temperature=0, max_tokens=120):
        if chunk.error:
            console.print(f"[red]stream error:[/] {chunk.error}")
            continue
        if chunk.content:
            console.print(chunk.content, end="")
        if chunk.tool_call_start:
            console.print(f"\n[cyan]tool_start:[/] {chunk.tool_call_start}")
        if chunk.tool_call_delta:
            console.print(f"\n[cyan]tool_delta:[/] {chunk.tool_call_delta}")
        if chunk.tool_call_done:
            console.print(f"\n[green]tool_done:[/] {chunk.tool_call_done}")
        if chunk.final_metadata:
            final_metadata = chunk.final_metadata
    console.print()

    if final_metadata:
        print_metadata(final_metadata, "Provider Stream Final Metadata")


def run_provider_tool_calling(model: str) -> None:
    """Run provider-direct tool calling and optional roundtrip."""
    provider = ProviderRegistry.get_provider("cohere", model=model)
    tool = GetWeatherTool()
    tools = [tool]
    messages = [
        Message(
            role=MessageRole.USER,
            content=(
                "Use get_current_weather for Boston, MA and return a tool call. "
                "Do not answer from memory."
            ),
        )
    ]

    response = provider.chat(messages, tools=tools, temperature=0, max_tokens=120)
    console.print(Panel("3) Provider Tool Calling", border_style="yellow"))
    print_metadata(response.metadata, "Provider Tool Call Metadata")

    if not response.tool_calls:
        console.print(
            "[yellow]No tool call emitted in this run (model-dependent behavior).[/]"
        )
        return

    tc = response.tool_calls[0]
    console.print(f"[green]tool requested:[/] {tc.name} args={tc.arguments}")

    tool_output = tool.execute(**tc.arguments)
    followup_messages = messages + [
        Message(
            role=MessageRole.ASSISTANT,
            content=response.content,
            tool_calls=response.tool_calls,
        ),
        Message(role=MessageRole.TOOL, tool_call_id=tc.id, content=json.dumps(tool_output)),
        Message(
            role=MessageRole.USER,
            content="Now summarize the weather in one short sentence.",
        ),
    ]

    followup = provider.chat(followup_messages, tools=tools, temperature=0, max_tokens=120)
    console.print(
        Panel(
            followup.content or "(empty response)",
            title="Tool Roundtrip Result",
            border_style="green",
        )
    )
    print_metadata(followup.metadata, "Provider Tool Roundtrip Metadata")


def run_agent_examples(model: str) -> None:
    """Run Agent.run and Agent.stream_run with Cohere."""
    config = AgentConfig(
        provider_name="cohere",
        model=model,
        no_tools=True,
        provider_call_options={"temperature": 0, "max_tokens": 120},
    )
    agent = Agent(config)

    sync_result = agent.run(
        "Return exactly this token and nothing else: AGENT_COHERE_RUN_OK"
    )
    console.print(
        Panel(sync_result, title="4) Agent.run()", border_style="blue")
    )
    if agent.last_run_metadata:
        print_metadata(agent.last_run_metadata, "Agent.run Metadata")

    console.print(Panel("5) Agent.stream_run()", border_style="blue"))
    stream_text = []
    final_stream_metadata = None
    for chunk in agent.stream_run(
        "Return exactly this token and nothing else: AGENT_COHERE_STREAM_OK"
    ):
        if chunk.content:
            stream_text.append(chunk.content)
            console.print(chunk.content, end="")
        if chunk.error:
            console.print(f"\n[red]agent stream error:[/] {chunk.error}")
        if chunk.final_metadata:
            final_stream_metadata = chunk.final_metadata
    console.print()

    if final_stream_metadata:
        print_metadata(final_stream_metadata, "Agent.stream_run Final Metadata")


def main() -> None:
    """Execute all Cohere demo sections."""
    if not os.getenv("COHERE_API_KEY"):
        console.print("[red]COHERE_API_KEY is not set.[/]")
        console.print("Set it and rerun this example.")
        return

    model = os.getenv("ALLOS_COHERE_MODEL", "command-r7b-12-2024")
    console.print(
        Panel.fit(
            f"[bold]Allos Native Cohere Demo[/bold]\nmodel={model}",
            border_style="magenta",
        )
    )

    run_provider_chat(model)
    run_provider_stream(model)
    run_provider_tool_calling(model)
    run_agent_examples(model)

    console.print(
        Panel.fit(
            "Completed Cohere native integration demo.",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
