#!/usr/bin/env python3
# examples/ollama_usage.py

"""Ollama Complete Usage Guide.

This comprehensive example demonstrates all aspects of using Ollama with Allos,
comparing native provider vs chat completions compatibility endpoint.

Topics covered:
1. Basic local model usage (native vs compat)
2. Model comparison (native vs API providers)
3. Streaming examples
4. Tool calling capabilities
5. Model warm-up detection and metadata
6. Streaming retry configuration
7. When to use which approach

Requirements:
- Ollama installed and running: `ollama serve`
- Models pulled: `ollama pull llama3.1` and `ollama pull mistral`

Usage: python examples/ollama_usage.py
"""

import os
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from allos import Agent, AgentConfig
from allos.providers import ollama_running
from allos.providers.base import Message, MessageRole

console = Console()


# ============================================================================
# SECTION 1: BASIC USAGE - Native vs Compat
# ============================================================================


def example_1_basic_native():
    """Example 1: Using the native Ollama provider."""
    console.print(
        Panel.fit(
            "[bold cyan]Example 1: Native Ollama Provider[/]",
            border_style="cyan",
        )
    )

    config = AgentConfig(
        provider_name="ollama",  # Native provider
        model="qwen3:8b",  # Requires: ollama pull llama3.1
        no_tools=True,  # No tools for this simple example
    )

    agent = Agent(config)
    prompt = "What is the capital of France? Answer in one sentence."

    console.print(f"\n[bold]Prompt:[/bold] {prompt}")
    console.print("[dim]Using native ollama Python library...[/dim]\n")

    response = agent.run(prompt)
    console.print(
        Panel(response, title="Native Provider Response", border_style="cyan")
    )


def example_2_basic_compat():
    """Example 2: Using the OpenAI-compatible endpoint."""
    console.print(
        Panel.fit(
            "[bold magenta]Example 2: Ollama Compat (OpenAI-Compatible Endpoint)[/]",
            border_style="magenta",
        )
    )

    config = AgentConfig(
        provider_name="ollama_compat",  # Compatibility mode
        model="mistral:latest",  # Requires: ollama pull mistral
        no_tools=True,
    )

    agent = Agent(config)
    prompt = "What is the capital of France? Answer in one sentence."

    console.print(f"\n[bold]Prompt:[/bold] {prompt}")
    console.print(
        "[dim]Using OpenAI-compatible endpoint at localhost:11434/v1...[/dim]\n"
    )

    response = agent.run(prompt)
    console.print(
        Panel(response, title="Compat Provider Response", border_style="magenta")
    )


# ============================================================================
# SECTION 2: PROVIDER COMPARISON
# ============================================================================


def example_3_provider_comparison():
    """Example 3: Compare local Ollama vs cloud API providers."""
    console.print(
        Panel.fit(
            "[bold yellow]Example 3: Provider Comparison (Local vs Cloud)[/]",
            border_style="yellow",
        )
    )

    prompt = "Explain quantum computing in one sentence."

    # Comparison table
    table = Table(title="Provider Comparison")
    table.add_column("Provider", style="cyan")
    table.add_column("Model", style="magenta")
    table.add_column("Location", style="green")
    table.add_column("Cost", style="yellow")
    table.add_column("Speed", style="blue")

    table.add_row("ollama (native)", "llama3.1", "Local", "Free", "Varies by hardware")
    table.add_row("ollama_compat", "mistral", "Local", "Free", "Varies by hardware")
    table.add_row(
        "openai", "gpt-4o", "Cloud (OpenAI)", "$2.50/$10 per 1M tokens", "Fast"
    )
    table.add_row(
        "anthropic",
        "claude-sonnet-4-5",
        "Cloud (Anthropic)",
        "$3/$15 per 1M tokens",
        "Fast",
    )

    console.print(table)

    # Run local model
    console.print(f"\n[bold]Prompt:[/bold] {prompt}\n")
    console.print("[cyan]Testing with local Ollama...[/cyan]")

    try:
        config = AgentConfig(
            provider_name="ollama",
            model="llama3.1:latest",
            no_tools=True,
        )
        agent = Agent(config)
        response = agent.run(prompt)
        console.print(
            Panel(response, title="Local Model (llama3.1)", border_style="cyan")
        )

        # Show metadata
        if agent.last_run_metadata:
            console.print("\n[dim]Metadata:[/dim]")
            console.print(
                f"  Input tokens: {agent.last_run_metadata.usage.input_tokens}"
            )
            console.print(
                f"  Output tokens: {agent.last_run_metadata.usage.output_tokens}"
            )
            console.print("  Cost: $0.00 (local)")

    except Exception as e:
        console.print(f"[red]Ollama not available: {e}[/red]")


# ============================================================================
# SECTION 3: STREAMING
# ============================================================================


def example_4_streaming_native():
    """Example 4: Streaming with native Ollama provider."""
    console.print(
        Panel.fit(
            "[bold green]Example 4: Streaming with Native Provider[/]",
            border_style="green",
        )
    )

    config = AgentConfig(
        provider_name="ollama",
        model="llama3.1:latest",
        no_tools=True,
    )

    agent = Agent(config)
    prompt = "Write a haiku about artificial intelligence."

    console.print(f"\n[bold]Prompt:[/bold] {prompt}")
    console.print("[dim]Streaming response from local model...[/dim]\n")

    console.print("[green]Response:[/green] ", end="")

    for chunk in agent.stream_run(prompt):
        if chunk.content:
            console.print(chunk.content, end="")

    console.print("\n")


def example_5_streaming_compat():
    """Example 5: Streaming with compat provider."""
    console.print(
        Panel.fit(
            "[bold blue]Example 5: Streaming with Compat Provider[/]",
            border_style="blue",
        )
    )

    config = AgentConfig(
        provider_name="ollama_compat",
        model="mistral:latest",
        no_tools=True,
    )

    agent = Agent(config)
    prompt = "Write a short poem about local AI models."

    console.print(f"\n[bold]Prompt:[/bold] {prompt}")
    console.print("[dim]Streaming via OpenAI-compatible endpoint...[/dim]\n")

    console.print("[blue]Response:[/blue] ", end="")

    for chunk in agent.stream_run(prompt):
        if chunk.content:
            console.print(chunk.content, end="")

    console.print("\n")


# ============================================================================
# SECTION 4: TOOL CALLING
# ============================================================================


def example_6_tool_calling_native():
    """Example 6: Tool calling with native provider."""
    console.print(
        Panel.fit(
            "[bold cyan]Example 6: Tool Calling with Native Provider[/]",
            border_style="cyan",
        )
    )

    # Create a test file
    test_file = "test_data.txt"
    Path(test_file).write_text("The secret code is: ALLOS-2026")

    try:
        config = AgentConfig(
            provider_name="ollama",
            model="qwen3:8b",  # llama3.1 supports native tool calling
            no_tools=False,
            tool_names=["read_file", "list_directory"],
        )

        agent = Agent(config)
        prompt = f"Read the file '{test_file}' and tell me what's inside."

        console.print(f"\n[bold]Prompt:[/bold] {prompt}")
        console.print("[dim]Native Ollama with tool calling support...[/dim]\n")

        response = agent.run(prompt)
        console.print(
            Panel(response, title="Tool Calling Response", border_style="cyan")
        )

        # Show tool usage
        if agent.last_run_metadata and agent.last_run_metadata.tools.tool_calls:
            console.print("\n[dim]Tools used:[/dim]")
            for i, tool_detail in enumerate(
                agent.last_run_metadata.tools.tool_calls, 1
            ):
                console.print(
                    f"  {i}. {tool_detail.tool_name} with args: {tool_detail.arguments}"
                )

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")

    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)


def example_7_tool_calling_compat_limitations():
    """Example 7: Tool calling limitations with compat provider."""
    console.print(
        Panel.fit(
            "[bold yellow]Example 7: Tool Calling with Compat Provider (Limitations)[/]",
            border_style="yellow",
        )
    )

    explanation = """
    The `ollama_compat` provider uses the OpenAI-compatible endpoint, which has
    limitations compared to the native provider:

    ⚠️  [bold white]Limitations:[/bold white]
    - No native tool calling support through this endpoint
    - Must use `no_tools=True` for reliability
    - Best for simple query/response scenarios

    ✅  [bold white]Recommendation:[/bold white]
    - Use native `ollama` provider for tool calling
    - Use `ollama_compat` for simple queries without tools
    """

    console.print(Panel(explanation, border_style="yellow"))

    # Example showing it should be used without tools
    config = AgentConfig(
        provider_name="ollama_compat",
        model="mistral:latest",
        no_tools=True,  # Recommended for compat mode
    )

    agent = Agent(config)
    prompt = "Explain why tool calling might not work with compatibility endpoints."

    console.print(f"\n[bold]Prompt:[/bold] {prompt}\n")

    response = agent.run(prompt)
    console.print(
        Panel(response, title="Compat Mode (No Tools)", border_style="yellow")
    )


# ============================================================================
# SECTION 5: ADVANCED FEATURES
# ============================================================================


def example_8_warmup_detection():
    """Example 8: Understanding model warm-up detection."""
    console.print(
        Panel.fit(
            "[bold yellow]Example 8: Model Warm-Up Detection[/]",
            border_style="yellow",
        )
    )

    explanation = """
    [bold white]What is Model Warm-Up?[/bold white]

    When you make the first request to a model, Ollama loads it into memory (GPU/CPU VRAM).
    This can take 10-30 seconds depending on model size and hardware. Subsequent requests
    are much faster since the model is already loaded.

    [bold white]Allos SDK Features:[/bold white]
    - Automatically detects warm-up events (≥10 seconds on first request)
    - Logs an informative message to help you understand the delay
    - Exposes warm-up info in response metadata for your application to use
    - Tracks per (model, host) combination
    """

    console.print(Panel(explanation, border_style="yellow"))

    config = AgentConfig(
        provider_name="ollama",
        model="llama3.1:latest",
        no_tools=True,
    )

    agent = Agent(config)

    # First request (might trigger warm-up if model not loaded)
    console.print("\n[bold]Making first request...[/bold]")
    console.print(
        "[dim](Watch console for warm-up notice if model needs loading)[/dim]\n"
    )

    response = agent.run("Say hello in one sentence.")

    # Check metadata for warm-up info
    if agent.last_run_metadata and agent.last_run_metadata.provider_specific.ollama:
        ollama_meta = agent.last_run_metadata.provider_specific.ollama
        if ollama_meta.warm_up:
            console.print("\n[yellow]⚠️  Model Warm-Up Detected![/yellow]")
            console.print(f"   Duration: {ollama_meta.warm_up_duration_seconds:.1f}s")
            console.print("   The model was loading into memory.")
            console.print("   [dim]Subsequent requests will be much faster.[/dim]")
        else:
            console.print("\n[green]✓ Model already loaded (fast response)[/green]")
    else:
        console.print("\n[dim]Metadata not available or warm-up info not present[/dim]")

    console.print(f"\n[bold]Response:[/bold] {response}")

    # Second request (should be fast, no warm-up)
    console.print("\n[bold]Making second request...[/bold]")
    console.print("[dim](Should be fast - model already in memory)[/dim]\n")

    response2 = agent.run("Say goodbye in one sentence.")

    if agent.last_run_metadata and agent.last_run_metadata.provider_specific.ollama:
        ollama_meta = agent.last_run_metadata.provider_specific.ollama
        if not ollama_meta.warm_up:
            console.print("\n[green]✓ No warm-up needed (model already loaded)[/green]")

    console.print(f"\n[bold]Response:[/bold] {response2}")


def example_9_streaming_with_retry():
    """Example 9: Streaming with automatic retry on connection failures."""
    console.print(
        Panel.fit(
            "[bold red]Example 9: Streaming Retry Configuration[/]",
            border_style="red",
        )
    )

    explanation = """
    [bold white]Streaming Retry Feature[/bold white]

    The native Ollama provider includes automatic retry logic for streaming requests.
    If a stream is interrupted (network issue, connection reset, server hiccup), it will
    automatically retry up to `max_stream_retries` times (default: 3) with exponential
    backoff (1s → 2s → 4s).

    [bold white]Retry Behavior:[/bold white]
    - ✅ Retries on: Network errors, connection resets, timeouts
    - ❌ No retry on: API errors, invalid requests (fail fast)
    - 🔄 Backoff: Exponential (1s, 2s, 4s, 8s...)

    [bold white]Configuration:[/bold white]
    You can configure retry behavior with the `max_stream_retries` parameter.
    """

    console.print(Panel(explanation, border_style="red"))

    # Example with custom retry configuration
    from allos.providers.ollama import OllamaProvider

    # Create provider with custom retry settings
    provider = OllamaProvider(
        model="llama3.1:latest",
        max_stream_retries=5,  # Allow up to 5 retry attempts (default: 3)
    )

    console.print("\n[dim]Provider configured with max_stream_retries=5[/dim]")
    console.print(
        "[dim]Streaming response (will auto-retry on connection failures)...[/dim]\n"
    )

    messages = [Message(role=MessageRole.USER, content="Count from 1 to 5.")]

    console.print("[red]Response:[/red] ", end="")

    try:
        for chunk in provider.stream_chat(messages):
            if chunk.content:
                console.print(chunk.content, end="")
            if chunk.error:
                console.print(f"\n[red]Error: {chunk.error}[/red]")
                break
    except Exception as e:
        console.print(f"\n[red]Stream failed: {e}[/red]")
    else:
        console.print("\n\n[green]✓ Stream completed successfully![/green]")

    console.print("\n[dim]Note: Under normal conditions, you won't see retries.[/dim]")
    console.print(
        "[dim]The retry mechanism activates automatically on connection failures.[/dim]"
    )


# ============================================================================
# SECTION 6: DECISION GUIDE
# ============================================================================


def show_decision_guide():
    """Show a guide for choosing between native and compat providers."""
    console.print(
        Panel.fit(
            "[bold magenta]When to Use Which Provider?[/]",
            border_style="magenta",
        )
    )

    guide = """
# Native Provider (`ollama`)

**Use when:**
- ✅ You need tool calling support
- ✅ You want full feature parity with other providers
- ✅ You need model verification and validation
- ✅ You want optimal performance and native integration

**Requires:**
- `ollama` Python package installed
- Compatible model with tool calling (e.g., llama3.1)

---

# Compat Provider (`ollama_compat`)

**Use when:**
- ✅ Simple query/response without tools
- ✅ Quick setup without additional dependencies
- ✅ Testing OpenAI-compatible endpoint compatibility
- ✅ Using models that don't support native tool calling

**Advantages:**
- Simpler configuration
- No need for ollama Python library
- Good for basic use cases

---

# Quick Reference

| Feature | Native (`ollama`) | Compat (`ollama_compat`) |
|---------|-------------------|--------------------------|
| Tool Calling | ✅ Full support | ❌ Not available |
| Streaming | ✅ Yes | ✅ Yes |
| Model Verification | ✅ Yes | ⚠️  Limited |
| Setup Complexity | Medium | Low |
| **Recommended For** | **Production use** | **Simple queries** |
"""

    md = Markdown(guide)
    console.print(md)


# ============================================================================
# MAIN RUNNER
# ============================================================================


def main():
    """Run all Ollama usage examples."""
    console.print(
        Panel.fit(
            "[bold white on blue] Ollama Complete Usage Guide [/]",
            border_style="blue",
        )
    )

    # Check if Ollama is running
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    if not ollama_running(ollama_host):
        console.print(
            "[bold red]❌ Ollama is not running![/bold red]\n"
            "Please start Ollama with: [bold]ollama serve[/bold]\n"
            "Then pull required models:\n"
            "  - [bold]ollama pull llama3.1[/bold]\n"
            "  - [bold]ollama pull mistral[/bold]"
        )
        return

    console.print("[bold green]✅ Ollama is running[/bold green]\n")

    examples = [
        ("Basic Usage - Native", example_1_basic_native),
        ("Basic Usage - Compat", example_2_basic_compat),
        ("Provider Comparison", example_3_provider_comparison),
        ("Streaming - Native", example_4_streaming_native),
        ("Streaming - Compat", example_5_streaming_compat),
        ("Tool Calling - Native", example_6_tool_calling_native),
        (
            "Tool Calling - Compat Limitations",
            example_7_tool_calling_compat_limitations,
        ),
        ("Model Warm-Up Detection", example_8_warmup_detection),
        ("Streaming with Retry", example_9_streaming_with_retry),
    ]

    for i, (name, func) in enumerate(examples, 1):
        console.print(f"\n{'=' * 70}")
        console.print(f"[bold]{i}. {name}[/bold]")
        console.print(f"{'=' * 70}\n")

        try:
            func()
        except Exception as e:
            console.print(f"[red]Error in {name}: {e}[/red]")

        console.print()

    # Show decision guide
    console.print(f"\n{'=' * 70}")
    show_decision_guide()

    console.print("\n[bold green]✅ All examples completed![/bold green]")
    console.print("\n[dim]For more examples, see:[/dim]")
    console.print("  - examples/local_models.py (native provider with tools)")
    console.print("  - examples/ollama_compat.py (compat provider)")
    console.print("  - examples/local_vs_cloud.py (detailed comparison)")


if __name__ == "__main__":
    main()
