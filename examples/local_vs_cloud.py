# examples/local_vs_cloud.py

"""The Great Debate: Local vs Cloud.

Demonstrates the trade-offs between local models (Ollama) and cloud providers.
This example provides a side-by-side comparison covering:
1. Performance: Response time comparison
2. Cost: Token usage and API costs
3. Privacy: Where your data goes
4. Capability: Tool calling and model quality
Requirements:
- Ollama running locally with llama3.1: `ollama pull llama3.1`
- OpenAI API key: OPENAI_API_KEY environment variable
- Cohere API key (optional): COHERE_API_KEY environment variable
Usage: python examples/local_vs_cloud.py
"""

import os
from dataclasses import dataclass
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from allos import Agent, AgentConfig
from allos.providers import ollama_running

console = Console()


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    provider: str
    model: str
    response_time: float
    input_tokens: int
    output_tokens: int
    tps: float
    response: str
    error: Optional[str] = None
    warm_up: bool = False  # Ollama-specific: model warm-up detected


def check_provider_availability() -> dict:
    """Check which providers are available for the comparison."""
    available = {}

    # Check Ollama
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    available["ollama"] = ollama_running(ollama_host)

    # Check OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    available["openai"] = bool(openai_key)

    # Check Anthropic (optional bonus)
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    available["anthropic"] = bool(anthropic_key)

    # Check Cohere (optional bonus)
    cohere_key = os.getenv("COHERE_API_KEY")
    available["cohere"] = bool(cohere_key)

    return available


def run_benchmark(
    provider: str, model: str, prompt: str, use_tools: bool = False
) -> BenchmarkResult:
    """Run a single benchmark against a provider."""
    tool_names = ["read_file", "list_directory"] if use_tools else []

    config = AgentConfig(
        provider_name=provider,
        model=model,
        no_tools=not use_tools,
        tool_names=tool_names,
    )

    try:
        agent = Agent(config)

        response = agent.run(prompt)

        # Extract token usage from the last response metadata
        input_tokens = 0
        output_tokens = 0
        duration = 0.0
        tps = 0.0
        warm_up_detected = False

        if agent.context.messages:
            meta = agent.last_run_metadata
            if not meta:
                raise ValueError("No metadata returned")
            duration = meta.latency.total_duration_ms / 1000.0
            output_tokens = meta.usage.output_tokens
            input_tokens = meta.usage.input_tokens
            tps = output_tokens / duration if duration > 0 else 0

            # Check for Ollama warm-up
            if provider == "ollama" and meta.provider_specific.ollama:
                warm_up_detected = meta.provider_specific.ollama.warm_up
                if warm_up_detected:
                    console.print(
                        f"[yellow]  ⚠️  Note: Model warm-up detected for {model} "
                        f"({meta.provider_specific.ollama.warm_up_duration_seconds:.1f}s)[/yellow]"
                    )

        return BenchmarkResult(
            provider=provider,
            model=model,
            response_time=duration,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            response=response,
            tps=tps,
            warm_up=warm_up_detected,
        )

    except Exception as e:
        return BenchmarkResult(
            provider=provider,
            model=model,
            response_time=0,
            input_tokens=0,
            output_tokens=0,
            response="",
            error=str(e),
            tps=0,
            warm_up=False,
        )


def estimate_cost(provider: str, input_tokens: int, output_tokens: int) -> Optional[float]:
    """Estimate cost using static reference pricing (illustrative only)."""
    # Approximate pricing per 1M tokens (reference values, not billing truth).
    pricing = {
        "openai": {"input": 2.50, "output": 10.00},  # GPT-4o
        "anthropic": {"input": 3.00, "output": 15.00},  # Claude 3.5 Sonnet
        "groq": {"input": 0.05, "output": 0.08},  # Llama 3.3 70B
        "ollama": {"input": 0.00, "output": 0.00},  # Free (local)
        "cohere": {"input": 0.0375, "output": 0.15} # Command R7B
    }

    if provider not in pricing:
        return None

    rates = pricing[provider]
    input_cost = (input_tokens / 1_000_000) * rates["input"]
    output_cost = (output_tokens / 1_000_000) * rates["output"]
    return input_cost + output_cost


def display_comparison_table(results: list[BenchmarkResult]):
    """Display a comparison table of benchmark results."""
    table = Table(title="Local vs Cloud Comparison")

    table.add_column("Provider", style="cyan", no_wrap=True)
    table.add_column("Model", style="blue")
    table.add_column("Time (s)", justify="right", style="green")
    table.add_column("Tokens (In/Out)", justify="right")
    table.add_column("Est. Cost", justify="right", style="yellow")
    table.add_column("Data Privacy", style="magenta")
    table.add_column("Speed (Tokens per second)")
    table.add_column("Warm-Up", style="yellow")  # New column for Ollama

    for result in results:
        if result.error:
            table.add_row(
                result.provider,
                result.model,
                "[red]Error[/]",
                "-",
                "-",
                "-",
                "-",
                "-",
            )
        else:
            cost = estimate_cost(result.provider, result.input_tokens, result.output_tokens)
            if cost is None:
                cost_str = "[dim]N/A[/]"
            else:
                cost_str = f"${cost:.6f}" if cost > 0 else "[green]Free[/]"
            privacy = (
                "[green]Local[/]" if result.provider == "ollama" else "[yellow]Cloud[/]"
            )
            tokens_str = f"{result.input_tokens}/{result.output_tokens}"

            # Warm-up status (Ollama-specific)
            if result.provider == "ollama":
                warm_up_str = "[yellow]Yes[/]" if result.warm_up else "[green]No[/]"
            else:
                warm_up_str = "[dim]N/A[/]"

            table.add_row(
                result.provider,
                result.model,
                f"{result.response_time:.2f}",
                tokens_str,
                cost_str,
                privacy,
                f"{result.tps:.2f}",
                warm_up_str,
            )

    console.print(table)


def display_privacy_comparison():
    """Display a privacy comparison panel."""
    privacy_info = """
[bold cyan]Local Models (Ollama)[/]
  - Data never leaves your machine
  - No API calls to external servers
  - Full control over model and data
  - Ideal for: Sensitive data, PII, confidential documents

[bold yellow]Cloud Providers (OpenAI, Anthropic, Cohere)[/]
  - Data sent to provider's servers
  - Subject to provider's data policies
  - Higher quality models available
  - Ideal for: General tasks, when quality matters most
"""
    console.print(
        Panel(privacy_info, title="Privacy Considerations", border_style="blue")
    )


def display_use_case_guide():
    """Display a guide for when to use each option."""
    guide = """
[bold]When to use LOCAL models:[/]
  - Processing sensitive/confidential data
  - Working with PII (Personal Identifiable Information)
  - Air-gapped or offline environments
  - Cost-conscious high-volume usage
  - When latency consistency matters

[bold]When to use CLOUD models:[/]
  - Maximum quality/capability needed
  - Complex reasoning or coding tasks
  - When you need cutting-edge models
  - Low-volume usage where cost is minimal
  - Production systems with SLA requirements
"""
    console.print(Panel(guide, title="Use Case Guide", border_style="green"))


def print_provider_availability(available: dict) -> None:
    """Print provider readiness and cost-estimate note."""
    console.print("\n[bold]Provider Availability:[/]")
    for provider, is_available in available.items():
        status = "[green]Ready[/]" if is_available else "[red]Not Available[/]"
        console.print(f"  {provider}: {status}")
    console.print(
        "[dim]Cost values in this demo are reference estimates and may differ from "
        "actual provider billing.[/dim]"
    )


def print_missing_provider_hints(available: dict) -> None:
    """Print setup hints for unavailable providers."""
    if not available["ollama"]:
        console.print(
            "\n[yellow]Ollama not running. Start with: [bold]ollama serve[/][/]"
        )
    if not available["openai"]:
        console.print(
            "\n[yellow]OpenAI not configured. Set OPENAI_API_KEY environment variable.[/]"
        )
    if not available["cohere"]:
        console.print(
            "\n[yellow]Cohere not configured. Set COHERE_API_KEY environment variable.[/]"
        )


def run_available_benchmarks(available: dict, test_prompt: str) -> list[BenchmarkResult]:
    """Run all benchmarks for currently available providers."""
    results = []

    if available["ollama"]:
        with console.status("[cyan]Testing Ollama (local)...[/]", spinner="dots"):
            result = run_benchmark("ollama", "llama3.1:latest", test_prompt)
            results.append(result)

    if available["openai"]:
        with console.status("[yellow]Testing OpenAI (cloud)...[/]", spinner="dots"):
            result = run_benchmark("openai", "gpt-4o-mini", test_prompt)
            results.append(result)

    if available["anthropic"]:
        with console.status("[magenta]Testing Anthropic (cloud)...[/]", spinner="dots"):
            result = run_benchmark(
                "anthropic", "claude-3-5-haiku-20241022", test_prompt
            )
            results.append(result)

    if available["cohere"]:
        with console.status("[blue]Testing Cohere (cloud)...[/]", spinner="dots"):
            result = run_benchmark("cohere", "command-r7b-12-2024", test_prompt)
            results.append(result)

    return results


def main():
    """Run the local vs cloud comparison benchmark."""
    console.print(
        Panel.fit(
            "[bold white]Local vs Cloud: A Comparison[/]",
            style="bold blue",
        )
    )

    # Check availability
    available = check_provider_availability()
    print_provider_availability(available)
    print_missing_provider_hints(available)

    if not any(available.values()):
        console.print("\n[red]No providers available for comparison.[/]")
        return

    # Run benchmarks
    console.print("\n[bold]Running benchmarks...[/]")

    # Simple reasoning task (no tools)
    test_prompt = (
        "Explain in 2-3 sentences why the sky is blue. Be concise and scientific."
    )
    console.print(f'\n[dim]Test prompt: "{test_prompt}"[/dim]')

    results = run_available_benchmarks(available, test_prompt)

    # Display results
    console.print()
    display_comparison_table(results)

    # Show responses
    console.print("\n[bold]Responses:[/]")
    for result in results:
        if not result.error:
            console.print(
                Panel(
                    result.response,
                    title=f"{result.provider} ({result.model})",
                    border_style="dim",
                )
            )
        else:
            console.print(f"[red]{result.provider}: {result.error}[/]")

    # Display educational content
    console.print()
    display_privacy_comparison()
    display_use_case_guide()

    # Summary
    console.print(
        Panel(
            "[bold]Key Takeaway:[/]\n"
            "Local models (Ollama) offer privacy and zero cost at the expense of "
            "some capability. Cloud models offer maximum quality but send data "
            "to external servers. The Allos SDK lets you switch seamlessly between "
            "them based on your needs.",
            title="Summary",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
