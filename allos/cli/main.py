# allos/cli/main.py

import os
from typing import List, Optional

import click
from rich.console import Console
from rich.panel import Panel

from ..agent import Agent, AgentConfig
from ..providers import ProviderRegistry
from ..tools import ToolRegistry
from ..utils.errors import AllosError
from ..utils.logging import setup_logging
from .interactive import start_interactive_session

# --- Helper to load API keys from a .env file if it exists ---
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # python-dotenv is not a core dependency

console = Console()

# --- Set of known flag names to prevent misuse as prompts ---
KNOWN_FLAG_WORDS = {
    "verbose",
    "interactive",
    "list-providers",
    "list-tools",
    "provider",
    "model",
    "tool",
    "session",
    "auto-approve",
    "help",
}


def print_providers(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return
    providers = ProviderRegistry.list_providers()
    console.print(
        Panel(
            "\n".join(f"- {p}" for p in providers),
            title="Available Providers",
            border_style="blue",
        )
    )
    ctx.exit()


def print_tools(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return
    tools = ToolRegistry.get_all_tools()
    output = "\n".join(
        f"- [bold]{tool.name}[/] ({tool.permission.value}): {tool.description}"
        for tool in tools
    )
    console.print(Panel(output, title="Available Tools", border_style="magenta"))
    ctx.exit()


@click.command(name="allos", context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging.")
@click.option(
    "--interactive", "-i", is_flag=True, help="Start an interactive chat session."
)
@click.option(
    "--list-providers",
    is_flag=True,
    callback=print_providers,
    expose_value=False,
    is_eager=True,
    help="List available providers and exit.",
)
@click.option(
    "--list-tools",
    is_flag=True,
    callback=print_tools,
    expose_value=False,
    is_eager=True,
    help="List available tools and exit.",
)
@click.option(
    "--provider",
    "-p",
    default="openai",
    type=click.Choice(ProviderRegistry.list_providers()),
    help="The LLM provider to use (e.g., 'openai', 'anthropic').",
)
@click.option(
    "--model",
    "-m",
    default=None,
    help="The specific model to use. If not provided, a default will be chosen.",
)
@click.option(
    "--tool",
    "tool_names",
    multiple=True,
    help="Specify a tool the agent can use. Can be used multiple times.",
    type=click.Choice(ToolRegistry.list_tools()),
)
@click.option(
    "--session",
    "-s",
    "session_file",
    type=click.Path(dir_okay=False, writable=True),
    help="Load an agent session from a file and save it back after completion.",
)
@click.option(
    "--auto-approve",
    is_flag=True,
    help="Automatically approve all tool execution requests. Use with caution.",
)
@click.argument("prompt", nargs=-1)  # Capture all remaining arguments as the prompt
def main(
    verbose: bool,
    interactive: bool,
    provider: str,
    model: Optional[str],
    tool_names: List[str],
    session_file: Optional[str],
    auto_approve: bool,
    prompt: tuple,
):
    """
    🚀 Run an agent to execute a given PROMPT.

    Example: allos "Create a FastAPI hello world app"
    """
    setup_logging(level="DEBUG" if verbose else "INFO")

    # --- CLI Logic and Safety Nets ---

    # If interactive mode is requested, start it and exit.
    if interactive:
        start_interactive_session(
            provider, model, list(tool_names), session_file, auto_approve
        )
        return

    # If no prompt was given, show help and exit.
    if not prompt:
        click.echo(click.get_current_context().get_help())
        return

    # Safety Net 1: Check if the user mistyped a flag as a prompt.
    if prompt[0] in KNOWN_FLAG_WORDS:
        console.print(
            f"[bold red]Error:[/] You used '{prompt[0]}' as a prompt, but it looks like a command-line option."
        )
        console.print(f"Did you mean to use [bold]'--{prompt[0]}'[/bold]?")
        return

    # Safety Net 2: Confirm single-word prompts that don't contain spaces.
    # This prevents accidental execution of typos like 'hbdhbdvh'.
    # A single argument with spaces (e.g., "hello world") is a clear intent and skips this check.
    if len(prompt) == 1 and " " not in prompt[0]:
        if not click.confirm(
            f"You provided a single-word prompt: '{prompt[0]}'.\nDo you want to proceed?"
        ):
            console.print("Aborted.")
            return

    # If all checks pass, proceed to run the agent.
    full_prompt = " ".join(prompt)
    run_agent(
        full_prompt, provider, model, list(tool_names), session_file, auto_approve
    )


# --- Helper function to contain the 'run' logic ---
def run_agent(
    prompt: str,
    provider: str,
    model: Optional[str],
    tool_names: List[str],
    session_file: Optional[str],
    auto_approve: bool,
):
    """The core logic for running the agent."""

    # --- Determine the model ---
    if model is None:
        model = "gpt-4o" if provider == "openai" else "claude-3-haiku-20240307"
        console.print(
            f"[dim]Model not specified, defaulting to '{model}' for provider '{provider}'.[/dim]"
        )

    # --- Check for API Keys ---
    key_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }
    if key_map.get(provider) not in os.environ:
        console.print(
            f"[bold red]Error:[/] API key '{key_map[provider]}' not found in environment variables."
        )
        return

    try:
        if session_file and os.path.exists(session_file):
            console.print(f"🔄 Loading session from '{session_file}'...")
            agent = Agent.load_session(session_file)
            # Override loaded config with any new CLI flags
            agent.config.provider_name = provider
            agent.config.model = model
            agent.config.auto_approve = auto_approve
            if tool_names:
                agent.config.tool_names = list(tool_names)
        else:
            config = AgentConfig(
                provider_name=provider,
                model=model,
                tool_names=list(tool_names)
                or ToolRegistry.list_tools(),  # Default to all tools if none specified
                auto_approve=auto_approve,
            )
            agent = Agent(config)

        # --- Handle Auto-Approve ---
        if agent.config.auto_approve:
            console.print(
                "[bold yellow]⚠️ Auto-approve is enabled. All tool executions will be approved automatically.[/bold yellow]"
            )

        # --- Run the Agent ---
        agent.run(prompt)

        # --- Save Session if specified ---
        if session_file:
            agent.save_session(session_file)

    except AllosError as e:
        console.print(
            Panel(
                f"[bold red]An error occurred:[/] {e}",
                title="Agent Error",
                border_style="red",
            )
        )
    except Exception as e:
        console.print(
            Panel(
                f"[bold red]An unexpected error occurred:[/] {e}",
                title="System Error",
                border_style="red",
            )
        )
