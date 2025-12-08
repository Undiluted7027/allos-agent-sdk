# allos/cli/main.py
r"""Defines the main command-line interface (CLI) entry point for the Allos SDK.

This module serves as the "front door" for the entire application when used from
a terminal. It uses the `click` library to create a rich, self-documenting CLI
with a full suite of options, arguments, and flags.

Its primary responsibility is to parse all user-provided command-line arguments,
translate them into a valid `AgentConfig`, and then dispatch the execution to the
appropriate handler—either a one-shot run, a streaming run, or an interactive
REPL session.

Key Responsibilities:
 - Defining all CLI options (`--provider`, `--model`, `--tool`, etc.) and arguments.
 - Handling informational flags that provide data and exit immediately, such as
   `--list-providers`, `--active-providers`, and `--list-tools`.
 - Parsing user input to configure and initialize the `Agent`.
 - Dispatching execution to the correct handler:
    - `interactive.start_interactive_session` for the REPL mode (`-i`).
    - `run_agent` for standard, synchronous execution.
    - `run_agent_stream` for streaming responses (`--stream`).
 - Implementing user-experience safety nets, such as confirming single-word
   prompts to prevent accidental execution of typos.
 - Setting up the global logging level based on the `--verbose` flag.

Usage Examples:
    # Run a simple, one-shot prompt with default settings (OpenAI).
    $ allos "Create a Python file named app.py that prints 'Hello, World!'."

    # Start an interactive chat session.
    $ allos -i

    # Use a different provider and model, specifying a tool and enabling auto-approve.
    $ allos -p anthropic -m claude-3-haiku-20240307 --tool write_file --auto-approve \\
      "Write a short story about an AI and save it to story.txt"

    # Run in streaming mode to see the response as it's generated.
    $ allos --stream "Tell me a long and detailed story about a space explorer."

    # List all available tools and their descriptions.
    $ allos --list-tools
"""

import os
from typing import List, Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..agent import Agent, AgentConfig
from ..providers import ProviderRegistry
from ..tools import ToolRegistry
from ..utils.errors import AllosError
from ..utils.logging import setup_logging
from .interactive import start_interactive_session
from .logo import LOGO_BANNER

# --- Helper to load API keys from a .env file if it exists ---
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # python-dotenv is not a core dependency

console = Console()


class RichHelpCommand(click.Command):
    """A custom `click.Command` subclass to display a logo banner with the help message.

    This class overrides the default `format_help` method to first print the
    Allos SDK's ASCII art logo banner to the console before proceeding with the
    standard help text provided by Click. This enhances the branding and user
    experience of the CLI's help command (`--help`, `-h`).
    """

    def format_help(self, ctx, formatter):
        """Formats the help message, prepending the SDK's logo banner.

        This is an override of the base `click.Command.format_help` method.

        Args:
            ctx: The click Context object.
            formatter: The click HelpFormatter object.
        """
        console.print(LOGO_BANNER, style="bold cyan")
        super().format_help(ctx, formatter)


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
    "base-url",
    "api-key",
    "max-tokens",
    "no-tools",
}


def print_providers(ctx, param, value):
    """A callback function for the `--list-providers` CLI flag.

    This function is triggered by Click when the `--list-providers` flag is used.
    It fetches all registered provider names from the `ProviderRegistry`, formats
    them into a `rich.panel.Panel`, prints the result to the console, and then
    terminates the program.

    The check `if not value or ctx.resilient_parsing:` prevents the callback from
    running during shell completion activities.

    Args:
        ctx: The click Context object.
        param: The click Parameter object that triggered the callback.
        value: The value of the parameter; for a flag, this is `True` if present.
    """
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


def print_active_providers(ctx, param, value):
    """A callback for the `--active-providers` flag to show provider readiness.

    This function is triggered by Click when the `--active-providers` flag is used.
    It prints a `rich.table.Table` detailing each available provider and its
    current configuration status. The status is determined by checking if the
    required API key environment variable for each provider is set.

    Possible statuses:
     - "Ready": The required environment variable is set.
     - "Missing Key": The required environment variable is not set.
     - "Manual Config Required": The provider does not rely on a standard env var.

    The program terminates after printing the table.

    Args:
        ctx: The click Context object.
        param: The click Parameter object that triggered the callback.
        value: The value of the parameter; for a flag, this is `True` if present.
    """
    if not value or ctx.resilient_parsing:
        return

    providers = ProviderRegistry.list_providers()
    table = Table(title="Active Providers Configuration")
    table.add_column("Provider", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Required Env Var", style="dim")

    for p in providers:
        env_var = ProviderRegistry.get_env_var_name(p)

        # Special case for generic providers or those without env vars
        if not env_var:
            status = "[yellow]Manual Config Required[/]"
            var_display = "N/A"
        elif env_var in os.environ:
            status = "[green]Ready[/]"
            var_display = f"{env_var} (Set)"
        else:
            status = "[red]Missing Key[/]"
            var_display = f"{env_var} (Not Set)"

        table.add_row(p, status, var_display)

    console.print(table)
    ctx.exit()


def print_tools(ctx, param, value):
    """A callback function for the `--list-tools` CLI flag.

    This function is triggered by Click when the `--list-tools` flag is used.
    It fetches all registered tools from the `ToolRegistry`, formats their
    name, permission level, and description into a `rich.panel.Panel`, prints
    the result to the console, and then terminates the program.

    The check `if not value or ctx.resilient_parsing:` prevents the callback from
    running during shell completion activities.

    Args:
        ctx: The click Context object.
        param: The click Parameter object that triggered the callback. (unused)
        value: The value of the parameter; for a flag, this is `True` if present.
    """
    if not value or ctx.resilient_parsing:
        return
    tools = ToolRegistry.get_all_tools()
    output = "\n".join(
        f"- [bold]{tool.name}[/] ({tool.permission.value}): {tool.description}"
        for tool in tools
    )
    console.print(Panel(output, title="Available Tools", border_style="magenta"))
    ctx.exit()


@click.command(
    name="allos",
    cls=RichHelpCommand,
    context_settings={"help_option_names": ["-h", "--help"]},
)
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
    "--active-providers",
    is_flag=True,
    callback=print_active_providers,
    expose_value=False,
    is_eager=True,
    help="List providers that are configured and ready to use.",
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
    "--base-url",
    help="The base URL for the API (only used for 'chat_completions' or compatible providers).",
)
@click.option(
    "--api-key",
    help="The API key to use. Overrides environment variables.",
)
@click.option(
    "--max-tokens",
    type=int,
    help="Maximum number of tokens to generate.",
)
@click.option(
    "--tool",
    "tool_names",
    multiple=True,
    help="Specify a tool the agent can use. Can be used multiple times.",
    type=click.Choice(ToolRegistry.list_tools()),
)
@click.option(
    "--no-tools",
    is_flag=True,
    help="Disable all tools for this session.",
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
@click.option("--stream", is_flag=True, help="Enable streaming of the response.")
@click.argument("prompt", nargs=-1)  # Capture all remaining arguments as the prompt
def main(
    verbose: bool,
    interactive: bool,
    provider: str,
    model: Optional[str],
    base_url: Optional[str],
    api_key: Optional[str],
    max_tokens: Optional[int],
    tool_names: List[str],
    no_tools: bool,
    session_file: Optional[str],
    auto_approve: bool,
    stream: bool,
    prompt: tuple,
):
    """🚀 Run an agent to execute a given PROMPT.

    Example: allos "Create a FastAPI hello world app"
    """
    setup_logging(level="DEBUG" if verbose else "INFO")

    # --- CLI Logic and Safety Nets ---

    # If interactive mode is requested, start it and exit.
    if interactive:
        start_interactive_session(
            provider,
            model,
            base_url,
            api_key,
            max_tokens,
            list(tool_names),
            no_tools,
            session_file,
            auto_approve,
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
    if stream:
        run_agent_stream(
            full_prompt,
            provider,
            model,
            base_url,
            api_key,
            max_tokens,
            list(tool_names),
            no_tools,
            session_file,
            auto_approve,
        )
    else:
        run_agent(
            full_prompt,
            provider,
            model,
            base_url,
            api_key,
            max_tokens,
            list(tool_names),
            no_tools,
            session_file,
            auto_approve,
        )


# --- Helper Functions ---
def _determine_model(provider: str, model: Optional[str]) -> str:
    """Selects a default model if none is provided."""
    if model is not None:
        return model

    default_model = "gpt-4o" if provider == "openai" else "claude-3-haiku-20240307"
    console.print(
        f"[dim]Model not specified, defaulting to '{default_model}' for provider '{provider}'.[/dim]"
    )
    return default_model


def _validate_api_key(provider: str, api_key: Optional[str]) -> bool:
    """Checks if an API key is available.

    Returns True if valid, False if missing (and prints error).
    """
    if api_key:
        return True

    required_env_var = ProviderRegistry.get_env_var_name(provider)

    if required_env_var and required_env_var not in os.environ:
        console.print(
            f"[bold red]Error:[/] API key not found. "
            f"Please set the [bold]{required_env_var}[/] environment variable "
            f"or use the [bold]--api-key[/] option."
        )
        return False
    return True


def _initialize_agent(
    provider: str,
    model: str,
    base_url: Optional[str],
    api_key: Optional[str],
    max_tokens: Optional[int],
    tool_names: List[str],
    no_tools: bool,
    session_file: Optional[str],
    auto_approve: bool,
) -> Agent:
    """Loads an existing agent or creates a new one."""
    if session_file and os.path.exists(session_file):
        console.print(f"🔄 Loading session from '{session_file}'...")
        agent = Agent.load_session(session_file)
        _update_agent_config(
            agent,
            provider,
            model,
            base_url,
            api_key,
            max_tokens,
            tool_names,
            no_tools,
            auto_approve,
        )
        return agent

    config = AgentConfig(
        provider_name=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
        max_tokens=max_tokens,
        no_tools=no_tools,
        tool_names=list(tool_names)
        or ToolRegistry.list_tools(),  # Default to all tools if none specified
        auto_approve=auto_approve,
    )
    return Agent(config)


def _update_agent_config(
    agent: Agent,
    provider: str,
    model: str,
    base_url: Optional[str],
    api_key: Optional[str],
    max_tokens: Optional[int],
    tool_names: List[str],
    no_tools: bool,
    auto_approve: bool,
):
    """Updates an existing agent's config with CLI overrides."""
    # Detect if provider changed
    if agent.config.provider_name != provider:
        # Reset provider-specific settings to defaults to avoid pollution
        agent.config.base_url = None
        agent.config.no_tools = False
        agent.config.api_key = None
    agent.config.provider_name = provider
    agent.config.model = model
    agent.config.auto_approve = auto_approve
    if base_url:
        agent.config.base_url = base_url
    if api_key:
        agent.config.api_key = api_key
    if max_tokens:
        agent.config.max_tokens = max_tokens
    if no_tools:
        agent.config.no_tools = True
        agent.tools = []
    elif tool_names:
        agent.config.tool_names = list(tool_names)
        # Re-initialize tools if names changed
        agent.tools = [ToolRegistry.get_tool(name) for name in tool_names]

    # The agent.provider instance holds the client. We must recreate it
    # so it picks up the new provider_name, model, base_url, and api_key.
    provider_kwargs = {}
    if agent.config.base_url:
        provider_kwargs["base_url"] = agent.config.base_url
    if agent.config.api_key:
        provider_kwargs["api_key"] = agent.config.api_key

    agent.provider = ProviderRegistry.get_provider(
        agent.config.provider_name, model=agent.config.model, **provider_kwargs
    )

    # 5. Re-initialize Tools Instance
    if agent.config.no_tools:
        agent.tools = []
    else:
        # Reload tools based on potentially updated tool_names
        agent.tools = [ToolRegistry.get_tool(name) for name in agent.config.tool_names]


# --- Helper function to contain the 'run' logic ---
def run_agent(
    prompt: str,
    provider: str,
    model: Optional[str],
    base_url: Optional[str],
    api_key: Optional[str],
    max_tokens: Optional[int],
    tool_names: List[str],
    no_tools: bool,
    session_file: Optional[str],
    auto_approve: bool,
):
    """The core logic for running the agent."""
    # --- Determine the model ---
    model = _determine_model(provider, model)

    # --- Validate API Key ---
    if not _validate_api_key(provider, api_key):
        return

    console.print(f"[dim] Using {provider} with model {model}.")
    try:
        # --- Initialize Agent ---
        agent = _initialize_agent(
            provider,
            model,
            base_url,
            api_key,
            max_tokens,
            tool_names,
            no_tools,
            session_file,
            auto_approve,
        )

        # --- Handle Auto-Approve ---
        if agent.config.auto_approve and not agent.config.no_tools:
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


def run_agent_stream(
    prompt: str,
    provider: str,
    model: Optional[str],
    base_url: Optional[str],
    api_key: Optional[str],
    max_tokens: Optional[int],
    tool_names: List[str],
    no_tools: bool,
    session_file: Optional[str],
    auto_approve: bool,
):
    """Initializes and runs the agent in streaming mode, printing chunks to the console."""
    model = _determine_model(provider, model)
    if not _validate_api_key(provider, api_key):
        return

    console.print(f"[dim] Using {provider} with model {model} (streaming).[/dim]")
    try:
        agent = _initialize_agent(
            provider,
            model,
            base_url,
            api_key,
            max_tokens,
            tool_names,
            no_tools,
            session_file,
            auto_approve,
        )
        if agent.config.auto_approve and not agent.config.no_tools:
            console.print(
                "[bold yellow]⚠️ Auto-approve is enabled. All tool executions will be approved automatically.[/bold yellow]"
            )

        # --- Render the stream to the console ---
        console.print(Panel("Streaming Response", border_style="green"), end="")
        # console.print("\n[bold assistant]Agent:[/] ", end="")

        for chunk in agent.stream_run(prompt):
            if chunk.content:
                console.print(chunk.content, end="", style="blue")
            elif chunk.tool_call_start:
                console.print(
                    f"\n[yellow]└─ Calling Tool: {chunk.tool_call_start['name']}(...)[/]"
                )
            elif chunk.error:
                console.print(f"\n[bold red]Stream Error: {chunk.error}[/]")

        console.print()  # Final newline

        # --- Save Session if specified ---
        if session_file:
            agent.save_session(session_file)

    except AllosError as e:
        console.print(f"\n[bold red]An error occurred:[/] {e}")
    except Exception as e:
        console.print(f"\n[bold red]An unexpected error occurred:[/] {e}")
