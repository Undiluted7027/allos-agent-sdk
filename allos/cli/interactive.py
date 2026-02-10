# allos/cli/interactive.py

"""Manages the interactive REPL (Read-Eval-Print Loop) session for the Allos CLI.

This module provides the logic for the interactive chat mode, launched via the
`--interactive` or `-i` flag from the main CLI entry point. It allows users to
have a continuous, stateful conversation with an `Agent` directly from their
terminal, making it ideal for development, debugging, and multi-step tasks.

The primary entry point is the `start_interactive_session` function, which
initializes the agent and handles the entire lifecycle of the REPL session.

Key functionalities of this module include:
 - Displaying a welcome banner and user instructions.
 - Creating a new agent session based on CLI arguments.
 - Loading a previous agent session from a specified file.
 - Intelligently overriding a loaded session's configuration with any new
   command-line arguments provided at launch (e.g., changing the model or
   enabling auto-approve).
 - Running the main input loop to capture and process user prompts.
 - Gracefully handling agent errors and user interruptions (e.g., Ctrl+D, Ctrl+C)
   without crashing the session.
 - Saving the updated conversation context back to a session file upon exit,
   ensuring persistence.
"""

from pathlib import Path
from typing import Optional, Tuple

from rich.console import Console
from rich.panel import Panel

from ..agent import Agent, AgentConfig
from ..providers import ProviderRegistry
from ..tools import ToolRegistry
from ..utils.errors import AllosError
from .logo import LOGO_BANNER
from .utils import (
    display_provider_info,
    display_validation_error,
    validate_model_and_api_key,
)

console = Console()


def start_interactive_session(
    provider: str,
    model: Optional[str],
    base_url: Optional[str],
    api_key: Optional[str],
    max_tokens: Optional[int],
    tool_names: list[str],
    no_tools: bool,
    session_file: Optional[str],
    auto_approve: bool,
    stream: bool = False,
):
    """Starts and manages an interactive REPL session with an agent."""
    _print_welcome_message(stream)

    try:
        # --- Determine the model ---
        validation_result = validate_model_and_api_key(provider, model, api_key)

        if not validation_result.success or not validation_result.model:
            display_validation_error(validation_result, provider, console)
            return

        validated_model = validation_result.model
        display_provider_info(validation_result, provider, console, stream)

        agent = _load_or_create_agent(
            provider,
            validated_model,
            base_url,
            api_key,
            max_tokens,
            tool_names,
            no_tools,
            session_file,
            auto_approve,
        )
        if agent.config.auto_approve and not agent.config.no_tools:
            console.print("[bold yellow]⚠️ Auto-approve is enabled.[/bold yellow]")
        if agent.config.no_tools:
            console.print(
                "[bold blue] Tools are disabled for this session.[/bold blue]"
            )
    except AllosError as e:
        _print_panel(f"Failed to initialize agent: {e}", "Initialization Error", "red")
        return

    _run_repl_loop(agent, stream)

    if session_file:
        _save_session(agent, session_file)

    console.print("\n[bold blue]Exiting interactive session. Goodbye![/]")


# --- Helper functions ---


def _print_welcome_message(stream: bool = False) -> None:
    console.print(LOGO_BANNER, style="bold blue")
    stream_info = (
        "[cyan]streaming enabled[/]" if stream else "[dim]streaming disabled[/]"
    )
    console.print(
        Panel(
            "[bold]Welcome to the Allos Interactive Session![/]\n\n"
            f"Mode: {stream_info}\n\n"
            "Commands:\n"
            "  [cyan]/stream[/]         - Show streaming status\n"
            "  [cyan]/stream on[/]      - Enable streaming mode\n"
            "  [cyan]/stream off[/]     - Disable streaming mode\n"
            "  [cyan]/help[/]           - Show this help\n"
            "  [cyan]/exit[/] or [cyan]/quit[/] - Exit session\n\n"
            "Type your prompts below or use commands starting with [cyan]/[/]",
            title="Interactive Mode",
            border_style="bold blue",
        )
    )


def _load_or_create_agent(
    provider: str,
    model: str,
    base_url: Optional[str],
    api_key: Optional[str],
    max_tokens: Optional[int],
    tool_names: list[str],
    no_tools: bool,
    session_file: Optional[str],
    auto_approve: bool,
) -> Agent:
    """Handles session loading or new agent creation."""
    if session_file and Path(session_file).exists():
        console.print(f"🔄 Loading session from '{session_file}'...")
        agent = Agent.load_session(session_file)
        # Override loaded config with any new CLI flags
        _override_agent_config(
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
        tool_names=list(tool_names) or ToolRegistry.list_tools(),
        auto_approve=auto_approve,
    )
    return Agent(config)


def _override_agent_config(
    agent: Agent,
    provider: str,
    model: Optional[str],
    base_url: Optional[str],
    api_key: Optional[str],
    max_tokens: Optional[int],
    tool_names: list[str],
    no_tools: bool,
    auto_approve: bool,
) -> None:
    """Apply CLI overrides to a loaded agent config."""
    # Detect if provider changed
    if agent.config.provider_name != provider:
        # Reset provider-specific settings to defaults to avoid pollution
        agent.config.base_url = None
        agent.config.no_tools = False
        agent.config.api_key = None
    agent.config.provider_name = provider
    agent.config.model = model or agent.config.model
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
        # Re-initialize tools
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


def _run_repl_loop(agent: Agent, stream: bool = False) -> None:
    """Main REPL loop for user input."""
    streaming_enabled = stream  # Mutable state for toggling
    end_flag = False
    while not end_flag:
        try:
            prompt = console.input("[bold cyan]>>> [/]")

            # Skip empty input
            if not prompt.strip():
                continue

            # Handle REPL commands
            if prompt.startswith("/"):
                streaming_state, end_check = _handle_repl_command(
                    prompt, streaming_enabled, end_flag
                )
                if streaming_state is not None:
                    streaming_enabled = streaming_state
                if end_check:
                    end_flag = end_check
                continue

            # Execute agent prompt
            if streaming_enabled:
                _execute_streaming(agent, prompt)
            else:
                _execute_sync(agent, prompt)

        except (KeyboardInterrupt, EOFError):
            break
        except AllosError as e:
            _print_panel(f"An agent error occurred: {e}", "Agent Error", "red")
        except Exception as e:
            _print_panel(f"An unexpected error occurred: {e}", "System Error", "red")


def _handle_repl_command(
    command: str, current_stream_state: bool, end_flag: bool
) -> Tuple[Optional[bool], bool]:
    """Handle REPL commands starting with /.

    Returns:
        New streaming state if changes, None if no state change
    """
    parts = command.lower().split()
    cmd = parts[0]

    new_streaming_state = None

    if cmd == "/help":
        console.print(
            Panel(
                "[bold]Available commands:[/]\n\n"
                "   [cyan]/stream[/]        - Show streaming status\n"
                "   [cyan]/stream on[/]     - Enable streaming mode\n"
                "   [cyan]/stream off[/]    - Disable streaming mode\n"
                "   [cyan]/help[/]          - Show this help\n"
                "   [cyan]/exit[/] or [cyan]/quit[/] - Exit session",
                title="Help",
                border_style="blue",
            )
        )

    elif cmd == "/stream":
        if len(parts) == 1:
            # Show current status
            status = "[green]enabled[/]" if current_stream_state else "[dim]disabled[/]"
            console.print(f"Streaming is currently {status}")
            console.print(
                "[dim]Use [cyan]/stream on[/] or [cyan]/stream off[/] to toggle[/]"
            )

        elif len(parts) == 2:
            subcommand = parts[1]
            if subcommand == "on":
                if not current_stream_state:
                    console.print("[green]✓[/] Streaming enabled")
                    new_streaming_state = True
            elif subcommand == "off":
                if current_stream_state:
                    console.print("[yellow]✓[/] Streaming disabled")
                    new_streaming_state = False
            else:
                console.print(f"[red]Unknown command:[/] {cmd} {subcommand}")
                console.print("[dim]Type [cyan]/help[/] for available commands[/]")
                new_streaming_state = None
    elif cmd == "/exit" or cmd == "/quit":
        end_flag = True

    return (new_streaming_state, end_flag)


def _execute_sync(agent: Agent, prompt: str) -> None:
    """Execute agent prompt synchronously."""
    agent.run(prompt)


def _execute_streaming(agent: Agent, prompt: str) -> None:
    """Execute agent prompt with streaming output."""
    for chunk in agent.stream_run(prompt):
        if chunk.content:
            console.print(chunk.content, end="", style="blue")
        elif chunk.tool_call_start:
            console.print(
                f"\n[yellow]└─ Calling Tool: {chunk.tool_call_start['name']}(...)[/]"
            )
        elif chunk.error:
            console.print(f"\n[bold red]Stream Error: {chunk.error}[/]")

    console.print()


def _save_session(agent: Agent, session_file: str) -> None:
    """Save session to file on exit."""
    try:
        agent.save_session(session_file)
    except AllosError as e:
        _print_panel(f"Failed to save session on exit: {e}", "Save Error", "red")


def _print_panel(message: str, title: str, color: str) -> None:
    """Utility for consistent panel printing."""
    console.print(
        Panel(
            f"[bold {color}]{message}[/]",
            title=title,
            border_style=color,
        )
    )
