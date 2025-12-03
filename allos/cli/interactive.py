# allos/cli/interactive.py

from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel

from ..agent import Agent, AgentConfig
from ..providers import ProviderRegistry
from ..tools import ToolRegistry
from ..utils.errors import AllosError
from .logo import LOGO_BANNER

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
):
    """Starts and manages an interactive REPL session with an agent."""
    _print_welcome_message()

    try:
        agent = _load_or_create_agent(
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
            console.print("[bold yellow]⚠️ Auto-approve is enabled.[/bold yellow]")
        if agent.config.no_tools:
            console.print(
                "[bold blue] Tools are disabled for this session.[/bold blue]"
            )
    except AllosError as e:
        _print_panel(f"Failed to initialize agent: {e}", "Initialization Error", "red")
        return

    _run_repl_loop(agent)

    if session_file:
        _save_session(agent, session_file)

    console.print("\n[bold blue]Exiting interactive session. Goodbye![/]")


# --- Helper functions ---


def _print_welcome_message() -> None:
    console.print(LOGO_BANNER, style="bold blue")
    console.print(
        Panel(
            "[bold]Welcome to the Allos Interactive Session![/]\n\n"
            "Type your prompts below. To exit, type `exit`, `quit`, or press Ctrl+D.",
            title="Interactive Mode",
            border_style="bold blue",
        )
    )


def _load_or_create_agent(
    provider: str,
    model: Optional[str],
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

    model = model or ("gpt-4o" if provider == "openai" else "claude-3-haiku-20240307")
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


def _run_repl_loop(agent: Agent) -> None:
    """Main REPL loop for user input."""
    while True:
        try:
            prompt = console.input("[bold cyan]>>> [/]")

            if prompt.lower() in {"exit", "quit"}:
                break
            if not prompt.strip():
                continue

            agent.run(prompt)

        except (KeyboardInterrupt, EOFError):
            break
        except AllosError as e:
            _print_panel(f"An agent error occurred: {e}", "Agent Error", "red")
        except Exception as e:
            _print_panel(f"An unexpected error occurred: {e}", "System Error", "red")


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
