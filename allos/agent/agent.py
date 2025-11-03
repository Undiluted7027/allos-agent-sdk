# allos/agent/agent.py

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Union, cast

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from ..context import ConversationContext
from ..providers import ProviderRegistry
from ..providers.base import BaseProvider, ProviderResponse, ToolCall
from ..tools import ToolRegistry
from ..tools.base import BaseTool, ToolPermission
from ..utils.errors import AllosError, ContextWindowExceededError, ToolExecutionError
from ..utils.logging import logger
from ..utils.token_counter import count_tokens


@dataclass
class AgentConfig:
    """Configuration for the Agent."""

    provider_name: str
    model: str
    tool_names: List[str] = field(default_factory=list)
    max_iterations: int = 10
    auto_approve: bool = False
    # Provider-specific kwargs can be added here if needed in the future


class Agent:
    """
    The core agent class that orchestrates interactions between an LLM provider
    and a set of tools.
    """

    def __init__(
        self, config: AgentConfig, context: Optional[ConversationContext] = None
    ):
        """Initializes the agent with a given configuration and optional context."""
        self.config = config
        self.context = context or ConversationContext()
        self.console = Console()

        # Initialize provider and tools from registries
        self.provider: BaseProvider = ProviderRegistry.get_provider(
            config.provider_name, model=config.model
        )
        self.tools: List[BaseTool] = [
            ToolRegistry.get_tool(name) for name in config.tool_names
        ]

    def save_session(self, filepath: Union[str, Path]) -> None:
        """
        Saves the agent's current state (config and context) to a JSON file.

        Args:
            filepath: The path to the file where the session will be saved.
        """
        self.console.print(f"[dim]💾 Saving session to '{filepath}'...[/dim]")
        session_data = {
            "config": asdict(self.config),
            "context": self.context.to_dict(),
        }
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(session_data, f, indent=2)
            self.console.print("[green]✅ Session saved successfully.[/green]")
        except (IOError, TypeError) as e:
            raise AllosError(f"Failed to save session to '{filepath}': {e}") from e

    @classmethod
    def load_session(cls, filepath: Union[str, Path]) -> "Agent":
        """
        Loads an agent's state from a JSON file and returns a new Agent instance.

        Args:
            filepath: The path to the session file to load.

        Returns:
            A new Agent instance with the loaded config and context.
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                session_data = json.load(f)

            config_data = session_data["config"]
            context_data = session_data["context"]

            config = AgentConfig(**config_data)
            context = ConversationContext.from_dict(context_data)

            return cls(config=config, context=context)
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            raise AllosError(f"Failed to load session from '{filepath}': {e}") from e

    def run(self, prompt: str) -> str:
        """
        Runs the agentic loop to process a user prompt.

        Args:
            prompt: The user's initial prompt.

        Returns:
            The final textual response from the agent.
        """
        # The run method should always add the new prompt. If a user wants to continue,
        # they can manage the context object themselves.
        self.context.add_user_message(prompt)
        self.console.print(
            Panel(f"[bold user]User:[/] {prompt}", title="Input", border_style="cyan")
        )

        for i in range(self.config.max_iterations):
            logger.debug(f"Starting agent iteration {i + 1}")

            # 1. Get LLM response based on the CURRENT full context
            llm_response = self._get_llm_response()

            # 2. Add the assistant's thinking/action to the context. This is now part of the history.
            self.context.add_assistant_message(
                llm_response.content, llm_response.tool_calls
            )

            # 3. If there are no tool calls, the loop is done. Return the final answer.
            if not llm_response.tool_calls:
                # The response had no tool calls, so it's the final answer.
                final_answer = llm_response.content or "No response generated."
                self.console.print(
                    Panel(
                        f"[bold assistant]Agent:[/] {final_answer}",
                        title="Final Response",
                        border_style="green",
                    )
                )
                return final_answer
            # 4. If there are tool calls, execute them.
            tool_results = self._execute_tool_calls(llm_response.tool_calls)

            # 5. Add the tool results to the context.
            for tool_call, result in zip(llm_response.tool_calls, tool_results):
                self.context.add_tool_result_message(tool_call.id, json.dumps(result))

            # The loop will now continue with the tool results in the context.

        # If loop finishes, it means max iterations were reached
        exhausted_message = "Agent reached maximum iterations without a final answer."
        self.console.print(Panel(exhausted_message, title="Error", border_style="red"))
        raise AllosError(exhausted_message)

    def _get_llm_response(self) -> ProviderResponse:
        """Sends the current context to the provider and gets a response."""
        self.console.print("[dim]🧠 Thinking...[/dim]")

        # --- Proactive Context Window Check ---
        # We'll use a simple token counting method for the MVP.
        # This can be made more sophisticated in the future.
        context_text = " ".join([msg.content or "" for msg in self.context.messages])
        estimated_tokens = count_tokens(context_text, model=self.config.model)

        # Get the provider's context window and leave a buffer for the response.
        context_window = self.provider.get_context_window()
        TOKEN_BUFFER = 2048  # Reserve tokens for the model's response

        if estimated_tokens > (context_window - TOKEN_BUFFER):
            error_msg = (
                f"Conversation context has grown too large. "
                f"Estimated tokens: {estimated_tokens}, "
                f"Model limit: {context_window}. "
                f"Please start a new session."
            )
            raise ContextWindowExceededError(error_msg)

        logger.debug(
            f"Context size check OK. Estimated tokens: {estimated_tokens}/{context_window}"
        )

        # The provider is responsible for handling the message history correctly.
        # We pass a shallow copy to prevent accidental mutation.
        response = self.provider.chat(
            messages=self.context.messages[:],
            tools=self.tools,
        )

        # DO NOT modify context here. The run loop is responsible for that.
        return response

    def _execute_tool_calls(self, tool_calls: List[ToolCall]) -> List[dict]:
        """Executes a list of tool calls after checking permissions."""
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.name
            tool_args = tool_call.arguments

            panel_content = (
                f"[bold tool]Tool:[/] {tool_name}\n[bold arguments]Arguments:[/] "
            )
            panel_content += json.dumps(tool_args, indent=2)
            self.console.print(
                Panel(panel_content, title="Tool Call Requested", border_style="yellow")
            )

            try:
                tool = ToolRegistry.get_tool(tool_name)

                # Check permissions before execution
                if not self._check_tool_permission(tool):
                    raise ToolExecutionError(tool_name, "Permission denied by user.")

                # Validate and execute
                tool.validate_arguments(tool_args)
                result = tool.execute(**tool_args)
                results.append(result)

                result_syntax = Syntax(
                    json.dumps(result, indent=2),
                    "json",
                    theme="monokai",
                    line_numbers=False,
                )
                self.console.print(
                    Panel(
                        result_syntax,
                        title=f"Tool Result: {tool_name}",
                        border_style="magenta",
                    )
                )

            except (AllosError, Exception) as e:
                error_result = {"status": "error", "message": str(e)}
                results.append(error_result)
                self.console.print(
                    Panel(str(e), title=f"Tool Error: {tool_name}", border_style="red")
                )

        return results

    def _check_tool_permission(self, tool: BaseTool) -> bool:
        """Checks if the agent has permission to run a tool."""
        if self.config.auto_approve:
            return True
        if tool.permission == ToolPermission.ALWAYS_ALLOW:
            return True
        if tool.permission == ToolPermission.ALWAYS_DENY:
            return False

        # Ask the user for permission
        if tool.permission == ToolPermission.ASK_USER:
            try:
                response = cast(
                    str,
                    self.console.input(
                        f"[bold yellow]❓ Allow tool '{tool.name}' to run? (y/n): [/]"
                    ),
                ).lower()
                return response == "y"
            except (KeyboardInterrupt, EOFError):
                self.console.print(
                    "\n[bold red]Permission denied by user (interrupted).[/]"
                )
                return False

        return False
