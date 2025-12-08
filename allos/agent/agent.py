# allos/agent/agent.py
"""Defines the core Agent class, the central orchestrator of the Allos SDK.

This module contains the primary logic that drives the agentic behavior in the
Allos SDK. It acts as the engine that connects the user's prompt to the
Language Model (LLM), manages the conversation history, and orchestrates the
execution of tools to fulfill complex tasks.

The Agent operates on a reason-act loop, iteratively thinking, acting, and
observing until it reaches a final answer or a set limit. It supports both
synchronous (request-response) and streaming modes of operation.

Key Classes:
    - Agent: The main class that orchestrates the interaction between the user,
             the LLM provider, and the available tools.
    - AgentConfig: A dataclass that holds the configuration for an Agent
                   instance, including the selected provider, model, tools,
                   and operational parameters.

Core Concepts:
    The "Agentic Loop" implemented in the `run` method follows these steps:
    1. A user prompt is added to the conversation context.
    2. The entire conversation context is sent to the configured LLM provider.
    3. The LLM's response is processed.
        a. If it's a textual answer, the loop concludes and the answer is returned.
        b. If it's a request to use one or more tools, the agent proceeds.
    4. The agent checks for user permissions to run the requested tools.
    5. If approved, the tools are executed with the arguments provided by the LLM.
    6. The results of the tool executions are added back to the conversation context.
    7. The loop repeats from step 2 with the updated context, allowing the agent
       to reason about the new information.

Example:
    >>> from allos.agent import Agent, AgentConfig
    >>>
    >>> # Configure the agent to use OpenAI and the shell execution tool.
    >>> # Ensure the OPENAI_API_KEY environment variable is set.
    >>> config = AgentConfig(
    ...     provider_name="openai",
    ...     model="gpt-4o",
    ...     tool_names=["shell_exec"],
    ...     auto_approve=True  # Use with caution
    ... )
    >>>
    >>> # Instantiate the agent with the configuration.
    >>> agent = Agent(config=config)
    >>>
    >>> # Define a prompt that requires tool use.
    >>> prompt = "What is the current date in UTC?"
    >>>
    >>> # Run the agent to get the final answer.
    >>> final_answer = agent.run(prompt)
    >>> print(final_answer)
"""

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Tuple,
    TypedDict,
    Union,
    cast,
)

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from ..context import ConversationContext
from ..providers import ProviderRegistry
from ..providers.base import BaseProvider, ProviderChunk, ProviderResponse, ToolCall
from ..providers.metadata import Metadata, ToolCallDetail, TurnLog, TurnTokensUsed
from ..tools import ToolRegistry
from ..tools.base import BaseTool, ToolPermission
from ..utils.errors import AllosError, ContextWindowExceededError, ToolExecutionError
from ..utils.logging import logger
from ..utils.modality_counter import calculate_modality_usage
from ..utils.token_counter import count_tokens

ToolExecutionResult = Tuple[Dict[str, Any], ToolCallDetail]


@dataclass
class AgentConfig:
    """Configuration for the Agent."""

    provider_name: str
    model: str
    tool_names: List[str] = field(default_factory=list)
    max_iterations: int = 10
    auto_approve: bool = False
    max_tokens: Optional[int] = None
    no_tools: bool = False
    # field for custom API endpoints
    base_url: Optional[str] = None
    # api_key, exclude from repr for security logs (for providers like Together AI)
    api_key: Optional[str] = field(default=None, repr=False)
    # Provider-specific kwargs can be added here if needed in the future


class CumulativeState(TypedDict):
    """A TypedDict for tracking the accumulated state across all turns in a single agent run.

    This object aggregates metrics like token counts, costs, and a complete history
    of tool calls and turns. It is initialized at the beginning of an agent run
    and progressively updated after each iteration of the agentic loop.

    Attributes:
        all_tool_details: A list of all `ToolCallDetail` objects from every tool
                          call in the run, including execution status and timing.
        input_tokens: The cumulative count of input tokens sent to the provider.
        output_tokens: The cumulative count of output tokens received from the provider.
        cost: The total estimated cost in USD for the entire run.
        last_metadata: The most recent `Metadata` object received from the provider.
                       This is used as a base for building the final aggregate metadata.
        turn_history: A list of `TurnLog` objects, recording the details of each
                      iteration of the agentic loop.
    """

    all_tool_details: List[ToolCallDetail]
    input_tokens: int
    output_tokens: int
    cost: float
    last_metadata: Optional[Metadata]
    turn_history: List[Any]


class Agent:
    """The core agent class that orchestrates interactions between an LLM and tools.

    The Agent is the central engine of the Allos SDK. It manages the entire
    lifecycle of a task, from receiving a user prompt to generating a final
    response. It operates on a reason-act loop, allowing it to solve complex
    problems by iteratively using tools to gather information and build towards
    a solution.

    The Agent is designed to be LLM-agnostic through the use of `Providers` and
    can be equipped with various capabilities through `Tools`. Its behavior is
    configured via an `AgentConfig` object.

    Attributes:
        config (AgentConfig): The configuration object that defines the agent's
                              behavior, including the provider, model, and tools to use.
        context (ConversationContext): The conversation history manager, which serves
                                       as the agent's short-term memory.
        console (Console): A rich Console object for formatted output.
        provider (BaseProvider): The instantiated LLM provider client.
        tools (List[BaseTool]): A list of instantiated tools available to the agent.
        last_run_metadata (Optional[Metadata]): A comprehensive metadata object
                                                detailing the metrics of the last
                                                completed `run` or `stream_run`.
                                                This is populated after a run finishes.
    """

    def __init__(
        self, config: AgentConfig, context: Optional[ConversationContext] = None
    ):
        """Initializes the agent with a given configuration and optional context."""
        self.config = config
        self.context = context or ConversationContext()
        self.console = Console()

        # Initialize provider and tools from registries
        # Pass base_url if it exists in the config
        provider_kwargs: Dict[str, Union[str, int]] = {}
        if config.base_url:
            provider_kwargs["base_url"] = config.base_url
        if config.api_key:
            provider_kwargs["api_key"] = config.api_key

        # Initialize provider and tools from registries
        self.provider: BaseProvider = ProviderRegistry.get_provider(
            config.provider_name, model=config.model, **provider_kwargs
        )
        # Initialize tools
        # If no_tools is set, we ignore tool_names and load nothing.
        if config.no_tools:
            self.tools: List[BaseTool] = []
        else:
            self.tools = [ToolRegistry.get_tool(name) for name in config.tool_names]

    def save_session(self, filepath: Union[str, Path]) -> None:
        """Saves the agent's current state (config and context) to a JSON file.

        Args:
            filepath: The path to the file where the session will be saved.

        Raises:
            AllosError: If the session fails to save due to an IO or Type error.
        """
        self.console.print(f"[dim]💾 Saving session to '{filepath}'...[/dim]")
        config_data = asdict(self.config)
        if "api_key" in config_data:
            config_data.pop("api_key")
        session_data = {
            "config": config_data,
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
        """Loads an agent's state from a JSON file and returns a new Agent instance.

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
        """Runs the agentic loop to process a user prompt.

        Args:
            prompt: The user's initial prompt.

        Returns:
            The final textual response from the agent.

        Raises:
            ContextWindowExceededError: If the conversation history exceeds the
            model's context window before execution.

            ProviderError: If the underlying LLM provider returns an error.

            ToolExecutionError: If a tool fails during execution and is not handled
            internally.

            AllosError: If the agent reaches its maximum iteration limit without
            producing a final answer.
        """
        # The run method should always add the new prompt. If a user wants to continue,
        # they can manage the context object themselves.
        self.context.add_user_message(prompt)
        self.console.print(
            Panel(f"[bold user]User:[/] {prompt}", title="Input", border_style="cyan")
        )

        # Initialize cumulative tracking
        cumulative_state: CumulativeState = {
            "all_tool_details": [],
            "input_tokens": 0,
            "output_tokens": 0,
            "cost": 0.0,
            "last_metadata": None,
            "turn_history": [],
        }

        for i in range(self.config.max_iterations):
            logger.debug(f"Starting agent iteration {i + 1}")
            turn_start_time = time.time()

            # 1. Get LLM response based on the CURRENT full context
            llm_response = self._get_llm_response()

            turn_duration_ms = int((time.time() - turn_start_time) * 1000)

            # Track this turn in history
            self._record_turn(
                turn_number=i + 1,
                metadata=llm_response.metadata,
                tool_calls=llm_response.tool_calls,
                duration_ms=turn_duration_ms,
                cumulative_state=cumulative_state,
            )

            # Accumulate usage state
            self._accumulate_usage_stats(llm_response.metadata, cumulative_state)

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
                self._finalize_run_metadata(cumulative_state)
                return final_answer
            # 4. If there are tool calls, execute them.
            tool_execution_results = self._execute_tool_calls(llm_response.tool_calls)

            # 5. Add the tool results to the context.
            for tool_call, (result, tool_detail) in zip(
                llm_response.tool_calls, tool_execution_results
            ):
                cumulative_state["all_tool_details"].append(tool_detail)
                self.context.add_tool_result_message(tool_call.id, json.dumps(result))

            # The loop will now continue with the tool results in the context.

        # If loop finishes, it means max iterations were reached
        self._finalize_run_metadata(cumulative_state)
        exhausted_message = "Agent reached maximum iterations without a final answer."
        self.console.print(Panel(exhausted_message, title="Error", border_style="red"))
        raise AllosError(exhausted_message)

    def _record_turn(
        self,
        turn_number: int,
        metadata: Metadata,
        tool_calls: List[ToolCall],
        duration_ms: int,
        cumulative_state: CumulativeState,
    ) -> None:
        """Records a turn in the turn history."""
        turn_log = TurnLog(
            turn_number=turn_number,
            model_used=metadata.model.model_id,
            content_type="tool_calls" if tool_calls else "text_response",
            tokens_used=TurnTokensUsed(
                input_tokens=metadata.usage.input_tokens,
                output_tokens=metadata.usage.output_tokens,
            ),
            duration_ms=duration_ms,
            tools_called=[tc.name for tc in tool_calls],
            stop_reason=metadata.quality_signals.finish_reason,
        )
        cumulative_state["turn_history"].append(turn_log)

    def _finalize_run_metadata(self, cumulative_state: CumulativeState) -> None:
        """Creates aggregate metadata and stores it as instance variable."""
        if cumulative_state["last_metadata"]:
            self.last_run_metadata: Optional[Metadata] = (
                self._create_aggregate_metadata(
                    cumulative_state["last_metadata"],
                    cumulative_state["all_tool_details"],
                    cumulative_state["turn_history"],
                    cumulative_state["input_tokens"],
                    cumulative_state["output_tokens"],
                    cumulative_state["cost"],
                )
            )
        else:
            self.last_run_metadata = None

    def _create_aggregate_metadata(
        self,
        base_metadata: Metadata,
        all_tool_details: List[ToolCallDetail],
        turn_history: List[TurnLog],
        total_input_tokens: int,
        total_output_tokens: int,
        total_cost: float,
    ) -> Metadata:
        """Creates aggregated metadata for entire agentic run."""
        from copy import deepcopy

        aggregate = deepcopy(base_metadata)

        # Update turns
        aggregate.turns.total_turns = len(turn_history)
        aggregate.turns.turn_history = turn_history
        aggregate.turns.max_turns_reached = (
            len(turn_history) >= self.config.max_iterations
        )

        # Update tool calls
        aggregate.tools.total_tool_calls = len(all_tool_details)
        aggregate.tools.tool_calls = all_tool_details
        # aggregate.tools.tool_calls = [
        #     ToolCallDetail(
        #         tool_call_id=tc.id,
        #         tool_name=tc.name,
        #         arguments=tc.arguments,
        #     )
        #     for tc in all_tool_calls
        # ]

        # Update usage
        aggregate.usage.total_tokens = total_input_tokens + total_output_tokens
        aggregate.usage.input_tokens = total_input_tokens
        aggregate.usage.output_tokens = total_output_tokens

        if aggregate.usage.estimated_cost:
            aggregate.usage.estimated_cost.total_usd = total_cost

        return aggregate

    def stream_run(self, prompt: str) -> Iterator[ProviderChunk]:
        """Runs the agentic loop in a streaming fashion, yielding chunks back to the caller.

        Args:
            prompt: The user's initial prompt.

        Yields:
            ProviderChunk: An iterator of chunks representing the streaming response. Chunks can contain content, tool call data, or final

        Raises:
            ContextWindowExceededError: If the conversation history exceeds the
            model's context window before execution.

            ProviderError: If the underlying LLM provider returns an error.

            ToolExecutionError: If a tool fails during execution and is not handled
            internally.

            AllosError: If the agent reaches its maximum iteration limit without
            producing a final answer.
        """
        self.context.add_user_message(prompt)
        # Initialize cumulative tracking across all iterations
        cumulative_state: CumulativeState = {
            "all_tool_details": [],
            "input_tokens": 0,
            "output_tokens": 0,
            "cost": 0.0,
            "last_metadata": None,
            "turn_history": [],
        }

        for i in range(self.config.max_iterations):
            logger.debug(f"Starting streaming agent iteration {i + 1}")
            self.console.print("[dim]🧠 Thinking (streaming)...[/dim]")

            turn_start_time = time.time()

            # Process a single streaming iteration
            iteration_result = yield from self._process_streaming_iteration(
                cumulative_state,
                turn_start_time,
            )

            turn_duration_ms = int((time.time() - turn_start_time) * 1000)

            # --- Construct and store the TurnLog for this iteration ---
            last_meta: Optional[Metadata] = cumulative_state.get("last_metadata")
            if last_meta:
                last_meta.latency.time_to_first_token_ms = iteration_result.get(
                    "ttft_ms"
                )
                turn_log = TurnLog(
                    turn_number=i + 1,
                    model_used=last_meta.model.model_id,
                    content_type=(
                        "tool_calls"
                        if iteration_result["tool_calls"]
                        else "text_response"
                    ),
                    tokens_used=TurnTokensUsed(
                        input_tokens=last_meta.usage.input_tokens,
                        output_tokens=last_meta.usage.output_tokens,
                    ),
                    duration_ms=turn_duration_ms,
                    tools_called=[tc.name for tc in iteration_result["tool_calls"]],
                    stop_reason=last_meta.quality_signals.finish_reason,
                )
                cumulative_state["turn_history"].append(turn_log)

            # Update context with iteration results
            self._update_context_after_streaming(
                iteration_result["content"], iteration_result["tool_calls"]
            )

            # If no tools were called, we're done
            if not iteration_result["tool_calls"]:
                yield from self._yield_final_aggregate_metadata(cumulative_state)
                return

            # Execute tools and prepare for next iteration
            self._execute_and_record_tools(
                iteration_result["tool_calls"], cumulative_state
            )

        # If loop finishes, we've exceeded max iterations
        raise AllosError("Agent reached maximum iterations without a final answer.")

    def _process_streaming_iteration(
        self,
        cumulative_state: CumulativeState,
        turn_start_time: float,
    ) -> Generator[ProviderChunk, None, Dict[str, Any]]:
        """Processes a single streaming iteration, yielding chunks and accumulating state.

        Args:
            cumulative_state: TypedDict tracking cumulative stats across all iterations.
            turn_start_time: Time when the turn started.

        Yields:
            ProviderChunk: Chunks from the provider stream.

        Returns:
            Dict containing accumulated content and tool_calls from this iteration.
        """
        accumulated_content: List[str] = []
        iteration_tool_calls: List[ToolCall] = []

        # TTFT Calculation State
        time_to_first_token_ms: Optional[int] = None
        first_chunk_received = False

        # Get streaming response from provider
        stream = self._get_provider_stream()

        # Process each chunk from the stream
        for chunk in stream:
            # Calculate TTFT on first content chunk
            if not first_chunk_received and chunk.content:
                time_to_first_token_ms = int((time.time() - turn_start_time) * 1000)
                first_chunk_received = True

            # Accumulate chunk data
            if chunk.content:
                accumulated_content.append(chunk.content)

            if chunk.tool_call_done:
                iteration_tool_calls.append(chunk.tool_call_done)
                # cumulative_state["all_tool_details"].append(chunk.tool_call_done)

            if chunk.final_metadata:
                self._update_chunk_metadata(chunk, iteration_tool_calls)
                self._accumulate_usage_stats(chunk.final_metadata, cumulative_state)
                yield chunk
            else:
                yield chunk

            if chunk.error:
                raise AllosError(f"Streaming provider error: {chunk.error}")

        # Return iteration results
        return {
            "content": "".join(accumulated_content),
            "tool_calls": iteration_tool_calls,
            "ttft_ms": time_to_first_token_ms,
        }

    def _get_provider_stream(self) -> Iterator[ProviderChunk]:
        """Gets the streaming iterator from the provider with configured parameters."""
        chat_kwargs: Dict[str, Any] = {}
        if self.config.max_tokens:
            chat_kwargs["max_tokens"] = self.config.max_tokens
        if self.tools:
            chat_kwargs["tools"] = self.tools

        return self.provider.stream_chat(
            messages=self.context.messages[:], **chat_kwargs
        )

    def _update_chunk_metadata(
        self, chunk: ProviderChunk, tool_calls: List[ToolCall]
    ) -> None:
        """Updates chunk metadata with tool call information for this iteration."""
        if chunk.final_metadata:
            chunk.final_metadata.tools.total_tool_calls = len(tool_calls)
            chunk.final_metadata.tools.tool_calls = [
                ToolCallDetail(
                    tool_call_id=tc.id,
                    tool_name=tc.name,
                    arguments=tc.arguments,
                )
                for tc in tool_calls
            ]

    def _accumulate_usage_stats(
        self, metadata: Metadata, cumulative_state: CumulativeState
    ) -> None:
        """Accumulates token usage and cost statistics across iterations."""
        cumulative_state["last_metadata"] = metadata
        cumulative_state["input_tokens"] += metadata.usage.input_tokens
        cumulative_state["output_tokens"] += metadata.usage.output_tokens

        if metadata.usage.estimated_cost:
            cumulative_state["cost"] += metadata.usage.estimated_cost.total_usd

    def _update_context_after_streaming(
        self, content: str, tool_calls: List[ToolCall]
    ) -> None:
        """Updates conversation context with streaming iteration results."""
        self.context.add_assistant_message(
            content=content if content else None,
            tool_calls=tool_calls,
        )

    def _yield_final_aggregate_metadata(
        self, cumulative_state: CumulativeState
    ) -> Iterator[ProviderChunk]:
        """Creates and yields final aggregate metadata if tool calls were made."""
        if cumulative_state["last_metadata"]:  # and cumulative_state["all_tool_calls"]:
            final_aggregate = self._create_aggregate_metadata(
                cumulative_state["last_metadata"],
                cumulative_state["all_tool_details"],
                cumulative_state["turn_history"],
                cumulative_state["input_tokens"],
                cumulative_state["output_tokens"],
                cumulative_state["cost"],
            )
            yield ProviderChunk(final_metadata=final_aggregate)

    def _execute_and_record_tools(
        self, tool_calls: List[ToolCall], cumulative_state: CumulativeState
    ) -> None:
        """Executes tool calls and records results in context."""
        tool_execution_results = self._execute_tool_calls(tool_calls)

        # Store enriched tool details and add results to context
        for tool_call, (result, tool_detail) in zip(tool_calls, tool_execution_results):
            cumulative_state["all_tool_details"].append(tool_detail)
            self.context.add_tool_result_message(tool_call.id, json.dumps(result))

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
        # --- Pre-Computation Step (Future) ---
        # This step would analyze the input for multi-model content.
        # The result of this can be passed to MetadataBuilder later.
        modality_details = calculate_modality_usage(  # noqa: F841
            self.context.messages[:]
        )
        # `modality_details` would then be stored and injected into the metadata object when it's built.

        logger.debug(
            f"Context size check OK. Estimated tokens: {estimated_tokens}/{context_window}"
        )

        # Prepare kwargs for chat
        chat_kwargs: Dict[str, Any] = {}
        if self.config.max_tokens:
            chat_kwargs["max_tokens"] = self.config.max_tokens

        # Only pass tools if we have them
        if self.tools:
            chat_kwargs["tools"] = self.tools

        # The provider is responsible for handling the message history correctly.
        # We pass a shallow copy to prevent accidental mutation.
        response = self.provider.chat(messages=self.context.messages[:], **chat_kwargs)

        # DO NOT modify context here. The run loop is responsible for that.
        return response

    def _execute_tool_calls(
        self, tool_calls: List[ToolCall]
    ) -> List[ToolExecutionResult]:
        """Executes a list of tool calls after checking permissions."""
        results: List[ToolExecutionResult] = []
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

            tool_start_time = time.time()
            status = "success"
            result_dict = {}

            try:
                tool = ToolRegistry.get_tool(tool_name)

                # Check permissions before execution
                if not self._check_tool_permission(tool):
                    raise ToolExecutionError(tool_name, "Permission denied by user.")

                # Validate and execute
                tool.validate_arguments(tool_args)
                result_dict = tool.execute(**tool_args)

                result_syntax = Syntax(
                    json.dumps(result_dict, indent=2),
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
                status = "error"
                result_dict = {"status": "error", "message": str(e)}
                self.console.print(
                    Panel(str(e), title=f"Tool Error: {tool_name}", border_style="red")
                )
            execution_time_ms = int((time.time() - tool_start_time) * 1000)

            tool_detail = ToolCallDetail(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                arguments=tool_call.arguments,
                execution_time_ms=execution_time_ms,
                status=status,
            )
            results.append((result_dict, tool_detail))

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
