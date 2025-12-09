# allos/providers/openai.py

"""Provides the concrete implementation for OpenAI's modern `/v1/responses` API.

This module contains the `OpenAIProvider`, a specific and optimized implementation
for interacting with OpenAI's newer, more structured Responses API. It should be
the preferred provider for all native OpenAI models (e.g., GPT-4o, GPT-4-Turbo).

It is distinct from the `ChatCompletionsProvider`, which is designed for broader
compatibility with older OpenAI APIs and third-party services.

Key responsibilities of this provider include:
 - Translating Allos messages into the structured `input` list format required by
   the Responses API, including `function_call` and `function_call_output` items.
 - Parsing the structured `output` list from an OpenAI `Response` object.
 - Handling the event-based streaming protocol specific to the Responses API.
 - Managing authentication and wrapping OpenAI-specific errors in the standard
   `ProviderError` for consistent handling by the agent.
"""

import json
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

import openai
from openai.types.responses import (
    Response,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
)
from pydantic import ValidationError

from ..tools.base import BaseTool
from ..utils.errors import ProviderError
from ..utils.logging import logger
from .base import (
    BaseProvider,
    Message,
    MessageRole,
    ProviderChunk,
    ProviderResponse,
    ToolCall,
)
from .metadata import MetadataBuilder
from .registry import provider

# A mapping of known OpenAI models to their context window sizes (in tokens)
# This can be expanded over time.
MODEL_CONTEXT_WINDOWS = {
    "gpt-4o": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-3.5-turbo": 16385,
}


@provider("openai")
class OpenAIProvider(BaseProvider):
    """An Allos provider for OpenAI's modern `/v1/responses` API.

    This class implements the `BaseProvider` interface to connect the Allos Agent
    with OpenAI's language models via their most current API. It handles the
    translation of messages and tools, API authentication, and error handling
    specific to the `openai` Python library.

    Authentication is handled automatically by the `openai` library, which primarily
    looks for the `OPENAI_API_KEY` environment variable.

    Attributes:
        client (openai.OpenAI): The authenticated OpenAI API client instance.
    """

    env_var = "OPENAI_API_KEY"

    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs: Any):
        """Initializes the OpenAIProvider.

        Args:
            model: The OpenAI model to use (e.g., 'gpt-4o').
            api_key: The OpenAI API key. If not provided, it will be read from the
                     `OPENAI_API_KEY` environment variable.
            **kwargs: Additional arguments for the OpenAI client.
        """
        super().__init__(model, **kwargs)
        try:
            self.client = openai.OpenAI(api_key=api_key, **kwargs)
        except Exception as e:
            raise ProviderError(
                f"Failed to initialize OpenAI client: {e}", provider="openai"
            ) from e

    @staticmethod
    def _convert_to_openai_messages(
        messages: List[Message],
    ) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        """Converts a list of Allos Messages into the OpenAI Responses API format.

        This method performs the translation from the SDK's generic `Message`
        format to the specific, structured list of items required by the
        `/v1/responses` endpoint's `input` parameter.

        Key transformations:
        - The first `SYSTEM` message is extracted to be used as the `instructions`
          parameter in the API call.
        - `ASSISTANT` messages with text are converted to `role: "assistant"` items.
        - `ASSISTANT` messages with tool calls are converted to `type: "function_call"` items.
        - `TOOL` messages are converted to `type: "function_call_output"` items,
          linking them to the original `function_call` via the `call_id`.

        Args:
            messages: A list of `allos.providers.base.Message` objects.

        Returns:
            A tuple containing:
            - An optional string for the `instructions` (system prompt).
            - A list of dictionaries formatted for the `input` parameter.
        """
        instructions = None
        openai_messages = []

        if messages and messages[0].role == MessageRole.SYSTEM:
            instructions = messages[0].content
            messages = messages[1:]

        for msg in messages:
            if msg.role == MessageRole.TOOL:
                # The new Responses API expects a specific format for tool results
                openai_messages.append(
                    {
                        "type": "function_call_output",
                        # The ID of this item itself, can be a new UUID. Let's just use the tool_call_id for simplicity.
                        "id": f"fco_{msg.tool_call_id}",
                        "call_id": msg.tool_call_id,  # This MUST match the `call_id` of the function_call it answers
                        "status": "completed",  # Assuming success for now
                        "output": msg.content,
                    }
                )
            elif msg.role == MessageRole.USER:
                # Only include USER messages, not ASSISTANT messages
                openai_messages.append({"role": msg.role.value, "content": msg.content})
            elif msg.role == MessageRole.ASSISTANT:
                # An assistant turn can have text and/or tool calls.
                # These are represented as separate items in the history.
                if msg.content:
                    openai_messages.append(
                        {"role": "assistant", "content": msg.content}
                    )

                for tc in msg.tool_calls:
                    openai_messages.append(
                        {
                            "type": "function_call",
                            "id": f"fc_{tc.id}",
                            "call_id": tc.id,  # The correlation ID
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        }
                    )

        return instructions, openai_messages

    @staticmethod
    def _convert_to_openai_tools(tools: List[BaseTool]) -> List[Dict[str, Any]]:
        """Converts Allos tools into the OpenAI function tool format.

        Args:
            tools: A list of `allos.tools.base.BaseTool` objects.

        Returns:
            A list of dictionaries formatted for the `tools` parameter of the API.
        """
        openai_tools = []
        for tool in tools:
            properties = {}
            required_params = []
            for param in tool.parameters:
                properties[param.name] = {
                    "type": param.type,
                    "description": param.description,
                }
                if param.required:
                    required_params.append(param.name)

            param_schema: Dict[str, Any] = {
                "type": "object",
                "properties": properties,
                "required": required_params,
                "additionalProperties": False,
            }

            # Only enable strict mode if all parameters are required
            all_required = len(required_params) == len(tool.parameters)

            openai_tools.append(
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": param_schema,
                    "strict": all_required,
                }
            )
        return openai_tools

    @staticmethod
    def _parse_openai_response(
        response: Response,
    ) -> Tuple[Optional[str], List[ToolCall]]:
        """Parses an OpenAI `Response` object into the standard Allos format.

        This method iterates through the `output` list of the `Response` object,
        extracting text content from `message` items and tool call requests from
        `function_call` items.

        Args:
            response: The `openai.types.responses.Response` object from the API.

        Returns:
            A tuple containing:
            - An optional string for the message content.
            - A list of `allos.providers.base.ToolCall` objects.

        Raises:
            ProviderError: If a tool call's arguments are not valid JSON.
        """
        text_content: list[str] = []
        tool_calls: list[ToolCall] = []

        if not response.output:
            return None, []

        for item in response.output:
            if item.type == "message":
                _process_message(item, text_content)
            elif item.type == "function_call":
                _process_tool_call(item, tool_calls)

        return "".join(text_content) or None, tool_calls

    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[BaseTool]] = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Sends a request to the OpenAI Responses API.

        Args:
            messages: A list of messages forming the conversation.
            tools: An optional list of tools available for the agent.
            **kwargs: Additional provider-specific parameters.

        Returns:
            An Allos ProviderResponse object containing the LLM's reply, tool calls, and detailed metadata.

        Raises:
            ProviderError: If the API call fails due to connection issues, authentication errors, rate limits, or other API-side errors.
        """
        instructions, input_messages = self._convert_to_openai_messages(messages)

        if "max_tokens" in kwargs:
            # Map max_tokens to max_completion_tokens for Responses API
            # OR just remove it if the model doesn't support it,
            # but usually max_completion_tokens is the modern equivalent.
            kwargs["max_output_tokens"] = kwargs.pop("max_tokens")

        api_kwargs: Dict[str, Any] = {
            "model": self.model,
            "input": input_messages,
            **kwargs,
        }
        if instructions:
            api_kwargs["instructions"] = instructions
        if tools:
            api_kwargs["tools"] = self._convert_to_openai_tools(tools)
            metadata_tools = tools
        else:
            metadata_tools = []

        builder_kwargs = api_kwargs.copy()
        builder_kwargs["tools"] = metadata_tools

        start_time = time.time()
        try:
            response = self.client.responses.create(**api_kwargs)

            # --- METADATA GENERATION ---
            builder = MetadataBuilder(
                provider_name="openai",
                request_kwargs=builder_kwargs,
                start_time=start_time,
            )
            metadata = builder.with_response_obj(response).build()

            content, tool_calls = self._parse_openai_response(response)

            return ProviderResponse(
                metadata=metadata,
                content=content,
                tool_calls=tool_calls,
            )

        except (
            openai.RateLimitError,
            openai.AuthenticationError,
            openai.PermissionDeniedError,
            openai.NotFoundError,
            openai.BadRequestError,
        ) as e:
            # These errors are subclasses of APIStatusError and have a .response attribute
            error_text = (
                e.response.text
                if hasattr(e, "response") and hasattr(e.response, "text")
                else e.message
            )
            raise ProviderError(
                f"{type(e).__name__}: {error_text}", provider="openai"
            ) from e
        except openai.APIConnectionError as e:
            raise ProviderError(
                f"Connection error: {e.__cause__}", provider="openai"
            ) from e
        except openai.APIError as e:
            raise ProviderError(
                f"OpenAI API error: {e.message}", provider="openai"
            ) from e

    def stream_chat(
        self,
        messages: List[Message],
        tools: Optional[List[BaseTool]] = None,
        **kwargs: Any,
    ) -> Iterator[ProviderChunk]:
        """Sends a request to the OpenAI Responses API and streams the response.

        This method handles two types of errors:
        1. Pre-stream errors (e.g., authentication, invalid request) will raise a
           ProviderError synchronously.
        2. In-stream errors (e.g., failed data parsing, metadata generation) will
           be yielded as a ProviderChunk with the 'error' field populated, allowing
           for graceful termination of the stream by the caller.

        Args:
            messages: A list of messages forming the conversation.
            tools: An optional list of tools available for the agent.
            **kwargs: Additional provider-specific parameters.

        Yields:
            Iterator[ProviderChunk]: An iterator of `ProviderChunk` objects. Each
                                     chunk can contain text content, tool call
                                     information, final usage metadata, or an error
                                     message if a failure occurs during stream processing.

        Raises:
            ProviderError: If a fatal API error occurs before the stream begins.
        """
        instructions, input_messages = self._convert_to_openai_messages(messages)

        api_kwargs: Dict[str, Any] = {
            "model": self.model,
            "input": input_messages,
            "stream": True,  # Enable streaming
            **kwargs,
        }
        if instructions:
            api_kwargs["instructions"] = instructions
        if tools:
            api_kwargs["tools"] = self._convert_to_openai_tools(tools)
            metadata_tools = tools
        else:
            metadata_tools = []

        builder_kwargs = api_kwargs.copy()
        builder_kwargs["tools"] = metadata_tools

        start_time = time.time()

        api_call_context = {
            "builder_kwargs": builder_kwargs,
            "start_time": start_time,
        }

        try:
            stream = self.client.responses.create(**api_kwargs)
            in_progress_tool_calls: Dict[int, Dict[str, Any]] = {}

            for event in stream:
                yield from self._dispatch_event(
                    event, in_progress_tool_calls, api_call_context
                )

        except openai.APIError as e:
            raise ProviderError(
                f"OpenAI API error during streaming: {e}", provider="openai"
            ) from e

    def get_context_window(self) -> int:
        """Returns the context window size for the current model.

        It uses a hardcoded mapping of known OpenAI model identifiers to their
        official context window sizes. For unknown models, it returns a
        conservative default of 4096 tokens.

        Returns:
            An integer representing the context window size in tokens.
        """
        return MODEL_CONTEXT_WINDOWS.get(self.model, 4096)  # Default to 4k if unknown

    # --- OpenAI Specific Streaming Utility Functions ---
    def _dispatch_event(self, event, tool_state, _api_call_context):
        """Routes a streaming event to the appropriate handler method.

        This acts as a central dispatcher for the event-based streaming protocol
        of the OpenAI Responses API.

        Args:
            event: The event object from the OpenAI stream.
            tool_state: A dictionary managing the state of in-progress tool calls.
            _api_call_context: A dictionary holding the state of the API call.
        """
        handlers = {
            "response.output_text.delta": self._handle_text_delta,
            "response.output_item.added": self._handle_tool_start,
            "response.function_call_arguments.delta": self._handle_tool_args_delta,
            "response.output_item.done": self._handle_tool_done,
            "response.completed": self._handle_completed,
            "error": self._handle_error,
        }

        handler = handlers.get(event.type)
        if handler:
            yield from handler(event, tool_state, _api_call_context)

    def _handle_text_delta(self, event, _state, _api_call_context):
        """Handles the `response.output_text.delta` event."""
        yield ProviderChunk(content=event.delta)

    def _handle_tool_start(self, event, state, _api_call_context):
        """Handles the `response.output_item.added` event for a function_call."""
        if event.item.type != "function_call":
            return

        state[event.output_index] = {
            "id": event.item.call_id,
            "name": event.item.name,
            "arguments": "",
        }

        yield ProviderChunk(
            tool_call_start={
                "id": event.item.call_id,
                "name": event.item.name,
                "index": event.output_index,
            }
        )

    def _handle_tool_args_delta(self, event, state, _api_call_context):
        """Handles the `response.function_call_arguments.delta` event."""
        if event.output_index not in state:
            return

        state[event.output_index]["arguments"] += event.delta
        yield ProviderChunk(tool_call_delta=event.delta)

    def _handle_tool_done(self, event, state, _api_call_context):
        """Handles the `response.output_item.done` event for a function_call.

        This finalizes the tool call by parsing its arguments and yielding a
        `tool_call_done` chunk.
        """
        if event.item.type != "function_call":
            return

        call_state = state.get(event.output_index)
        if not call_state:
            return

        raw_args = call_state["arguments"]
        try:
            parsed_args = json.loads(raw_args)
            tool_call = ToolCall(
                id=call_state["id"],
                name=call_state["name"],
                arguments=parsed_args,
            )
            yield ProviderChunk(tool_call_done=tool_call)

        except json.JSONDecodeError as e:
            yield ProviderChunk(
                error=f"Failed to parse tool arguments for {call_state['name']}: {e}"
            )

        finally:
            del state[event.output_index]

    def _handle_completed(self, event, _state, api_call_context):
        """Handles the final 'response.completed' event from the stream.

        Builds and yields the final metadata chunk. Catches and gracefully handles
        any validation or data processing errors during this final step.
        """
        try:
            if event.response and event.response.usage:
                final_response_obj = event.response

                builder = MetadataBuilder(
                    provider_name="openai",
                    request_kwargs=api_call_context["builder_kwargs"],
                    start_time=api_call_context["start_time"],
                )
                metadata = builder.with_response_obj(final_response_obj).build()

                if not metadata.tools.tool_calls and _state:
                    from .metadata import ToolCallDetail

                    logger.debug(
                        f"Response object missing tool calls. "
                        f"Extracting {len(_state)} from streaming state."
                    )

                    tool_calls_from_state = []
                    for call_state in _state.values():
                        try:
                            parsed_args = json.loads(call_state.get("arguments", "{}"))
                            tool_calls_from_state.append(
                                ToolCallDetail(
                                    tool_call_id=call_state["id"],
                                    tool_name=call_state["name"],
                                    arguments=parsed_args,
                                )
                            )
                        except (json.JSONDecodeError, KeyError) as e:
                            logger.warning(f"Failed to parse tool call from state: {e}")
                            continue

                    if tool_calls_from_state:
                        metadata.tools.tool_calls = tool_calls_from_state
                        metadata.tools.total_tool_calls = len(tool_calls_from_state)
                logger.debug(
                    f"Response object has output: {hasattr(event.response, 'output')}"
                )
                if hasattr(event.response, "output"):
                    logger.debug(f"Output items: {len(event.response.output or [])}")
                    for item in event.response.output or []:
                        logger.debug(f"  - Item type: {item.type}")

                logger.debug(f"Streaming state has {len(_state)} items")
                logger.debug(
                    f"Built metadata has {metadata.tools.total_tool_calls} tool calls"
                )

                yield ProviderChunk(final_metadata=metadata)
        except (ValidationError, ValueError, TypeError, AttributeError) as e:
            logger.error(
                f"Error building final metadata for OpenAI stream: {e}", exc_info=True
            )
            yield ProviderChunk(
                error="Internal error: Failed to process final stream metadata."
            )

    def _handle_error(self, event, _state, _api_call_context):
        """Handles the `error` event, yielding a chunk with the error message."""
        yield ProviderChunk(error=f"API Error: {event.error.message}")


# --- OpenAI Specific Utility Functions ---


def _process_message(item: ResponseOutputMessage, text_accumulator: list[str]) -> None:
    """Parses a `ResponseOutputMessage` item from a non-streaming response.

    Extracts text from the message's content parts and appends it to the accumulator.

    Args:
        item: The `ResponseOutputMessage` object.
        text_accumulator: A list of strings to which the content will be added.
    """
    if not getattr(item, "content", None):
        return

    for content_part in item.content:
        if content_part.type == "output_text":
            text_accumulator.append(content_part.text)


def _process_tool_call(
    item: ResponseFunctionToolCall, tool_calls: list[ToolCall]
) -> None:
    """Parses a `ResponseFunctionToolCall` item from a non-streaming response.

    Extracts the tool name, ID, and arguments, then creates a `ToolCall` object
    and appends it to the `tool_calls` list.

    Args:
        item: The `ResponseFunctionToolCall` object.
        tool_calls: The list to which the parsed `ToolCall` will be added.

    Raises:
        ProviderError: If the tool call's arguments are not valid JSON.
    """
    call_id_ = getattr(item, "call_id", None)
    # id_ = getattr(item, "id", None)
    if not call_id_:
        logger.warning(
            "Skipping tool call due to missing call_id: %s",
            getattr(item, "name", "<unknown>"),
        )
        return

    if getattr(item, "arguments", None):
        try:
            parsed_arguments = json.loads(item.arguments or "{}")
        except json.JSONDecodeError as e:
            raise ProviderError(
                f"Failed to decode tool call arguments for '{item.name}': {e}",
                provider="openai",
            ) from e
    else:
        parsed_arguments = {}

    tool_calls.append(ToolCall(id=call_id_, name=item.name, arguments=parsed_arguments))
