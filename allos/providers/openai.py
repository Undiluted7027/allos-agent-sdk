# allos/providers/openai.py

import json
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

import openai
from openai.types.responses import (
    Response,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
)

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
    """
    An Allos provider for interacting with the OpenAI API, specifically using the
    new Responses API (`/v1/responses`).
    """

    env_var = "OPENAI_API_KEY"

    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs: Any):
        """
        Initializes the OpenAIProvider.

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
        """
        Converts a list of Allos Messages into the format expected by the
        OpenAI Responses API, separating the system prompt.

        Returns:
            A tuple containing the instructions (system prompt) and the list of
            input messages.
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
        """Converts a list of Allos BaseTools into the OpenAI function tool format."""
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
        """Parses the OpenAI Response object into an Allos ProviderResponse."""
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
        """
        Sends a request to the OpenAI Responses API.

        Args:
            messages: A list of messages forming the conversation.
            tools: An optional list of tools available for the agent.
            **kwargs: Additional provider-specific parameters.

        Returns:
            An Allos ProviderResponse object.
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
        """
        Sends a request to the OpenAI Responses API and streams the response.
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
        """Returns the context window size for the current model."""
        return MODEL_CONTEXT_WINDOWS.get(self.model, 4096)  # Default to 4k if unknown

    # --- OpenAI Specific Streaming Utility Functions ---
    def _dispatch_event(self, event, tool_state, _api_call_context):
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
        yield ProviderChunk(content=event.delta)

    def _handle_tool_start(self, event, state, _api_call_context):
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
        if event.output_index not in state:
            return

        state[event.output_index]["arguments"] += event.delta
        yield ProviderChunk(tool_call_delta=event.delta)

    def _handle_tool_done(self, event, state, _api_call_context):
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
        logger.info(f"Response object has output: {hasattr(event.response, 'output')}")
        if hasattr(event.response, "output"):
            logger.info(f"Output items: {len(event.response.output or [])}")
            for item in event.response.output or []:
                logger.info(f"  - Item type: {item.type}")

        logger.info(f"Streaming state has {len(_state)} items")
        logger.info(f"Built metadata has {metadata.tools.total_tool_calls} tool calls")

        yield ProviderChunk(final_metadata=metadata)

    def _handle_error(self, event, _state, _api_call_context):
        yield ProviderChunk(error=f"API Error: {event.error.message}")


# --- OpenAI Specific Utility Functions ---


def _process_message(item: ResponseOutputMessage, text_accumulator: list[str]) -> None:
    if not getattr(item, "content", None):
        return

    for content_part in item.content:
        if content_part.type == "output_text":
            text_accumulator.append(content_part.text)


def _process_tool_call(
    item: ResponseFunctionToolCall, tool_calls: list[ToolCall]
) -> None:
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
