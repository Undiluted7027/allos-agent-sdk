# allos/providers/cohere.py

"""Provides the concrete implementation for interacting Cohere's V2 API."""

import json
import time
from typing import Any, Dict, Iterator, List, NoReturn, Optional, Tuple, cast

import cohere
from cohere.core import ApiError
from cohere.errors import (
    BadRequestError,
    ClientClosedRequestError,
    ForbiddenError,
    GatewayTimeoutError,
    InternalServerError,
    InvalidTokenError,
    NotFoundError,
    ServiceUnavailableError,
    TooManyRequestsError,
    UnauthorizedError,
    UnprocessableEntityError,
)
from cohere.errors import (
    NotImplementedError as CohereNotImplementedError,
)
from cohere.types import (
    AssistantChatMessageV2,
    AssistantMessageResponse,
    ChatMessageV2,
    GetModelResponse,
    SystemChatMessageV2,
    TextAssistantMessageV2ContentOneItem,
    TextContent,
    ToolCallV2,
    ToolCallV2Function,
    ToolChatMessageV2,
    ToolV2,
    ToolV2Function,
    UserChatMessageV2,
)
from cohere.v2.types import V2ChatResponse

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


@provider("cohere")
class CohereProvider(BaseProvider):
    """An Allos provider for interacting with the Cohere API."""

    env_var = "COHERE_API_KEY"

    _FOUR_XX_ERRORS = (
        ClientClosedRequestError,
        InvalidTokenError,
        NotFoundError,
        ForbiddenError,
        BadRequestError,
        UnauthorizedError,
        TooManyRequestsError,
        UnprocessableEntityError,
    )

    _FIVE_XX_ERRORS = (
        GatewayTimeoutError,
        InternalServerError,
        ServiceUnavailableError,
        CohereNotImplementedError,
    )

    @staticmethod
    def _safe_error_body(error: Exception) -> str:
        """Safely extract the error body from a Cohere API exception.

        Args:
            error: The exception from which to extract the body.

        Returns:
            The error body as a string, or the string representation of the error if no body is available.
        """
        body = getattr(error, "body", None)
        if body is None:
            return str(error)
        if isinstance(body, str):
            return body
        try:
            return json.dumps(body)
        except Exception:
            return str(body)

    def _raise_provider_error(self, error: Exception, prefix: str) -> NoReturn:
        """Raise a ProviderError with formatted error message.

        Args:
            error: The original exception that occurred.
            prefix: A prefix to add to the error message for context.

        Raises:
            ProviderError: Always raises with the appropriate error message.
        """
        if isinstance(error, self._FOUR_XX_ERRORS + self._FIVE_XX_ERRORS + (ApiError,)):
            raise ProviderError(
                f"{prefix}: {self._safe_error_body(error)}", provider="cohere"
            ) from error
        raise ProviderError(f"{prefix}: {error}", provider="cohere") from error

    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs: Any):
        """Initialize the Cohere provider.

        Args:
            model: The Cohere model name to use.
            api_key: The Cohere API key (if not provided, will use COHERE_API_KEY env var).
            **kwargs: Additional arguments to pass to the Cohere client.

        Raises:
            ProviderError: If authentication fails or the model is not available.
        """
        super().__init__(model, **kwargs)
        # This will be populated by _verify_model_available()
        self._model_context_window: Optional[int] = None
        try:
            self.client = cohere.ClientV2(api_key=api_key, **kwargs)
            self._verify_model_available()
        except ProviderError:
            raise
        except (*self._FOUR_XX_ERRORS, *self._FIVE_XX_ERRORS) as e:
            self._raise_provider_error(e, "Authentication/configuration error")
        except ApiError as e:
            # Catch-all for other API errors
            self._raise_provider_error(e, "Cohere API error")
        except Exception as e:
            # Non-API errors (network, etc.)
            raise ProviderError(
                f"Failed to initialize Cohere client: {e}", provider="cohere"
            ) from e

    def _verify_model_available(self):
        """Check if the configured model is available.

        This method verifies that:
        1. The model name is valid and can be used with Cohere APIs.
        2. Retrieves the model's actual context window size (if available).
        """
        try:
            pulled_models = self.client.models.list()
            available_model_names = {m.name for m in pulled_models.models}
            if self.model not in available_model_names:
                # Find closest matching models for better error message
                suggestions = self._find_similar_models(
                    self.model, pulled_models.models
                )

                error_msg = f"Model '{self.model}' not available."
                if suggestions:
                    error_msg += "\n\nDid you mean one of these?\n"
                    for suggestion in suggestions[:5]:  # Show top 5 suggestions
                        error_msg += f"  - {suggestion}\n"
                else:
                    shortened_models_list = list(pulled_models.models)
                    error_msg += (
                        f"\n\nAvailable models: {', '.join(sorted([m.name for m in shortened_models_list[:10] if m.name is not None]))}"
                        if pulled_models.models
                        else ""
                    )
                raise ProviderError(error_msg, provider="cohere")
            # Get detailed info
            model_info = next(
                (m for m in pulled_models.models if m.name == self.model), None
            )
            # Extract context window from model info
            if model_info and model_info.context_length:
                self._model_context_window = int(model_info.context_length)
                if self._model_context_window:
                    logger.debug(
                        f"Model '{self.model}' context window: "
                        f"{self._model_context_window} tokens"
                    )
        except ApiError as e:
            raise ProviderError(
                f"Could not verify model '{self.model}': {e.body}", provider="cohere"
            ) from e

    def _find_similar_models(
        self, requested_model: str, available_models: List[GetModelResponse]
    ) -> List[str]:
        """Find models similar to the requested model.

        Args:
            requested_model: The model name that was requested
            available_models: List of available model objects

        Returns:
            List of similar model names, sorted by similarity
        """
        import difflib

        # Extract clean model IDs from available models
        available_ids = [m.name for m in available_models if m.name is not None]

        # Use difflib to find close matches
        # cutoff = 0.4 means at least 40% similar
        close_matches = difflib.get_close_matches(
            requested_model, available_ids, n=5, cutoff=0.4
        )

        # If no close matches, suggest models with similar prefixes
        if not close_matches:
            requested_prefix = (
                requested_model.split("-")[0]
                if "-" in requested_model
                else requested_model
            )
            prefix_matches = [
                model_id
                for model_id in available_ids
                if model_id.startswith(requested_prefix)
            ]
            return sorted(prefix_matches)[:5]
        return close_matches

    def _convert_user_message(self, msg: Message) -> UserChatMessageV2:
        """Convert an Allos user message to Cohere V2 format.

        Args:
            msg: The Allos message to convert.

        Returns:
            A Cohere UserChatMessageV2 object.
        """
        return UserChatMessageV2(
            role="user", content=[TextContent(type="text", text=msg.content or "")]
        )

    def _convert_assistant_message(self, msg: Message) -> AssistantChatMessageV2:
        """Converts an assistant message to Cohere format."""
        kwargs: Dict[str, Any] = {}

        if msg.content:
            kwargs["content"] = [
                TextAssistantMessageV2ContentOneItem(type="text", text=msg.content)
            ]

        if msg.tool_calls:
            kwargs["tool_calls"] = self._convert_tool_calls(msg)
        return AssistantChatMessageV2(role="assistant", **kwargs)

    def _convert_tool_calls(self, msg: Message) -> List[cohere.ToolCallV2]:
        """Convert tool calls to Cohere function call parts."""
        parts: List[ToolCallV2] = []
        for tc in msg.tool_calls:
            function = ToolCallV2Function(
                name=tc.name, arguments=json.dumps(tc.arguments)
            )
            parts.append(ToolCallV2(id=tc.id, type="function", function=function))
        return parts

    def _convert_tool_message(self, msg: Message) -> ToolChatMessageV2:
        """Convert a tool result message to Cohere format."""
        return ToolChatMessageV2(
            role="tool", tool_call_id=msg.tool_call_id or "", content=msg.content or ""
        )

    def _convert_system_message(self, msg: Message) -> SystemChatMessageV2:
        """Convert an Allos system message to Cohere V2 format.

        Args:
            msg: The Allos message to convert.

        Returns:
            A Cohere SystemChatMessageV2 object.
        """
        return SystemChatMessageV2(role="system", content=msg.content or "")

    def _convert_messages(self, messages: List[Message]) -> List[ChatMessageV2]:
        """Convert Allos messages to Cohere format."""
        contents: List[ChatMessageV2] = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                contents.append(self._convert_system_message(msg))
            elif msg.role == MessageRole.USER:
                contents.append(self._convert_user_message(msg))
            elif msg.role == MessageRole.ASSISTANT:
                contents.append(self._convert_assistant_message(msg))
            elif msg.role == MessageRole.TOOL:
                contents.append(self._convert_tool_message(msg))

        return contents

    @staticmethod
    def _convert_tools(tools: List[BaseTool]) -> List[ToolV2]:
        """Convert Allos tools to Cohere V2 tool schema."""
        converted_tools: List[ToolV2] = []

        for tool in tools:
            properties: Dict[str, Dict[str, str]] = {}
            required_params: List[str] = []

            for param in tool.parameters:
                properties[param.name] = {
                    "type": param.type,
                    "description": param.description,
                }
                if param.required:
                    required_params.append(param.name)

            schema: Dict[str, Any] = {
                "type": "object",
                "properties": properties,
                "required": required_params,
                "additionalProperties": False,
            }

            converted_tools.append(
                ToolV2(
                    type="function",
                    function=ToolV2Function(
                        name=tool.name,
                        description=tool.description,
                        parameters=schema,
                    ),
                )
            )
        return converted_tools

    @staticmethod
    def _parse_response(
        response: AssistantMessageResponse,
    ) -> Tuple[Optional[str], List[ToolCall]]:
        """Parses the Cohere Message object into an Allos ProviderResponse."""
        text_accumulator: List[str] = []
        tool_calls: List[ToolCall] = []

        if not response.content:
            return None, []

        for block in response.content:
            if block.type == "text":
                if block.text:
                    text_accumulator.append(block.text)

        if response.tool_calls:
            for tc in response.tool_calls:
                tool_id = tc.id
                tool_name = tc.function.name if tc.function else None
                args_str = tc.function.arguments if tc.function else None

                if not tool_id or not tool_name:
                    logger.warning(
                        "Skipping tool call due to missing ID or name: %s",
                        tool_name or "<unknown>",
                    )
                    continue

                try:
                    parsed_args = json.loads(args_str) if args_str else {}
                except json.JSONDecodeError as e:
                    raise ProviderError(
                        f"Failed to decode tool arguments for '{tool_name}'",
                        provider="cohere",
                    ) from e
                if not isinstance(parsed_args, dict):
                    parsed_args = {}
                tool_calls.append(
                    ToolCall(id=tool_id, name=tool_name, arguments=parsed_args)
                )

        return "".join(text_accumulator) or None, tool_calls

    @staticmethod
    def _extract_usage_tokens(usage: Any) -> Tuple[int, int]:
        """Extract input/output tokens from Cohere usage object."""
        if usage is None:
            return 0, 0
        input_tokens = 0
        output_tokens = 0

        if getattr(usage, "tokens", None):
            input_tokens = int(getattr(usage.tokens, "input_tokens", 0) or 0)
            output_tokens = int(getattr(usage.tokens, "output_tokens", 0) or 0)

        if input_tokens == 0 and getattr(usage, "billed_units", None):
            input_tokens = int(getattr(usage.billed_units, "input_tokens", 0) or 0)
        if output_tokens == 0 and getattr(usage, "billed_units", None):
            output_tokens = int(getattr(usage.billed_units, "output_tokens", 0) or 0)

        return input_tokens, output_tokens

    def _build_metadata_from_usage(
        self,
        *,
        response_id: str,
        input_tokens: int,
        output_tokens: int,
        tool_calls: List[ToolCall],
        request_kwargs: Dict[str, Any],
        start_time: float,
    ):
        """Build metadata from Cohere response/stream usage with a synthetic response object."""
        usage_obj = type(
            "CohereUsageObj",
            (),
            {"input_tokens": input_tokens, "output_tokens": output_tokens},
        )()

        output_items = []
        for tc in tool_calls:
            output_items.append(
                type(
                    "CohereToolOutput",
                    (),
                    {
                        "type": "function_call",
                        "call_id": tc.id,
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                )()
            )

        synthetic_response = type(
            "CohereSyntheticResponse",
            (),
            {
                "id": response_id,
                "model": self.model,
                "status": "completed",
                "usage": usage_obj,
                "output": output_items,
            },
        )()

        builder = MetadataBuilder(
            provider_name="cohere", request_kwargs=request_kwargs, start_time=start_time
        )

        return builder.with_response_obj(synthetic_response).build()

    @staticmethod
    def _extract_stream_text(event: Any) -> Optional[str]:
        """Extract text content from a Cohere stream event.

        Args:
            event: The stream event from Cohere.

        Returns:
            The extracted text, or None if no text is present.
        """
        return getattr(
            getattr(
                getattr(getattr(event, "delta", None), "message", None),
                "content",
                None,
            ),
            "text",
            None,
        )

    @staticmethod
    def _extract_stream_tool_call(event: Any) -> Tuple[Optional[int], Optional[Any]]:
        """Extract tool call information from a Cohere stream event.

        Args:
            event: The stream event from Cohere.

        Returns:
            A tuple of (index, tool_call) where index is the tool call index and tool_call is the tool call data.
        """
        return (
            getattr(event, "index", None),
            getattr(
                getattr(getattr(event, "delta", None), "message", None),
                "tool_calls",
                None,
            ),
        )

    @staticmethod
    def _extract_stream_arguments_delta(event: Any) -> Optional[str]:
        """Extract incremental tool arguments from a Cohere stream event.

        Args:
            event: The stream event from Cohere.

        Returns:
            The arguments delta string, or None if no arguments are present.
        """
        return getattr(
            getattr(
                getattr(
                    getattr(getattr(event, "delta", None), "message", None),
                    "tool_calls",
                    None,
                ),
                "function",
                None,
            ),
            "arguments",
            None,
        )

    @staticmethod
    def _safe_parse_tool_arguments(
        raw_arguments: str,
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """Safely parse JSON tool arguments from a string.

        Args:
            raw_arguments: The raw JSON string containing tool arguments.

        Returns:
            A tuple of (parsed_dict, error_message) where parsed_dict is the parsed arguments
            and error_message is None on success or an error string on failure.
        """
        try:
            parsed_args = json.loads(raw_arguments or "{}")
        except json.JSONDecodeError as e:
            return {}, f"Failed to parse tool arguments: {e}"
        if not isinstance(parsed_args, dict):
            return {}, None
        return parsed_args, None

    def _tool_call_from_state(
        self, state: Dict[str, str]
    ) -> Tuple[ToolCall, Optional[str]]:
        """Construct a ToolCall object from streaming state.

        Args:
            state: Dictionary containing tool call state with 'id', 'name', and 'arguments' keys.

        Returns:
            A tuple of (ToolCall, error_message) where error_message is None on success.
        """
        parsed_args, error = self._safe_parse_tool_arguments(state.get("arguments", ""))
        return (
            ToolCall(
                id=state["id"],
                name=state["name"],
                arguments=parsed_args,
            ),
            error,
        )

    @staticmethod
    def _initialize_stream_state() -> Dict[str, Any]:
        """Initialize state dict for tracking streaming response.

        Returns:
            A dictionary with initial state for response ID, tool calls, and token counts.
        """
        return {
            "response_id": "cohere_stream",
            "in_progress_tool_calls": {},
            "completed_tool_calls": [],
            "input_tokens": 0,
            "output_tokens": 0,
        }

    def _on_message_start(
        self, event: Any, state: Dict[str, Any]
    ) -> List[ProviderChunk]:
        """Handle the message-start event from Cohere stream.

        Args:
            event: The stream event.
            state: The streaming state dictionary.

        Returns:
            List of ProviderChunk objects (empty for this event type).
        """
        if getattr(event, "id", None):
            state["response_id"] = event.id
        return []

    def _on_content_delta(
        self, event: Any, _state: Dict[str, Any]
    ) -> List[ProviderChunk]:
        """Handle the content-delta event from Cohere stream.

        Args:
            event: The stream event.
            _state: The streaming state dictionary (unused).

        Returns:
            List of ProviderChunk objects containing text content.
        """
        text = self._extract_stream_text(event)
        if not text:
            return []
        return [ProviderChunk(content=text)]

    def _on_tool_call_start(
        self, event: Any, state: Dict[str, Any]
    ) -> List[ProviderChunk]:
        """Handle the tool-call-start event from Cohere stream.

        Args:
            event: The stream event.
            state: The streaming state dictionary.

        Returns:
            List of ProviderChunk objects signaling tool call start.
        """
        index, tool_call = self._extract_stream_tool_call(event)
        if index is None or tool_call is None:
            return []
        if not tool_call.id or not tool_call.function or not tool_call.function.name:
            return []

        in_progress: Dict[int, Dict[str, str]] = state["in_progress_tool_calls"]
        in_progress[index] = {
            "id": tool_call.id,
            "name": tool_call.function.name,
            "arguments": "",
        }
        return [
            ProviderChunk(
                tool_call_start={
                    "id": tool_call.id,
                    "name": tool_call.function.name,
                    "index": index,
                }
            )
        ]

    def _on_tool_call_delta(
        self, event: Any, state: Dict[str, Any]
    ) -> List[ProviderChunk]:
        """Handle the tool-call-delta event from Cohere stream.

        Args:
            event: The stream event.
            state: The streaming state dictionary.

        Returns:
            List of ProviderChunk objects containing incremental tool argument data.
        """
        index = getattr(event, "index", None)
        in_progress: Dict[int, Dict[str, str]] = state["in_progress_tool_calls"]
        if index is None or index not in in_progress:
            return []

        args_delta = self._extract_stream_arguments_delta(event)
        if not args_delta:
            return []

        in_progress[index]["arguments"] += args_delta
        return [ProviderChunk(tool_call_delta=args_delta)]

    def _on_tool_call_end(
        self, event: Any, state: Dict[str, Any]
    ) -> List[ProviderChunk]:
        """Handle the tool-call-end event from Cohere stream.

        Args:
            event: The stream event.
            state: The streaming state dictionary.

        Returns:
            List of ProviderChunk objects signaling tool call completion.
        """
        index = getattr(event, "index", None)
        if index is None:
            return []

        in_progress: Dict[int, Dict[str, str]] = state["in_progress_tool_calls"]
        current = in_progress.pop(index, None)
        if not current:
            return []

        tool_call, error = self._tool_call_from_state(current)
        state["completed_tool_calls"].append(tool_call)

        chunks: List[ProviderChunk] = []
        if error:
            chunks.append(ProviderChunk(error=error))
        chunks.append(ProviderChunk(tool_call_done=tool_call))
        return chunks

    def _on_message_end(self, event: Any, state: Dict[str, Any]) -> List[ProviderChunk]:
        """Handle the message-end event from Cohere stream.

        Args:
            event: The stream event.
            state: The streaming state dictionary.

        Returns:
            List of ProviderChunk objects, potentially containing error information.
        """
        chunks: List[ProviderChunk] = []

        if getattr(event, "id", None):
            state["response_id"] = event.id

        delta = getattr(event, "delta", None)
        finish_reason = getattr(delta, "finish_reason", None)
        error_message = getattr(delta, "error", None)
        if finish_reason == "ERROR":
            chunks.append(
                ProviderChunk(
                    error=f"Cohere stream error: {error_message or 'finish_reason=ERROR'}"
                )
            )

        in_tokens, out_tokens = self._extract_usage_tokens(
            getattr(delta, "usage", None)
        )
        state["input_tokens"] = in_tokens
        state["output_tokens"] = out_tokens
        return chunks

    def _dispatch_stream_event(
        self, event: Any, state: Dict[str, Any]
    ) -> List[ProviderChunk]:
        """Dispatch a stream event to the appropriate handler method.

        Args:
            event: The stream event from Cohere.
            state: The streaming state dictionary.

        Returns:
            List of ProviderChunk objects from the event handler.
        """
        handlers = {
            "message-start": self._on_message_start,
            "content-delta": self._on_content_delta,
            "tool-call-start": self._on_tool_call_start,
            "tool-call-delta": self._on_tool_call_delta,
            "tool-call-end": self._on_tool_call_end,
            "message-end": self._on_message_end,
        }
        handler = handlers.get(cast(str, getattr(event, "type", None)))
        if not handler:
            return []
        return handler(event, state)

    def _flush_incomplete_tool_calls(self, state: Dict[str, Any]) -> None:
        """Flush any incomplete tool calls at the end of streaming.

        Args:
            state: The streaming state dictionary containing in-progress tool calls.
        """
        in_progress: Dict[int, Dict[str, str]] = state["in_progress_tool_calls"]
        completed: List[ToolCall] = state["completed_tool_calls"]
        for current in in_progress.values():
            tool_call, _ = self._tool_call_from_state(current)
            completed.append(tool_call)

    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[BaseTool]] = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Sends a request to the Cohere Messages API.

        Args:
            messages: A list of messages forming the conversation.
            tools: An optional list of tools available for the agent.
            **kwargs: Additional provider-specific parameters.

        Returns:
            ProviderResponse: An Allos ProviderResponse object containing the LLM's reply, tool calls, and detailed metadata.

        Raises:
            ProviderError: If the API call fails due to connection issues, authentication errors, rate limits, or other API-side errors.
        """
        cohere_messages = self._convert_messages(messages)

        config_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": cohere_messages,
            **kwargs,
        }
        if tools:
            config_kwargs["tools"] = self._convert_tools(tools)
            metadata_tools = tools
        else:
            metadata_tools = []

        builder_kwargs: Dict[str, Any] = {
            "model": self.model,
            **kwargs,
            "tools": metadata_tools,
        }

        start_time = time.time()

        try:
            response: V2ChatResponse = self.client.chat(**config_kwargs)

            content, tool_calls = self._parse_response(response.message)
            input_tokens, output_tokens = self._extract_usage_tokens(response.usage)

            metadata = self._build_metadata_from_usage(
                response_id=response.id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                tool_calls=tool_calls,
                request_kwargs=builder_kwargs,
                start_time=start_time,
            )

            return ProviderResponse(
                metadata=metadata, content=content, tool_calls=tool_calls
            )
        except self._FOUR_XX_ERRORS as e:
            self._raise_provider_error(e, type(e).__name__)
        except self._FIVE_XX_ERRORS as e:
            self._raise_provider_error(e, "Cohere API server error")
        except ApiError as e:
            self._raise_provider_error(e, "Cohere API error")

    def stream_chat(
        self,
        messages: List[Message],
        tools: Optional[List[BaseTool]] = None,
        **kwargs: Any,
    ) -> Iterator[ProviderChunk]:
        """Stream a chat request to the Cohere API.

        Args:
            messages: A list of messages forming the conversation.
            tools: An optional list of tools available for the agent.
            **kwargs: Additional provider-specific parameters.

        Yields:
            ProviderChunk: Chunks of the streaming response including text, tool calls, and metadata.

        Raises:
            ProviderError: If the API call fails due to connection issues, authentication errors, rate limits, or other API-side errors.
        """
        cohere_messages = self._convert_messages(messages)

        api_kwargs = {
            "model": self.model,
            "messages": cohere_messages,
            **kwargs,
        }

        if tools:
            api_kwargs["tools"] = self._convert_tools(tools)
            metadata_tools = tools
        else:
            metadata_tools = []

        builder_kwargs: Dict[str, Any] = {
            "model": self.model,
            **kwargs,
            "tools": metadata_tools,
        }

        start_time = time.time()
        state = self._initialize_stream_state()

        try:
            stream = self.client.chat_stream(**api_kwargs)

            for event in stream:
                for chunk in self._dispatch_stream_event(event, state):
                    yield chunk

            self._flush_incomplete_tool_calls(state)

            final_metadata = self._build_metadata_from_usage(
                response_id=state["response_id"],
                input_tokens=state["input_tokens"],
                output_tokens=state["output_tokens"],
                tool_calls=state["completed_tool_calls"],
                request_kwargs=builder_kwargs,
                start_time=start_time,
            )
            yield ProviderChunk(final_metadata=final_metadata)

        except self._FOUR_XX_ERRORS as e:
            self._raise_provider_error(e, type(e).__name__)
        except self._FIVE_XX_ERRORS as e:
            self._raise_provider_error(e, "Cohere API streaming server error")
        except ApiError as e:
            self._raise_provider_error(e, "Cohere API streaming error")

    def get_context_window(self) -> int:
        """Return the context window size for the model.

        Returns:
            The context window size in tokens, or 4096 as a default fallback.
        """
        return self._model_context_window or 4096
