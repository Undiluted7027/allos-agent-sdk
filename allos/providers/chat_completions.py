# allos/providers/chat_completions.py

"""Provides a versatile provider for any OpenAI-compatible Chat Completions API.

This module is the key to the Allos SDK's broad compatibility. The
`ChatCompletionsProvider` is not just for OpenAI's legacy API; it is the
primary mechanism for connecting to a wide range of third-party services and
local models that expose an OpenAI-compatible `/v1/chat/completions` endpoint.

This includes, but is not limited to:
 - Cloud services like Together AI, Groq, Mistral, Anyscale, and DeepSeek.
 - API gateways and routers like OpenRouter and Portkey.
 - Local models served via vLLM, TGI, or Ollama's compatibility mode.

The provider handles the translation from the Allos SDK's standard format to
the Chat Completions message and tool format (using `{"type": "function", ...}`),
making these diverse backends seamlessly available to the `Agent`.
"""

import json
import os
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

import openai
from openai.types.chat import ChatCompletion, ChatCompletionMessage
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


@provider("chat_completions")
class ChatCompletionsProvider(BaseProvider):
    """A generic provider for the OpenAI Chat Completions API (/v1/chat/completions).

    This provider uses the standard `openai` Python library to communicate with
    any API endpoint that conforms to the OpenAI Chat Completions specification.
    It is the default choice for most third-party and local model providers.

    To use with a compatible service, provide the service's `base_url` and the
    appropriate `api_key` during initialization.

    Attributes:
        client (openai.OpenAI): The authenticated OpenAI client instance, potentially
                                configured with a custom base URL.
        base_url (Optional[str]): The custom base URL the client is pointed to.
    """

    # This is a fallback variable. This provider automatically selects the env_var
    # based on --provider / -p flag in CLI and provider argument to AgentConfig.
    env_var = "OPENAI_API_KEY"

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initializes the ChatCompletionsProvider and its underlying OpenAI client.

        Args:
            model: The model identifier to use (e.g., "gpt-4", "meta-llama/Llama-3-70b-chat").
            api_key: The API key for the service. If not provided, it falls back to the
                     `OPENAI_API_KEY` environment variable.
            base_url: The API base URL for the compatible service (e.g.,
                      "https://api.together.xyz/v1").
            **kwargs: Additional keyword arguments to pass to the `openai.OpenAI` client.

        Raises:
            ProviderError: If the `openai.OpenAI` client fails to initialize.
        """
        super().__init__(model, **kwargs)
        self.base_url = base_url
        try:
            # We use the standard OpenAI client but point it to the user's desired URL
            self.client = openai.OpenAI(
                api_key=api_key or os.getenv("OPENAI_API_KEY"),
                base_url=base_url,
                **kwargs,
            )
        except Exception as e:
            raise ProviderError(
                f"Failed to initialize ChatCompletions client: {e}",
                provider="chat_completions",
            ) from e

    @staticmethod
    def _convert_messages(messages: List[Message]) -> List[Dict[str, Any]]:
        """Converts a list of Allos Messages into the Chat Completions `messages` format.

        This method translates the SDK's generic message structure into the specific
        list of dictionaries required by the `/v1/chat/completions` endpoint.

        Key transformations:
        - `ASSISTANT` messages with tool calls are formatted with a `tool_calls`
          list, where each tool call is explicitly typed as a `function`.
        - `TOOL` messages are converted into distinct messages with `role: "tool"`.

        Args:
            messages: A list of `allos.providers.base.Message` objects.

        Returns:
            A list of dictionaries formatted for the Chat Completions API.
        """
        chat_messages = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                chat_messages.append({"role": "system", "content": msg.content})

            elif msg.role == MessageRole.USER:
                chat_messages.append({"role": "user", "content": msg.content})

            elif msg.role == MessageRole.ASSISTANT:
                # Assistant messages can have text, tool_calls, or both.
                assistant_msg: Dict[str, Any] = {"role": "assistant"}

                if msg.content:
                    assistant_msg["content"] = msg.content

                if msg.tool_calls:
                    assistant_msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                chat_messages.append(assistant_msg)

            elif msg.role == MessageRole.TOOL:
                # In Chat Completions, tool results are distinct messages with role='tool'
                chat_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id,
                        "content": msg.content,
                    }
                )

        return chat_messages

    @staticmethod
    def _convert_tools(tools: List[BaseTool]) -> List[Dict[str, Any]]:
        """Converts Allos tools to the Chat Completions `tools` schema.

        This method wraps each tool's definition inside the `{"type": "function",
        "function": {...}}` structure required by the Chat Completions API.

        Args:
            tools: A list of `allos.tools.base.BaseTool` objects.

        Returns:
            A list of dictionaries formatted for the `tools` parameter of the API.
        """
        chat_tools = []
        for tool in tools:
            # Build parameters schema
            properties = {}
            required_params = []
            for param in tool.parameters:
                properties[param.name] = {
                    "type": param.type,
                    "description": param.description,
                }
                if param.required:
                    required_params.append(param.name)

            # The inner function definition
            function_def = {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required_params,
                    # Strict mode is often safer, but some 3rd party APIs might choke on
                    # 'additionalProperties: False'. We'll leave it standard for compatibility.
                },
            }

            # Wrap it in the tool object
            chat_tools.append({"type": "function", "function": function_def})
        return chat_tools

    @staticmethod
    def _parse_response(
        response: ChatCompletion,
    ) -> Tuple[Optional[str], List[ToolCall]]:
        """Parses a `ChatCompletion` response object into the standard Allos format.

        This extracts the text content and any tool call requests from the API
        response and converts them into a standard tuple of `(content, tool_calls)`.
        It gracefully handles cases where tool argument JSON is malformed by logging
        a warning and skipping the invalid tool call.

        Args:
            response: The `openai.types.chat.ChatCompletion` object from the API.

        Returns:
            A tuple containing:
            - An optional string for the message content.
            - A list of `allos.providers.base.ToolCall` objects.
        """
        choice = response.choices[0]
        message: ChatCompletionMessage = choice.message

        # --- Content ---
        content = message.content

        # --- Tool Calls ---
        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                # Check for specific tool type to satisfy type checkers and runtime safety
                if tc.type == "function":
                    try:
                        args = json.loads(tc.function.arguments)
                        tool_calls.append(
                            ToolCall(id=tc.id, name=tc.function.name, arguments=args)
                        )
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"Failed to decode arguments for tool '{tc.function.name}': {e}"
                        )
                else:
                    # Skip 'custom' or other unknown types
                    logger.debug(f"Skipping unsupported tool type: {tc.type}")

        return content, tool_calls

    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[BaseTool]] = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Sends a request to the /v1/chat/completions endpoint.

        Args:
            messages: A list of Allos Message objects representing the conversation history.
            tools: An optional list of Allos BaseTool objects for tool-calling.
            **kwargs: Additional provider-specific parameters.

        Returns:
            An Allos ProviderResponse object containing the LLM's reply, tool calls, and detailed metadata.

        Raises:
            ProviderError: If the API call fails due to connection issues, authentication errors, rate limits, or other API-side errors.
        """
        chat_messages = self._convert_messages(messages)

        api_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": chat_messages,
            **kwargs,
        }

        if tools:
            api_kwargs["tools"] = self._convert_tools(tools)
            # Store original tools for metadata
            metadata_tools = tools
        else:
            metadata_tools = []

        start_time = time.time()

        try:
            response = self.client.chat.completions.create(**api_kwargs)

            # Manually inject the original tools list into the kwargs for the metadata builder
            builder_kwargs = api_kwargs.copy()
            builder_kwargs["tools"] = metadata_tools
            builder = MetadataBuilder(
                provider_name="chat_completions",
                request_kwargs=builder_kwargs,
                start_time=start_time,
            )
            metadata = builder.with_response_obj(response).build()
            content, tool_calls = self._parse_response(response)

            return ProviderResponse(
                content=content,
                tool_calls=tool_calls,
                metadata=metadata,
            )

        except (
            openai.RateLimitError,
            openai.AuthenticationError,
            openai.PermissionDeniedError,
            openai.NotFoundError,
            openai.BadRequestError,
            openai.UnprocessableEntityError,
        ) as e:
            # Extract specific error message from response if available
            error_msg = e.message
            if hasattr(e, "body") and isinstance(e.body, dict):
                error_msg = e.body.get("message", error_msg)

            raise ProviderError(
                f"{type(e).__name__}: {error_msg}", provider="chat_completions"
            ) from e
        except openai.APIConnectionError as e:
            raise ProviderError(
                f"Connection error: {e.__cause__}", provider="chat_completions"
            ) from e
        except openai.APIError as e:
            raise ProviderError(
                f"API error: {e.message}", provider="chat_completions"
            ) from e

    def stream_chat(
        self,
        messages: List[Message],
        tools: Optional[List[BaseTool]] = None,
        **kwargs: Any,
    ) -> Iterator[ProviderChunk]:
        """Sends a request to the /v1/chat/completions endpoint and streams the response.

        Yields:
            Iterator[ProviderChunk]: An iterator of chunks representing the streaming
                                    response. Chunks can contain content, tool
                                    call information, final metadata, or an error
                                    message in case of an in-stream failure.

        Raises:
            ProviderError: If a connection or API error occurs before the stream begins.
        """
        chat_messages = self._convert_messages(messages)
        api_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": chat_messages,
            "stream": True,
            "stream_options": {
                "include_usage": True
            },  # Request usage data in the final chunk
            **kwargs,
        }

        if tools:
            api_kwargs["tools"] = self._convert_tools(tools)
            metadata_tools = tools
        else:
            metadata_tools = []

        builder_kwargs = api_kwargs.copy()
        builder_kwargs["tools"] = metadata_tools

        start_time = time.time()
        _api_context = {
            "builder_kwargs": builder_kwargs,
            "start_time": start_time,
            "tool_calls_state": [],
        }
        try:
            stream = self.client.chat.completions.create(**api_kwargs)

            for chunk in stream:
                yield from self._dispatch_chat_chunk(chunk, _api_context)

        except (openai.APIError, openai.APIConnectionError) as e:
            raise ProviderError(
                f"Streaming error: {e}", provider="chat_completions"
            ) from e

    def _dispatch_chat_chunk(self, chunk, _api_context):
        """Acts as a router for incoming stream chunks.

        It inspects the chunk's `delta` and calls the appropriate handler function
        for content, tool call deltas, or finalization. It also handles the final
        usage-only chunk.

        Args:
            chunk: The `ChatCompletionChunk` object from the stream.
            _api_context: A dictionary holding the state of the API call.
        """
        # Skip chunks without choices (e.g., final usage-only chunk)
        if not chunk.choices:
            if chunk.usage:
                yield from self._handle_final_usage_chunk(chunk, _api_context)
            return

        delta = chunk.choices[0].delta

        if delta.content:
            yield ProviderChunk(content=delta.content)

        if delta.tool_calls:
            yield from self._handle_tool_call_delta(delta.tool_calls, _api_context)

        # Only process tool calls when the reason indicates completion of that step.
        if chunk.choices[0].finish_reason == "tool_calls":
            yield from self._handle_tool_calls_completion(_api_context)

    def _handle_tool_call_delta(self, tool_call_deltas, _api_context):
        """Processes and accumulates streaming deltas for tool calls.

        This method manages the state of in-progress tool calls, creating them
        when they first appear and appending argument deltas as they arrive in
        subsequent chunks.

        Args:
            tool_call_deltas: The list of tool call deltas from a stream chunk.
            _api_context: A dictionary holding the state of the API call, including
                          the `tool_calls_state` list.
        """
        in_progress_tool_calls = _api_context["tool_calls_state"]

        for tc_delta in tool_call_deltas:
            index = tc_delta.index

            # Ensure list is long enough
            while len(in_progress_tool_calls) <= index:
                in_progress_tool_calls.append({})

            state = in_progress_tool_calls[index]

            # First chunk for a tool call, initialize it: includes id + function name
            if tc_delta.id:
                state.update(
                    {
                        "id": tc_delta.id,
                        "type": tc_delta.type,
                        "function": {
                            "name": tc_delta.function.name,
                            "arguments": "",
                        },
                    }
                )
                yield ProviderChunk(
                    tool_call_start={
                        "id": tc_delta.id,
                        "name": tc_delta.function.name,
                        "index": index,
                    }
                )

            # Subsequent chunks with agument deltas: accumulate arguments
            if tc_delta.function and tc_delta.function.arguments:
                state["function"]["arguments"] += tc_delta.function.arguments
                yield ProviderChunk(tool_call_delta=tc_delta.function.arguments)

    def _handle_tool_calls_completion(self, _api_context):
        """Finalizes tool calls at the end of a tool-calling turn.

        Triggered by a `finish_reason` of `tool_calls`, this method parses the
        fully accumulated argument strings into JSON and yields the complete
        `tool_call_done` chunks. It also clears the tool call state for the next turn.

        Args:
            _api_context: A dictionary holding the state of the API call.

        Yields:
            ProviderChunk: A `tool_call_done` chunk for each completed tool call,
                           or an `error` chunk if JSON parsing fails.
        """
        in_progress_tool_calls = _api_context["tool_calls_state"]
        for state in in_progress_tool_calls:
            try:
                parsed_args = json.loads(state["function"]["arguments"])
                tool_call = ToolCall(
                    id=state["id"],
                    name=state["function"]["name"],
                    arguments=parsed_args,
                )
                yield ProviderChunk(tool_call_done=tool_call)
            except (json.JSONDecodeError, KeyError) as e:
                yield ProviderChunk(error=f"Failed to parse tool arguments: {e}")

        # Clear the state to prepare for the next turn or the final metadata chunk.
        in_progress_tool_calls.clear()

    def _handle_final_usage_chunk(self, chunk, _api_context):
        """Handles the final usage chunk of the stream.

        This method builds the definitive metadata object for the entire stream.
        It gracefully handles potential data validation or processing errors
        during metadata construction by yielding a final ProviderChunk with an
        error field, ensuring the stream terminates cleanly.
        """
        try:
            synthetic = {
                "id": chunk.id,
                "model": chunk.model,
                "status": "completed",
                "usage": chunk.usage,
                "choices": [],
                "created": int(time.time()),
                "object": "chat.completion",
            }
            builder = MetadataBuilder(
                provider_name="chat_completions",
                request_kwargs=_api_context["builder_kwargs"],
                start_time=_api_context["start_time"],
            )

            valid_response = ChatCompletion.model_validate(synthetic)

            metadata = builder.with_response_obj(valid_response).build()
            yield ProviderChunk(final_metadata=metadata)

        except ValidationError as e:
            # This occurs if the synthetic object fails Pydantic validation
            logger.error(
                f"Pydantic validation failed while building final metadata chunk: {e}",
                exc_info=True,
            )
            yield ProviderChunk(
                error="Internal error: Failed to validate final stream metadata."
            )
        except (ValueError, TypeError, AttributeError, KeyError) as e:
            # This occurs if the MetadataBuilder fails for other reasons
            logger.error(
                f"Data processing error while building final metadata chunk: {e}",
                exc_info=True,
            )
            yield ProviderChunk(
                error="Internal error: Failed to process final stream metadata."
            )

    def get_context_window(self) -> int:
        """Returns the context window size for the configured model.

        It first checks for known OpenAI model identifiers (e.g., "gpt-4o") that
        have hardcoded context window sizes. If the model is not found in the
        known list, which is common for third-party compatible APIs, it returns a
        conservative default of 4096 tokens.

        Returns:
            An integer representing the context window size in tokens.
        """
        # Known OpenAI models map
        known_windows = {
            "gpt-4o": 128000,
            "gpt-4-turbo": 128000,
            "gpt-4": 8192,
            "gpt-3.5-turbo": 16385,
        }
        return known_windows.get(self.model, 4096)
