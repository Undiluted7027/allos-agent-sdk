# allos/providers/anthropic.py

import json
import time
from typing import Any, Dict, List, Optional, Tuple

import anthropic
from anthropic.types import Message as AnthropicMessage
from anthropic.types import TextBlock, ToolUseBlock

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

# A mapping of known Anthropic models to their context window sizes (in tokens)
MODEL_CONTEXT_WINDOWS = {
    "claude-3-opus-20240229": 200000,
    "claude-3-sonnet-20240229": 200000,
    "claude-3-haiku-20240307": 200000,
    "claude-3-5-sonnet-20240620": 200000,
    "claude-3-7-sonnet-latest": 200000,
}


@provider("anthropic")
class AnthropicProvider(BaseProvider):
    """
    An Allos provider for interacting with the Anthropic Messages API.
    """

    env_var = "ANTHROPIC_API_KEY"

    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs: Any):
        super().__init__(model, **kwargs)
        try:
            self.client = anthropic.Anthropic(api_key=api_key, **kwargs)
        except Exception as e:
            raise ProviderError(
                f"Failed to initialize Anthropic client: {e}", provider="anthropic"
            ) from e

    @staticmethod
    def _convert_to_anthropic_messages(
        messages: List[Message],
    ) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        """
        Converts a list of Allos Messages into the format expected by the
        Anthropic Messages API, separating the system prompt.
        """
        system_prompt = None
        anthropic_messages: List[Dict[str, Any]] = []

        if messages and messages[0].role == MessageRole.SYSTEM:
            system_prompt = messages[0].content
            messages = messages[1:]

        for msg in messages:
            if msg.role == MessageRole.USER:
                anthropic_messages.append(
                    {"role": "user", "content": msg.content or ""}
                )
            elif msg.role == MessageRole.ASSISTANT:
                # Assistant messages can contain multiple content blocks (text and tool_use)
                content_blocks: list[Dict[str, Any]] = []
                if msg.content:
                    content_blocks.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        }
                    )
                anthropic_messages.append(
                    {"role": "assistant", "content": content_blocks}
                )
            elif msg.role == MessageRole.TOOL:
                anthropic_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_call_id,
                                "content": msg.content,
                            }
                        ],
                    }
                )
        return system_prompt, anthropic_messages

    @staticmethod
    def _convert_to_anthropic_tools(tools: List[BaseTool]) -> List[Dict[str, Any]]:
        """Converts a list of Allos BaseTools into the Anthropic tool format."""
        anthropic_tools = []
        for tool in tools:
            param_schema: Dict[str, Any] = {
                "type": "object",
                "properties": {},
                "required": [],
            }
            for param in tool.parameters:
                param_schema["properties"][param.name] = {
                    "type": param.type,
                    "description": param.description,
                }
                if param.required:
                    param_schema["required"].append(param.name)

            anthropic_tools.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": param_schema,
                }
            )
        return anthropic_tools

    @staticmethod
    def _parse_anthropic_response(
        response: AnthropicMessage,
    ) -> Tuple[Optional[str], List[ToolCall]]:
        """Parses the Anthropic Message object into an Allos ProviderResponse."""
        text_accumulator: List[str] = []
        tool_calls: List[ToolCall] = []

        if not response.content:
            return None, []
        for block in response.content:
            if block.type == "text":
                _process_anthropic_message(block, text_accumulator)
            elif block.type == "tool_use":
                _process_anthropic_tool_use(block, tool_calls)

        return "".join(text_accumulator) or None, tool_calls

    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[BaseTool]] = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Sends a request to the Anthropic Messages API."""
        system_prompt, anthropic_messages = self._convert_to_anthropic_messages(
            messages
        )

        api_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": anthropic_messages,
            # Anthropic requires max_tokens
            "max_tokens": kwargs.pop("max_tokens", 4096),
            **kwargs,
        }
        if system_prompt:
            api_kwargs["system"] = system_prompt
        if tools:
            api_kwargs["tools"] = self._convert_to_anthropic_tools(tools)
            metadata_tools = tools
        else:
            metadata_tools = []

        builder_kwargs = api_kwargs.copy()
        builder_kwargs["tools"] = metadata_tools

        start_time = time.time()

        try:
            response = self.client.messages.create(**api_kwargs)

            # --- METADATA GENERATION ---
            # Create a synthetic response object for the builder
            synthetic_response = {
                "id": response.id,
                "model": response.model,
                "status": "completed",
                "usage": response.usage,
                "output": response.content,  # Pass content for tool call parsing
            }
            builder = MetadataBuilder(
                provider_name="anthropic",
                request_kwargs=builder_kwargs,
                start_time=start_time,
            )
            metadata = builder.with_response_obj(
                type("obj", (object,), synthetic_response)()
            ).build()

            content, tool_calls = self._parse_anthropic_response(response)

            return ProviderResponse(
                content=content, tool_calls=tool_calls, metadata=metadata
            )
        except anthropic.APIConnectionError as e:
            raise ProviderError(
                f"Connection error: {e.__cause__}", provider="anthropic"
            ) from e
        except anthropic.RateLimitError as e:
            raise ProviderError("Rate limit exceeded", provider="anthropic") from e
        except anthropic.AuthenticationError as e:
            raise ProviderError("Authentication error", provider="anthropic") from e
        except anthropic.BadRequestError as e:
            error_message = e.message
            # Safely check if the body is a dictionary and extract a more specific message
            if isinstance(e.body, dict):
                error_details = e.body.get("error", {})
                if isinstance(error_details, dict):
                    error_message = error_details.get("message", e.message)

            raise ProviderError(
                f"Bad request: {error_message}", provider="anthropic"
            ) from e
        except anthropic.APIStatusError as e:
            raise ProviderError(
                f"Anthropic API error ({e.status_code}): {e.message}",
                provider="anthropic",
            ) from e

    def stream_chat(self, messages, tools=None, **kwargs):
        api_kwargs, _ = self._build_api_kwargs(messages, tools, kwargs)
        builder_kwargs = self._build_builder_kwargs(api_kwargs, tools)
        state = self._init_stream_state()
        start_time = time.time()

        try:
            with self.client.messages.stream(**api_kwargs) as stream:
                for event in stream:
                    result = self._handle_event(event, state)
                    if result == "STOP":
                        yield self._finalize_stream(state, builder_kwargs, start_time)
                    elif result:
                        yield result

        except anthropic.APIError as e:
            raise ProviderError(
                f"Anthropic API streaming error: {e}", provider="anthropic"
            ) from e

    def get_context_window(self) -> int:
        """Returns the context window size for the current model."""
        # Use a generic key to match multiple versions of a model family
        for model_family, size in MODEL_CONTEXT_WINDOWS.items():
            if model_family in self.model:
                return size
        return 4096  # Default to a 4096 for unknown Claude models

    def _build_api_kwargs(self, messages, tools, kwargs):
        system_prompt, anthropic_messages = self._convert_to_anthropic_messages(
            messages
        )
        api_kwargs = {
            "model": self.model,
            "messages": anthropic_messages,
            "max_tokens": kwargs.pop("max_tokens", 4096),
            # "stream": True,
            **kwargs,
        }
        if system_prompt:
            api_kwargs["system"] = system_prompt

        if tools:
            api_kwargs["tools"] = self._convert_to_anthropic_tools(tools)

        return api_kwargs, system_prompt

    def _build_builder_kwargs(self, api_kwargs, tools):
        builder_kwargs = api_kwargs.copy()
        builder_kwargs["tools"] = tools or []
        return builder_kwargs

    def _init_stream_state(self):
        return {
            "in_progress_tool_calls": {},
            "final_usage": {},
            "response_id": "",
            "model_id": "",
        }

    def _handle_event(self, event, state):
        if event.type == "message_start":
            state["response_id"] = event.message.id
            state["model_id"] = event.message.model
            state["final_usage"]["input_tokens"] = event.message.usage.input_tokens

        elif (
            event.type == "content_block_start"
            and event.content_block.type == "tool_use"
        ):
            state["in_progress_tool_calls"][event.index] = {
                "id": event.content_block.id,
                "name": event.content_block.name,
                "arguments": "",
            }

        elif event.type == "content_block_delta":
            return self._handle_block_delta(event, state)

        elif event.type == "content_block_stop":
            return self._handle_block_stop(event, state)

        elif event.type == "message_delta":
            state["final_usage"]["output_tokens"] = event.usage.output_tokens

        elif event.type == "message_stop":
            return "STOP"

        return None

    def _handle_block_delta(self, event, state):
        if event.delta.type == "text_delta":
            return ProviderChunk(content=event.delta.text)

        if event.delta.type == "input_json_delta":
            call = state["in_progress_tool_calls"].get(event.index)
        if call:
            call["arguments"] += event.delta.partial_json

    def _handle_block_stop(self, event, state):
        call = state["in_progress_tool_calls"].get(event.index)
        if not call:
            return None

        try:
            parsed = json.loads(call["arguments"])
            tool_call = ToolCall(id=call["id"], name=call["name"], arguments=parsed)
            return ProviderChunk(tool_call_done=tool_call)
        except Exception as e:
            return ProviderChunk(error=f"Failed to parse tool arguments: {e}")

    def _finalize_stream(self, state, builder_kwargs, start_time):
        # Create a usage object with proper attributes
        usage_obj = type("Usage", (), state["final_usage"])()

        synthetic_response = {
            "id": state["response_id"],
            "model": state["model_id"],
            "status": "completed",
            "usage": usage_obj,
        }

        builder = MetadataBuilder(
            provider_name="anthropic",
            request_kwargs=builder_kwargs,
            start_time=start_time,
        )

        metadata = builder.with_response_obj(
            type("obj", (object,), synthetic_response)()
        ).build()
        return ProviderChunk(final_metadata=metadata)


def _process_anthropic_message(block: TextBlock, text_accumulator: List[str]) -> None:
    """Processes a text block from the Anthropic response."""
    if hasattr(block, "text") and block.text:
        text_accumulator.append(block.text)


def _process_anthropic_tool_use(
    block: ToolUseBlock, tool_calls: List[ToolCall]
) -> None:
    """Processes a tool_use block from the Anthropic response."""

    tool_id = getattr(block, "id", None)
    tool_name = getattr(block, "name", None)

    if not tool_id or not tool_name:
        logger.warning(
            "Skipping tool call due to missing ID or name: %s",
            tool_name or "<unknown>",
        )
        return

    tool_calls.append(ToolCall(id=tool_id, name=tool_name, arguments=block.input or {}))
