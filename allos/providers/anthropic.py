# allos/providers/anthropic.py

from typing import Any, Dict, List, Optional, Tuple

import anthropic
from anthropic.types import Message as AnthropicMessage
from anthropic.types import TextBlock, ToolUseBlock

from ..tools.base import BaseTool
from ..utils.errors import ProviderError
from ..utils.logging import logger
from .base import BaseProvider, Message, MessageRole, ProviderResponse, ToolCall
from .registry import provider
from .utils import _init_metadata

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
    def _parse_anthropic_response(response: AnthropicMessage) -> ProviderResponse:
        """Parses the Anthropic Message object into an Allos ProviderResponse."""
        text_accumulator: List[str] = []
        tool_calls: List[ToolCall] = []

        # --- Metadata Counters ---
        metadata = _init_metadata(len(response.content) if response.content else 0)
        if not response.content:
            return ProviderResponse(content=None, tool_calls=[], metadata=metadata)
        for block in response.content:
            if block.type == "text":
                _process_anthropic_message(block, text_accumulator, metadata)
            elif block.type == "tool_use":
                _process_anthropic_tool_use(block, tool_calls, metadata)

        # Aggregate totals
        metadata["overall"]["processed"] = (
            metadata["messages"]["processed"] + metadata["tool_calls"]["processed"]
        )
        metadata["overall"]["skipped"] = (
            metadata["messages"]["skipped"] + metadata["tool_calls"]["skipped"]
        )

        return ProviderResponse(
            content="".join(text_accumulator) or None,
            tool_calls=tool_calls,
            metadata=metadata,
        )

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

        try:
            response = self.client.messages.create(**api_kwargs)
            return self._parse_anthropic_response(response)
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

    def get_context_window(self) -> int:
        """Returns the context window size for the current model."""
        # Use a generic key to match multiple versions of a model family
        for model_family, size in MODEL_CONTEXT_WINDOWS.items():
            if model_family in self.model:
                return size
        return 4096  # Default to a 4096 for unknown Claude models


def _process_anthropic_message(
    block: TextBlock, text_accumulator: List[str], metadata: Dict
) -> None:
    """Processes a text block from the Anthropic response."""
    metadata["messages"]["total"] += 1
    if hasattr(block, "text") and block.text:
        text_accumulator.append(block.text)
        metadata["messages"]["processed"] += 1
    else:
        metadata["messages"]["skipped"] += 1


def _process_anthropic_tool_use(
    block: ToolUseBlock, tool_calls: List[ToolCall], metadata: Dict
) -> None:
    """Processes a tool_use block from the Anthropic response."""
    metadata["tool_calls"]["total"] += 1

    tool_id = getattr(block, "id", None)
    tool_name = getattr(block, "name", None)

    if not tool_id or not tool_name:
        logger.warning(
            "Skipping tool call due to missing ID or name: %s",
            tool_name or "<unknown>",
        )
        metadata["tool_calls"]["skipped"] += 1
        return

    tool_calls.append(ToolCall(id=tool_id, name=tool_name, arguments=block.input or {}))
    metadata["tool_calls"]["processed"] += 1
