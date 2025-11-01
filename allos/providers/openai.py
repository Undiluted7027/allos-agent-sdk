# allos/providers/openai.py

import json
from typing import Any, Dict, List, Optional, Tuple

import openai
from openai.types.responses import (
    Response,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
)

from ..tools.base import BaseTool
from ..utils.errors import ProviderError
from ..utils.logging import logger
from .base import BaseProvider, Message, MessageRole, ProviderResponse, ToolCall
from .registry import provider
from .utils import _init_metadata

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
                        "call_id": msg.tool_call_id,
                        "status": "completed",  # Assuming success for now
                        "output": msg.content,
                    }
                )
            else:
                openai_messages.append({"role": msg.role.value, "content": msg.content})

        return instructions, openai_messages

    @staticmethod
    def _convert_to_openai_tools(tools: List[BaseTool]) -> List[Dict[str, Any]]:
        """Converts a list of Allos BaseTools into the OpenAI function tool format."""
        openai_tools = []
        for tool in tools:
            param_schema: Dict[str, Any] = {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            }
            for param in tool.parameters:
                param_schema["properties"][param.name] = {
                    "type": param.type,
                    "description": param.description,
                }
                if param.required:
                    param_schema["required"].append(param.name)

            openai_tools.append(
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": param_schema,
                    "strict": True,
                }
            )
        return openai_tools

    @staticmethod
    def _parse_openai_response(response: Response) -> ProviderResponse:
        """Parses the OpenAI Response object into an Allos ProviderResponse."""
        text_content: list[str] = []
        tool_calls: list[ToolCall] = []

        # --- Metadata Counters ---
        metadata = _init_metadata(len(response.output) if response.output else 0)
        if not response.output:
            return ProviderResponse(content=None, tool_calls=[], metadata=metadata)

        for item in response.output:
            if item.type == "message":
                _process_message(item, text_content, metadata)
            elif item.type == "function_call":
                _process_tool_call(item, tool_calls, metadata)

        # Aggregate totals
        metadata["overall"]["processed"] = (
            metadata["messages"]["processed"] + metadata["tool_calls"]["processed"]
        )
        metadata["overall"]["skipped"] = (
            metadata["messages"]["skipped"] + metadata["tool_calls"]["skipped"]
        )

        return ProviderResponse(
            content="".join(text_content) or None,
            tool_calls=tool_calls,
            metadata=metadata,
        )

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

        api_kwargs: Dict[str, Any] = {
            "model": self.model,
            "input": input_messages,
            **kwargs,
        }
        if instructions:
            api_kwargs["instructions"] = instructions
        if tools:
            api_kwargs["tools"] = self._convert_to_openai_tools(tools)

        try:
            response = self.client.responses.create(**api_kwargs)
            return self._parse_openai_response(response)

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

    def get_context_window(self) -> int:
        """Returns the context window size for the current model."""
        return MODEL_CONTEXT_WINDOWS.get(self.model, 4096)  # Default to 4k if unknown


# --- OpenAI Specific Utility Functions ---


def _process_message(
    item: ResponseOutputMessage, text_accumulator: list[str], metadata: dict
) -> None:
    metadata["messages"]["total"] += 1
    if not getattr(item, "content", None):
        metadata["messages"]["skipped"] += 1
        return
    metadata["messages"]["processed"] += 1

    for content_part in item.content:
        if content_part.type == "output_text":
            text_accumulator.append(content_part.text)


def _process_tool_call(
    item: ResponseFunctionToolCall, tool_calls: list[ToolCall], metadata: dict
) -> None:
    metadata["tool_calls"]["total"] += 1
    id_ = getattr(item, "id", None)
    if not id_:
        logger.warning(
            "Skipping tool call due to missing ID: %s",
            getattr(item, "name", "<unknown>"),
        )
        metadata["tool_calls"]["skipped"] += 1
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

    tool_calls.append(ToolCall(id=id_, name=item.name, arguments=parsed_arguments))
    metadata["tool_calls"]["processed"] += 1
