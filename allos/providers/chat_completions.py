# allos/providers/chat_completions.py

import json
import os
from typing import Any, Dict, List, Optional

import openai
from openai.types.chat import ChatCompletion, ChatCompletionMessage

from ..tools.base import BaseTool
from ..utils.errors import ProviderError
from ..utils.logging import logger
from .base import BaseProvider, Message, MessageRole, ProviderResponse, ToolCall
from .registry import provider
from .utils import _init_metadata


@provider("chat_completions")
class ChatCompletionsProvider(BaseProvider):
    """
    A generic provider for the OpenAI Chat Completions API (/v1/chat/completions).

    This provider is designed for:
    1. Legacy OpenAI workflows.
    2. OpenAI-Compatible APIs (Together AI, Anyscale, vLLM, LocalAI, etc.) that
       do not yet support the newer Responses API.

    To use with a 3rd party service, simply provide the `base_url` and the appropriate `api_key`.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Args:
            model: The model identifier (e.g., "gpt-4", "meta-llama/Llama-3-70b-chat").
            api_key: The API key. Defaults to OPENAI_API_KEY env var if not set.
            base_url: The API base URL (e.g., "https://api.together.xyz/v1").
            **kwargs: Additional arguments for the OpenAI client.
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
        """
        Converts Allos messages to the Chat Completions `messages` array format.
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
        """
        Converts Allos tools to the Chat Completions `tools` schema.
        Note: This API uses the "external tagging" format (wrapped in "function").
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
    def _parse_response(response: ChatCompletion) -> ProviderResponse:
        """
        Parses the ChatCompletion response object into an Allos ProviderResponse.
        """
        choice = response.choices[0]
        message: ChatCompletionMessage = choice.message

        # --- Metadata ---
        # Chat Completions API doesn't give itemized breakdown like Responses API,
        # so we infer simple counts.
        metadata: Dict[str, Any] = _init_metadata(1)  # 1 choice processed
        metadata["overall"]["processed"] = 1
        metadata["response_id"] = response.id
        if response.usage:
            metadata["usage"] = response.usage.model_dump()

        # --- Content ---
        content = message.content

        # --- Tool Calls ---
        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                metadata["tool_calls"]["total"] += 1

                # Check for specific tool type to satisfy type checkers and runtime safety
                if tc.type == "function":
                    try:
                        args = json.loads(tc.function.arguments)
                        tool_calls.append(
                            ToolCall(id=tc.id, name=tc.function.name, arguments=args)
                        )
                        metadata["tool_calls"]["processed"] += 1
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"Failed to decode arguments for tool '{tc.function.name}': {e}"
                        )
                        metadata["tool_calls"]["skipped"] += 1
                else:
                    # Skip 'custom' or other unknown types
                    logger.debug(f"Skipping unsupported tool type: {tc.type}")
                    metadata["tool_calls"]["skipped"] += 1

        return ProviderResponse(
            content=content, tool_calls=tool_calls, metadata=metadata
        )

    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[BaseTool]] = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """
        Sends a request to the /v1/chat/completions endpoint.
        """
        chat_messages = self._convert_messages(messages)

        api_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": chat_messages,
            **kwargs,
        }

        if tools:
            api_kwargs["tools"] = self._convert_tools(tools)

        try:
            response = self.client.chat.completions.create(**api_kwargs)
            return self._parse_response(response)

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

    def get_context_window(self) -> int:
        """
        Returns context window. Since this class supports arbitrary models via base_url,
        we default to a safe 4k but check for known OpenAI models.
        """
        # Known OpenAI models map
        known_windows = {
            "gpt-4o": 128000,
            "gpt-4-turbo": 128000,
            "gpt-4": 8192,
            "gpt-3.5-turbo": 16385,
        }
        return known_windows.get(self.model, 4096)
