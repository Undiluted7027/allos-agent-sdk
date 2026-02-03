"""Provides the concerete implementation for interacting with a local Ollama server."""

import os
import time
from typing import Any, Dict, Iterator, List, Mapping, Optional, Set

import ollama
from ollama import Client, ShowResponse

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

# A mapping of known Ollama models to their context window sizes (in tokens)
# These serve as fallbacks when actual context window cannot be retrieved from ollama.show()
OLLAMA_CONTEXT_WINDOWS = {
    # Llama family
    "llama3": 8192,
    "llama3.1": 128000,
    "llama3.2": 131072,
    "codellama": 16384,
    # Mistral family
    "mistral": 32768,
    "mixtral": 32768,
    # Qwen family
    "qwen2": 32768,
    "qwen2.5": 32768,
    "qwen2.5-coder": 32768,
    "qwen3": 40960,
    "qwen3:8b": 40960,
    # Google family
    "gemma": 8192,
    "gemma2": 8192,
    # DeepSeek family
    "deepseek-coder": 16384,
    "deepseek-coder-v2": 32768,
}

# Set of models known to support native tool calling in Ollama
# These serve as fallbacks when actual capability cannot be retrieved from ollama.show()
# Models should be added here when they are confirmed to support function calling
OLLAMA_TOOL_SUPPORTED_MODELS: Set[str] = {
    # Llama family (3.1+)
    "llama3.1",
    "llama3.2",
    # Mistral family
    "mistral",
    "mixtral",
    # Qwen family
    "qwen2",
    "qwen2.5",
    "qwen2.5-coder",
    "qwen3",
    "qwen3:8b",
    # Google family
    "gemma2",
}

# Parameters from the Ollama library that can be passed through kwargs
OLLAMA_SUPPORTED_OPTIONS: Set[str] = {
    "mirostat",
    "mirostat_eta",
    "mirostat_tau",
    "num_ctx",
    "repeat_last_n",
    "repeat_penalty",
    "temperature",
    "seed",
    "stop",
    "tfs_z",
    "num_predict",
    "top_k",
    "top_p",
}


@provider("ollama")
class OllamaProvider(BaseProvider):
    """An Allos provider for local models via Ollama's native Python library.

    This provider connects directly to a running Ollama server, enabling completely
    local and private agent execution. It uses the `ollama` package for optimal
    performance and feature support.

    Authentication is not required, but the Ollama server must be running and
    the specified model must be pulled locally.

    Attributes:
        client (ollama.Client): The authenticated Ollama client instance.
    """

    env_var = "OLLAMA_HOST"

    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs: Any):
        """Initializes the OllamaProvider and its client.

        Args:
            model: The name of the local model to use (e.g., 'llama3').
            api_key: Not used by this provider, but included for interface consistency.
            **kwargs: Can include 'host' to specify the Ollama server address,
                      otherwise defaults to `OLLAMA_HOST` env var or localhost.

        Raises:
            ProviderError: If the `ollama` client fails to initialize, cannot connect
                           to the server, or the requested model is not available locally.
        """
        host = kwargs.pop("host", os.getenv("OLLAMA_HOST"))
        super().__init__(model, **kwargs)

        # These will be populated by _verify_model_available
        self._model_context_window: Optional[int] = None
        self._model_supports_tools_capability: Optional[bool] = None

        try:
            self.client = Client(host=host)
            self._verify_model_available()
            logger.debug(
                f"Ollama provider initialized for model '{model}' at {host or 'default host'}"
            )
        except ollama.ResponseError as e:
            raise ProviderError(
                f"Failed to connect to Ollama server: {e.error}", provider="ollama"
            ) from e
        except Exception as e:
            raise ProviderError(
                f"Failed to initialize Ollama client: {e}", provider="ollama"
            ) from e

    def _verify_model_available(self):
        """Checks if the configured model is available and retrieves its capabilities.

        This method verifies that:
        1. The model is pulled locally on the Ollama server
        2. Retrieves the model's actual context window size (if available)
        3. Checks if the model natively supports tool calling

        The retrieved capabilities are stored on the instance for later use.
        """
        try:
            # First, check if model is in the list of available models
            local_models = self.client.list().models
            available_model_names = {m.model for m in local_models}
            if self.model not in available_model_names:
                raise ProviderError(
                    f"Model '{self.model}' not available locally. "
                    f"Please run `ollama pull {self.model}`.",
                    provider="ollama",
                )

            # Get detailed model info using ollama.show()
            try:
                model_info: ShowResponse = self.client.show(self.model)

                # Extract context window from model info
                if model_info.modelinfo:
                    self._model_context_window = retrieve_context_length(
                        model_info.modelinfo
                    )
                    if self._model_context_window:
                        logger.debug(
                            f"Model '{self.model}' context window: "
                            f"{self._model_context_window} tokens"
                        )

                # Check tool calling capability
                self._model_supports_tools_capability = check_tools_capability(
                    model_info
                )
                if self._model_supports_tools_capability:
                    logger.debug(f"Model '{self.model}' supports native tool calling")
                else:
                    logger.debug(
                        f"Model '{self.model}' does not support native tool calling"
                    )

            except Exception as e:
                # Non-fatal: we can still use the model, just without capability info
                logger.debug(f"Could not retrieve model capabilities: {e}")

        except ollama.RequestError as e:
            raise ProviderError(
                "Could not connect to Ollama server to verify models. Is it running?",
                provider="ollama",
            ) from e

    @staticmethod
    def _convert_to_ollama_messages(messages: List[Message]) -> List[Dict[str, Any]]:
        """Converts a list of Allos Messages into the Ollama `messages` format."""
        ollama_messages = []
        for msg in messages:
            # Only handle system, user, assistant roles
            if msg.role in [
                MessageRole.SYSTEM,
                MessageRole.USER,
                MessageRole.ASSISTANT,
            ]:
                message_dict: Dict[str, Any] = {
                    "role": msg.role.value,
                    "content": msg.content or "",
                }
                if msg.tool_calls:
                    # Assistant message requesting tool call
                    message_dict["tool_calls"] = [
                        {"function": {"name": tc.name, "arguments": tc.arguments}}
                        for tc in msg.tool_calls
                    ]
                ollama_messages.append(message_dict)
            elif msg.role == MessageRole.TOOL:
                # A tool result message
                ollama_messages.append(
                    {
                        "role": "tool",
                        "content": msg.content,
                        "tool_call_id": msg.tool_call_id,
                    }
                )
        return ollama_messages

    @staticmethod
    def _convert_tools_to_ollama_format(tools: List[BaseTool]) -> List[Dict[str, Any]]:
        """Converts Allos BaseTools to Ollama's native tool format."""
        ollama_tools = []
        for tool in tools:
            openai_format = tool.to_provider_format("ollama")
            ollama_tools.append(openai_format)
        return ollama_tools

    def _extract_model_family(self) -> str:
        """Extract base model family, handling special cases and version tags."""
        # Remove common version tags, but keep special identifiers like ":8b"
        if ":" in self.model:
            base, tag = self.model.split(":", 1)
            # Keep the tag if it's a special size identifier that's in our mappings
            if (
                self.model in OLLAMA_TOOL_SUPPORTED_MODELS
                or self.model in OLLAMA_CONTEXT_WINDOWS
            ):
                return self.model
            # Strip "latest" and other version tags
            if tag in ("latest", "instruct", "chat", "custom"):
                return base
            # For size tags like "7b", "13b", "70b", keep just the base
            if tag.replace("b", "").replace(".", "").isdigit():
                return base
        return self.model

    def _model_supports_tools(self) -> bool:
        """Check if the current model supports tool calling.

        First checks the actual capability reported by the Ollama server (most accurate),
        then falls back to checking against our known list of tool-supporting models.

        Returns:
            True if the model supports native tool calling, False otherwise.
        """
        # Use the actual capability from ollama.show() if available
        if self._model_supports_tools_capability is not None:
            return self._model_supports_tools_capability

        # Fallback to checking against known model families
        model_family = self._extract_model_family()
        return any(
            model_family == family or model_family.startswith(f"{family}.")
            for family in OLLAMA_TOOL_SUPPORTED_MODELS
        )

    def _parse_tool_calls(self, response_message: Dict[str, Any]) -> List[ToolCall]:
        """Parse tool calls from Ollama response message."""
        tool_calls: List[ToolCall] = []
        if response_message.get("tool_calls"):
            for tc in response_message["tool_calls"]:
                tool_calls.append(
                    ToolCall(
                        id=tc.get("id", f"ollama-tool-{int(time.time() * 1000)}"),
                        name=tc["function"]["name"],
                        arguments=tc["function"]["arguments"],
                    )
                )
        return tool_calls

    def _build_metadata(
        self, response_dict: Dict[str, Any], start_time: float, **kwargs: Any
    ) -> Any:
        """Build metadata from Ollama response."""
        synthetic_usage = {
            "input_tokens": response_dict.get("prompt_eval_count", 0),
            "output_tokens": response_dict.get("eval_count", 0),
        }
        synthetic_response = {
            "model": response_dict.get("model", self.model),
            "usage": type("Usage", (), synthetic_usage)(),
        }

        builder = MetadataBuilder(
            provider_name="ollama",
            request_kwargs=kwargs,
            start_time=start_time,
        )
        return builder.with_response_obj(
            type("obj", (object,), synthetic_response)()
        ).build()

    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[BaseTool]] = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Sends a synchronous request to the Ollama server."""
        ollama_messages = self._convert_to_ollama_messages(messages)

        # Filter kwargs to only include valid Ollama options
        options = {k: v for k, v in kwargs.items() if k in OLLAMA_SUPPORTED_OPTIONS}

        # Prepare request for the client
        request_params: Dict[str, Any] = {
            "model": self.model,
            "messages": ollama_messages,
            "options": options,
        }

        if tools and self._model_supports_tools():
            request_params["tools"] = self._convert_tools_to_ollama_format(tools)
        elif tools:
            logger.warning(
                f"Model '{self.model}' does not support native tool calling. "
                f"Tools will be ignored and the agent may not perform well for tasks "
                f"requiring tool use. Consider using a tool-capable model like "
                f"'llama3.1', 'qwen2', or 'qwen3:8b', or use --no-tools flag."
            )

        start_time = time.time()
        try:
            response_dict = self.client.chat(**request_params)
            response_message = response_dict.get("message", {})

            # Parse tool calls and metadata
            tool_calls = self._parse_tool_calls(response_message)
            metadata = self._build_metadata(response_dict, start_time, **kwargs)
            content = response_message.get("content")

            return ProviderResponse(
                content=content,
                tool_calls=tool_calls,
                metadata=metadata,
            )
        except ollama.ResponseError as e:
            # Server responded with an error (4xx, 5xx status codes)
            error_msg = "Ollama API error"
            if hasattr(e, "status_code") and e.status_code:
                error_msg += f" (status {e.status_code})"
            if hasattr(e, "error") and e.error:
                error_msg += f": {e.error}"
            raise ProviderError(error_msg, provider="ollama") from e
        except ollama.RequestError as e:
            # Connection errors, timeouts, etc.
            error_msg = "Ollama connection error"
            if str(e):
                error_msg += f": {str(e)}"
            raise ProviderError(error_msg, provider="ollama") from e
        except Exception as e:
            # Catch any unexpected errors
            raise ProviderError(
                f"Unexpected error during Ollama chat: {str(e)}", provider="ollama"
            ) from e

    def _process_stream_chunk(
        self, chunk: Dict[str, Any], start_time: float, **kwargs: Any
    ) -> Iterator[ProviderChunk]:
        """Process a single chunk from the Ollama stream."""
        # Process content delta
        content_delta = chunk.get("message", {}).get("content")
        if content_delta:
            yield ProviderChunk(content=content_delta)

        # Process tool calls (Ollama streams them fully formed)
        tool_calls_chunk = chunk.get("message", {}).get("tool_calls")
        if tool_calls_chunk:
            for tc in tool_calls_chunk:
                tool_call = ToolCall(
                    id=tc.get("id", f"ollama-tool-{int(time.time() * 1000)}"),
                    name=tc["function"]["name"],
                    arguments=tc["function"]["arguments"],
                )
                yield ProviderChunk(tool_call_done=tool_call)

        # Handle final metadata
        if chunk.get("done"):
            metadata = self._build_metadata(chunk, start_time, **kwargs)
            yield ProviderChunk(final_metadata=metadata)

    def stream_chat(
        self,
        messages: List[Message],
        tools: Optional[List[BaseTool]] = None,
        **kwargs: Any,
    ) -> Iterator[ProviderChunk]:
        """Sends a streaming request to the Ollama server."""
        ollama_messages = self._convert_to_ollama_messages(messages)
        options = {k: v for k, v in kwargs.items() if k in OLLAMA_SUPPORTED_OPTIONS}

        request_params: Dict[str, Any] = {
            "model": self.model,
            "messages": ollama_messages,
            "options": options,
            "stream": True,
        }

        if tools and self._model_supports_tools():
            request_params["tools"] = self._convert_tools_to_ollama_format(tools)
        elif tools:
            logger.warning(
                f"Model '{self.model}' does not support native tool calling. "
                f"Tools will be ignored and the agent may not perform well for tasks "
                f"requiring tool use. Consider using a tool-capable model like "
                f"'llama3.1', 'qwen2', or 'qwen3:8b', or use --no-tools flag."
            )

        start_time = time.time()
        try:
            stream = self.client.chat(**request_params)
            for chunk in stream:
                yield from self._process_stream_chunk(chunk, start_time, **kwargs)

        except ollama.ResponseError as e:
            error_msg = "Ollama API error"
            if hasattr(e, "error") and e.error:
                error_msg += f": {e.error}"
            yield ProviderChunk(error=error_msg)
        except ollama.RequestError as e:
            error_msg = "Ollama connection error"
            if str(e):
                error_msg += f": {str(e)}"
            yield ProviderChunk(error=error_msg)

    def get_context_window(self) -> int:
        """Returns the context window size for the current model.

        First checks the actual context window retrieved from ollama.show() (most accurate),
        then falls back to our known mappings, and finally to a conservative default.

        Returns:
            The context window size in tokens.
        """
        # Use the actual context window from ollama.show() if available
        if self._model_context_window is not None:
            return self._model_context_window

        # Fallback to known mappings
        model_family = self._extract_model_family()

        # Try exact match first
        if model_family in OLLAMA_CONTEXT_WINDOWS:
            return OLLAMA_CONTEXT_WINDOWS[model_family]

        # Try prefix match (e.g., "llama3.1" matches "llama3.1-custom")
        for family, size in OLLAMA_CONTEXT_WINDOWS.items():
            if model_family.startswith(family):
                return size

        logger.warning(
            f"Unknown context window for '{self.model}'. Falling back to 4096."
        )
        return 4096


def retrieve_context_length(model_info: Mapping[str, Any]) -> Optional[int]:
    """Extract context length from model info using known paths."""
    # Try common paths in order of likelihood
    paths = [
        lambda d: d.get("num_ctx"),
        lambda d: d.get("details", {}).get("num_ctx"),
        lambda d: d.get("parameters", {}).get("num_ctx"),
        lambda d: next((v for k, v in d.items() if k.endswith("context_length")), None),
    ]

    for path_fn in paths:
        try:
            result: int = path_fn(model_info)
            if result is not None:
                return result
        except (AttributeError, TypeError):
            continue
    return None


def check_tools_capability(model: ShowResponse) -> bool:
    """Checks if the model supports tools.

    Args:
        model (ShowResponse): The model to check for tool support.

    Returns:
        bool: True if the model supports tools, False otherwise.
    """
    if model.capabilities and "tools" in model.capabilities:
        return True
    return False
