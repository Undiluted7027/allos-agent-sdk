"""Provides the concerete implementation for interacting with a local Ollama server."""

import os
import threading
import time
from typing import Any, Dict, Iterator, List, Mapping, Optional, Set, Union

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

# Connection pool for Ollama clients
# Maps host -> Client instance to reuse connections across provider instances
_OLLAMA_CLIENT_POOL: Dict[str, Client] = {}
_POOL_LOCK = threading.Lock()

# Module-level warm-up tracker to detect first model requests
# Maps (model, host) -> is_warmed_up to track per model+host combination
_MODEL_WARMUP_TRACKER: Dict[tuple, bool] = {}
_WARMUP_LOCK = threading.Lock()
_WARMUP_THRESHOLD_SECONDS = 10.0  # Log warm-up notice if first request takes ≥10s

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
    client: Client
    _max_stream_retries = 3

    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs: Any):
        """Initializes the OllamaProvider and its client.

        Args:
            model: The name of the local model to use (e.g., 'llama3').
            api_key: Not used by this provider, but included for interface consistency.
            **kwargs: Can include 'host' to specify the Ollama server address,
                      otherwise defaults to `OLLAMA_HOST` env var or localhost.
                      Can also include 'max_stream_retries' (default: 3) to configure
                      the number of retry attempts for interrupted streams.

        Raises:
            ProviderError: If the `ollama` client fails to initialize, cannot connect
                           to the server, or the requested model is not available locally.
        """
        host = kwargs.pop("host", os.getenv("OLLAMA_HOST"))
        # Extract streaming retry configuration
        self._max_stream_retries = kwargs.pop("max_stream_retries", 3)
        super().__init__(model, **kwargs)

        # These will be populated by _verify_model_available
        self._model_context_window: Optional[int] = None
        self._model_supports_tools_capability: Optional[bool] = None
        self._host = host  # Store for logging

        try:
            # Use pooled client for connection reuse
            pool_key = host or "default"
            with _POOL_LOCK:
                if pool_key not in _OLLAMA_CLIENT_POOL:
                    logger.debug(f"Creating new Ollama client for host: {pool_key}")
                    _OLLAMA_CLIENT_POOL[pool_key] = Client(host=host)
                else:
                    logger.debug(f"Reusing pooled Ollama client for host: {pool_key}")
                self.client = _OLLAMA_CLIENT_POOL[pool_key]

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

    def _verify_model_available(self) -> None:
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

    def _extract_function_from_tool_call(
        self, tc: Any, index: int
    ) -> Optional[tuple[str, Any, str]]:
        """Extract function name, arguments, and tool_id from a tool call.

        Handles both dict-style responses and typed objects from newer ollama library versions.

        Args:
            tc: The tool call object (dict or typed object)
            index: Index of the tool call (for logging purposes)

        Returns:
            Tuple of (func_name, func_args, tool_id) if valid, None if malformed.
        """
        # Handle both dict-style and typed object responses
        if isinstance(tc, dict):
            if "function" not in tc:
                logger.warning(f"Tool call #{index} missing 'function' field, skipping")
                return None
            function = tc["function"]
            tool_id = tc.get("id", f"ollama-tool-{int(time.time() * 1000)}")
        else:
            function = getattr(tc, "function", None)
            if function is None:
                logger.warning(
                    f"Tool call #{index} missing 'function' attribute, skipping"
                )
                return None
            tool_id = (
                getattr(tc, "id", None) or f"ollama-tool-{int(time.time() * 1000)}"
            )

        # Extract name and arguments from function (dict or typed object)
        if isinstance(function, dict):
            func_name = function.get("name")
            func_args = function.get("arguments")
        else:
            func_name = getattr(function, "name", None)
            func_args = getattr(function, "arguments", None)

        # Validate required fields
        if not func_name:
            logger.warning(
                f"Tool call #{index} missing 'function.name' field, skipping"
            )
            return None

        if func_args is None:
            logger.warning(
                f"Tool call #{index} missing 'function.arguments' field, skipping"
            )
            return None

        return (func_name, func_args, tool_id)

    def _handle_warmup_tracking(
        self, is_first_request: bool, elapsed_time: float
    ) -> bool:
        """Handle warm-up tracking and logging for first requests.

        Args:
            is_first_request: Whether this is the first request to this model+host
            elapsed_time: Time elapsed for the request

        Returns:
            True if warm-up was detected (first request >= threshold), False otherwise.
        """
        if not is_first_request:
            return False

        warmup_key = (self.model, self._host or "default")
        with _WARMUP_LOCK:
            _MODEL_WARMUP_TRACKER[warmup_key] = True

        if elapsed_time >= _WARMUP_THRESHOLD_SECONDS:
            logger.info(
                f"Model '{self.model}' first request took {elapsed_time:.1f}s "
                f"(loading into memory). Subsequent requests will be faster."
            )
            return True

        return False

    def _is_first_request_for_model(self) -> bool:
        """Check if this is the first request to this model+host combination."""
        warmup_key = (self.model, self._host or "default")
        with _WARMUP_LOCK:
            return warmup_key not in _MODEL_WARMUP_TRACKER

    def _warn_if_tools_unsupported(self, tools: Optional[List[BaseTool]]) -> None:
        """Log a warning if tools are provided but model doesn't support them."""
        if tools and not self._model_supports_tools():
            logger.warning(
                f"Model '{self.model}' does not support native tool calling. "
                f"Tools will be ignored and the agent may not perform well for tasks "
                f"requiring tool use. Consider using a tool-capable model like "
                f"'llama3.1', 'qwen2', or 'qwen3:8b', or use --no-tools flag."
            )

    def _build_request_params(
        self,
        messages: List[Message],
        tools: Optional[List[BaseTool]],
        stream: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Build request parameters for Ollama API call.

        Args:
            messages: List of messages to send
            tools: Optional list of tools
            stream: Whether this is a streaming request
            **kwargs: Additional options

        Returns:
            Dictionary of request parameters for ollama.Client.chat()
        """
        ollama_messages = self._convert_to_ollama_messages(messages)
        options = {k: v for k, v in kwargs.items() if k in OLLAMA_SUPPORTED_OPTIONS}

        request_params: Dict[str, Any] = {
            "model": self.model,
            "messages": ollama_messages,
            "options": options,
        }

        if stream:
            request_params["stream"] = True

        if tools and self._model_supports_tools():
            request_params["tools"] = self._convert_tools_to_ollama_format(tools)
        else:
            self._warn_if_tools_unsupported(tools)

        return request_params

    def _parse_tool_calls(self, response_message: Dict[str, Any]) -> List[ToolCall]:
        """Parse tool calls from Ollama response message.

        Returns empty list if tool calls are malformed, with appropriate logging.
        Handles various malformed scenarios gracefully to prevent crashes.
        Supports both dict-style responses and typed objects from newer ollama library versions.
        """
        tool_calls: List[ToolCall] = []
        raw_tool_calls = response_message.get("tool_calls")

        if not raw_tool_calls:
            return tool_calls

        # Validate tool_calls is a list
        if not isinstance(raw_tool_calls, list):
            logger.warning(
                f"Ollama returned malformed tool_calls (expected list, got {type(raw_tool_calls).__name__}). "
                "Ignoring tool calls."
            )
            return tool_calls

        # Parse each tool call with defensive validation
        for i, tc in enumerate(raw_tool_calls):
            try:
                result = self._extract_function_from_tool_call(tc, i)
                if result is None:
                    continue

                func_name, func_args, tool_id = result
                tool_calls.append(
                    ToolCall(id=tool_id, name=func_name, arguments=func_args)
                )

            except Exception as e:
                logger.warning(
                    f"Unexpected error parsing tool call #{i}: {e}. Skipping."
                )
                continue

        # Log if all tool calls were malformed
        if raw_tool_calls and not tool_calls:
            logger.error(
                "Ollama returned tool_calls but all were malformed. "
                "The model may not be properly supporting tool calling."
            )

        return tool_calls

    def _build_metadata(
        self,
        response_dict: Dict[str, Any],
        start_time: float,
        warm_up_detected: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Build metadata from Ollama response.

        Args:
            response_dict: The response from Ollama API
            start_time: When the request started
            warm_up_detected: Whether this was a model warm-up (first request ≥10s)
            **kwargs: Additional keyword arguments
        """
        elapsed_time = time.time() - start_time

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

        # Add Ollama-specific metadata if warm-up was detected
        ollama_specific: Dict[str, Union[bool, float]] = {}
        if warm_up_detected:
            ollama_specific["warm_up"] = True
            ollama_specific["warm_up_duration_seconds"] = elapsed_time

        return (
            builder.with_response_obj(type("obj", (object,), synthetic_response)())
            .with_provider_specific(ollama=ollama_specific if ollama_specific else None)
            .build()
        )

    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[BaseTool]] = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Sends a synchronous request to the Ollama server."""
        request_params = self._build_request_params(messages, tools, **kwargs)
        start_time = time.time()
        is_first_request = self._is_first_request_for_model()

        try:
            response_dict = self.client.chat(**request_params)
            response_message = response_dict.get("message", {})

            # Check for model warm-up (first request taking ≥10s)
            elapsed_time = time.time() - start_time
            warm_up_detected = self._handle_warmup_tracking(
                is_first_request, elapsed_time
            )

            # Parse tool calls and metadata
            tool_calls = self._parse_tool_calls(response_message)
            metadata = self._build_metadata(
                response_dict, start_time, warm_up_detected=warm_up_detected, **kwargs
            )
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

    def _process_stream_tool_calls(
        self, tool_calls_chunk: Any
    ) -> Iterator[ProviderChunk]:
        """Process tool calls from a stream chunk.

        Args:
            tool_calls_chunk: The tool_calls data from the stream chunk

        Yields:
            ProviderChunk with tool_call_done for each valid tool call
        """
        # Validate tool_calls is a list
        if not isinstance(tool_calls_chunk, list):
            logger.warning(
                f"Stream chunk contains malformed tool_calls (expected list, got {type(tool_calls_chunk).__name__}). "
                "Ignoring tool calls."
            )
            return

        # Parse each tool call with defensive validation
        for i, tc in enumerate(tool_calls_chunk):
            try:
                result = self._extract_function_from_tool_call(tc, i)
                if result is None:
                    continue

                func_name, func_args, tool_id = result
                tool_call = ToolCall(id=tool_id, name=func_name, arguments=func_args)
                yield ProviderChunk(tool_call_done=tool_call)

            except Exception as e:
                logger.warning(
                    f"Unexpected error parsing stream tool call #{i}: {e}. Skipping."
                )
                continue

    def _process_stream_chunk(
        self,
        chunk: Dict[str, Any],
        start_time: float,
        is_first_request: bool = False,
        **kwargs: Any,
    ) -> Iterator[ProviderChunk]:
        """Process a single chunk from the Ollama stream.

        Args:
            chunk: The chunk from Ollama stream
            start_time: When the request started
            is_first_request: Whether this is the first request to this model+host
            **kwargs: Additional keyword arguments
        """
        # Process content delta
        content_delta = chunk.get("message", {}).get("content")
        if content_delta:
            yield ProviderChunk(content=content_delta)

        # Process tool calls (Ollama streams them fully formed)
        tool_calls_chunk = chunk.get("message", {}).get("tool_calls")
        if tool_calls_chunk:
            yield from self._process_stream_tool_calls(tool_calls_chunk)

        # Handle final metadata
        if chunk.get("done"):
            elapsed_time = time.time() - start_time
            warm_up_detected = self._handle_warmup_tracking(
                is_first_request, elapsed_time
            )
            metadata = self._build_metadata(
                chunk, start_time, warm_up_detected=warm_up_detected, **kwargs
            )
            yield ProviderChunk(final_metadata=metadata)

    def stream_chat(
        self,
        messages: List[Message],
        tools: Optional[List[BaseTool]] = None,
        **kwargs: Any,
    ) -> Iterator[ProviderChunk]:
        """Sends a streaming request to the Ollama server with retry on interruption.

        This method includes automatic retry logic with exponential backoff for handling
        connection interruptions. If a stream is interrupted, it will retry up to
        max_stream_retries times (default: 3) before yielding an error chunk.

        Args:
            messages: A list of Message objects representing the conversation history.
            tools: An optional list of tools available for the agent.
            **kwargs: Additional provider-specific parameters for the API call.

        Yields:
            ProviderChunk: An iterator of chunks representing the streaming response.
                          May include an error chunk if all retry attempts fail.
        """
        request_params = self._build_request_params(
            messages, tools, stream=True, **kwargs
        )

        start_time = time.time()
        retry_delay = 1.0  # Initial retry delay in seconds
        is_first_request = self._is_first_request_for_model()

        # Retry loop with exponential backoff
        for attempt in range(self._max_stream_retries):
            try:
                stream = self.client.chat(**request_params)
                chunks_received = 0

                for chunk in stream:
                    chunks_received += 1
                    # Pass warm-up detection info to chunk processor
                    yield from self._process_stream_chunk(
                        chunk, start_time, is_first_request=is_first_request, **kwargs
                    )

                # Stream completed successfully
                logger.debug(
                    f"Stream completed successfully ({chunks_received} chunks received)"
                )
                return

            except ollama.ResponseError as e:
                error_msg = "Ollama API error"
                if hasattr(e, "error") and e.error:
                    error_msg += f": {e.error}"

                # Don't retry on API errors (bad request, etc.)
                logger.error(f"{error_msg} (no retry)")
                yield ProviderChunk(error=error_msg)
                return

            except ollama.RequestError as e:
                error_msg = "Ollama connection error"
                if str(e):
                    error_msg += f": {str(e)}"

                # This is a connection/network error - retry is appropriate
                if attempt < self._max_stream_retries - 1:
                    logger.warning(
                        f"{error_msg} - Retrying ({attempt + 1}/{self._max_stream_retries}) "
                        f"after {retry_delay:.1f}s..."
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    # Final attempt failed
                    logger.error(
                        f"{error_msg} - All {self._max_stream_retries} retry attempts failed"
                    )
                    yield ProviderChunk(
                        error=f"{error_msg} (retried {self._max_stream_retries} times)"
                    )
                    return

            except Exception as e:
                # Unexpected error - don't retry
                error_msg = f"Unexpected error during Ollama streaming: {str(e)}"
                logger.error(f"{error_msg} (no retry)")
                yield ProviderChunk(error=error_msg)
                return

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
