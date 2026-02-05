# allos/providers/google.py

import sys
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from allos.providers.base import (
    BaseProvider,
    Message,
    MessageRole,
    ProviderChunk,
    ProviderResponse,
    ToolCall,
)
from allos.providers.metadata import MetadataBuilder
from allos.tools.base import BaseTool
from allos.utils.errors import ProviderError

from ..utils.logging import logger
from .registry import provider

if sys.version_info < (3, 10):
    raise ImportError(
        "Google provider requires version Python 3.10 or higher. "
        f"Current version: {sys.version_info.major}.{sys.version_info.minor}"
    )

# Only runs on Python 3.10
MODEL_CONTEXT_WINDOWS = {
    "gemini-2.5-flash": 1048576,
    "gemini-2.5-pro": 1048576,
    "gemini-2.0-flash": 1048576,
    "gemini-1.5-flash": 1048576,
    "gemini-1.5-pro": 2097152,
    "gemini-1.0-pro": 32768,
}


@provider("google")
class GoogleProvider(BaseProvider):
    env_var = "GOOGLE_API_KEY"

    @classmethod
    def check_env_config(cls) -> Tuple[bool, str]:
        """Check for Gemini API key or Vertex AI configuration."""
        import os

        # Check Gemini API Keys
        if os.environ.get("GOOGLE_API_KEY"):
            return (True, "GOOGLE_API_KEY (Set)")
        if os.environ.get("GEMINI_API_KEY"):
            return (True, "GEMINI_API_KEY (Set)")

        # Check Vertex AI configuration
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        location = os.environ.get("GOOGLE_CLOUD_LOCATION")

        if project:
            loc_status = "Set" if location else "us-central1"
            return (True, f"Vertex AI (PROJECT=Set, LOCATION={loc_status})")
        return (False, "GOOGLE_API_KEY or GOOGLE_CLOUD_PROJECT (Not Set)")

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        vertexai: bool = False,
        project: Optional[str] = None,
        location: str = "us-central1",
        **kwargs: Any,
    ):
        super().__init__(model, **kwargs)
        self.vertexai = vertexai
        self.project = project
        self.location = location

        # This will be populated by _verify_model_available()
        self._model_context_window: Optional[int] = None

        try:
            if vertexai:
                self.client = genai.Client(
                    vertexai=True, project=project, location=location
                )
            else:
                self.client = genai.Client(api_key=api_key)
            self._verify_model_available()
            message = "Google "
            if self.vertexai:
                message += "Vertex AI "
            else:
                message += "Gemini "
            message += f"API provider initialized for '{model}'."
            logger.debug(message)
        except genai_errors.ClientError as e:
            # 4xx errors - likely auth/config issues
            raise ProviderError(
                f"Authentication or configuration error: {e.message}", provider="google"
            ) from e
        except genai_errors.ServerError as e:
            # 5xx errors - Google's problem
            raise ProviderError(
                f"Google API server error: {e.message}", provider="google"
            ) from e
        except genai_errors.APIError as e:
            # Catch-all for other API errors
            raise ProviderError(
                f"Google API error: {e.message}", provider="google"
            ) from e
        except Exception as e:
            # Non-API errors (network, etc.)
            raise ProviderError(
                f"Failed to initialize Google client: {e}", provider="google"
            ) from e

    def _verify_model_available(self):
        """Check if the configured model is available.
        This method verifies that:
        1. The model name is valid and can be used with Gemini/Vertex AI APIs.
        2. Retrieves the model's actual context window size (if available)
        """
        if not self.vertexai:
            model_name = "models/" + self.model
        else:
            model_name = self.model
        try:
            # First check if the model is in the list of available models
            pulled_models = self.client.models.list()
            available_model_names = {m.name for m in pulled_models}
            if model_name not in available_model_names:
                raise ProviderError(
                    f"Model '{self.model}' not available locally. "
                    f"If you're using Gemini API then ensure you use models/{self.model}.",
                    provider="google",
                )
            # Get detailed info
            model_info = next((m for m in pulled_models if m.name == self.model), None)
            # Extract context window from model info
            if model_info and model_info.input_token_limit:
                self._model_context_window = model_info.input_token_limit
                if self._model_context_window:
                    logger.debug(
                        f"Model '{self.model}' context window: "
                        f"{self._model_context_window} tokens"
                    )
        except genai_errors.APIError as e:
            raise ProviderError(
                f"Could not verify model '{self.model}': {e.message}",
                provider="google",
            ) from e

    @staticmethod
    def _convert_messages(
        messages: List[Message],
    ) -> tuple[Optional[str], List[types.Content]]:
        """
        Convert Allos messages to Google format.

        :param messages: A list of `allos.providers.base.Message` objects.
        :type messages: List[Message]
        :return: (system_instructtion, contents)
        :rtype: tuple[str | None, Any]
        """
        system_instruction = None
        contents: List[types.Content] = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_instruction = msg.content
            elif msg.role == MessageRole.USER:
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=msg.content or "")],
                    )
                )
            elif msg.role == MessageRole.ASSISTANT:
                parts: List[types.Part] = []
                if msg.content:
                    parts.append(types.Part.from_text(text=msg.content))
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        # Reconstructing function call part:
                        parts.append(
                            types.Part.from_function_call(
                                name=tc.name,
                                args=tc.arguments,
                            )
                        )
                    contents.append(types.Content(role="model", parts=parts))
            elif msg.role == MessageRole.TOOL:
                # Tool results are sent as 'user' role with function_response
                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_function_response(
                                name=msg.tool_call_id or "",
                                response={"result": msg.content},
                            )
                        ],
                    )
                )
        return system_instruction, contents

    @staticmethod
    def _convert_tools(tools: List[BaseTool]) -> List[types.Tool]:
        """
        Convert Allos tools to Google FunctionDeclaration format.

        :param tools: Description
        :type tools: List[BaseTool]
        :return: Description
        :rtype: List[Tool]
        """
        function_declarations = []

        for tool in tools:
            properties: Dict[str, Dict[str, str]] = {}
            required = []

            for param in tool.parameters:
                properties[param.name] = {
                    "type": param.type,
                    "description": param.description,
                }
                if param.required:
                    required.append(param.name)
            func_decl = types.FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters_json_schema={
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            )
            function_declarations.append(func_decl)
        return [types.Tool(function_declarations=function_declarations)]

    @staticmethod
    def _parse_response(
        response: types.GenerateContentResponse,
    ) -> Tuple[Optional[str], List[ToolCall]]:
        """
        Parse Google response to Allos format.

        :param response: Description
        :return: Description
        :rtype: Tuple[str | None, List[ToolCall]]
        """
        content = response.text  # can be None for function_call
        tool_calls = []

        if response.function_calls:
            for i, fc in enumerate(response.function_calls):
                tool_calls.append(
                    ToolCall(
                        id=f"call_{fc.name}_{int(time.time() * 1000)}_{i}",
                        name=fc.name or "unknown",
                        arguments=dict(fc.args) if fc.args else {},
                    )
                )
        return content, tool_calls

    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[BaseTool]] = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        system_instruction, contents = self._convert_messages(messages)

        config_kwargs = {**kwargs}
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if tools:
            config_kwargs["tools"] = self._convert_tools(tools)

        config = types.GenerateContentConfig(**config_kwargs)

        start_time = time.time()
        builder_kwargs = {
            "model": self.model,
            "contents": contents,
            "tools": tools or [],
        }
        try:
            response = self.client.models.generate_content(
                model=self.model, contents=contents, config=config
            )

            metadata = self._build_metadata(response, builder_kwargs, start_time)

            content, tool_calls = self._parse_response(response)

            return ProviderResponse(
                content=content, tool_calls=tool_calls, metadata=metadata
            )
        except genai_errors.ClientError as e:
            raise ProviderError(
                f"Google API client error ({e.code}): {e.message}", provider="google"
            ) from e
        except genai_errors.ServerError as e:
            raise ProviderError(
                f"Google API server error ({e.code}): {e.message}",
                provider="google",
            ) from e
        except genai_errors.APIError as e:
            raise ProviderError(
                f"Google API error: {e}",
                provider="google",
            ) from e

    def stream_chat(
        self,
        messages: List[Message],
        tools: Optional[List[BaseTool]] = None,
        **kwargs: Any,
    ) -> Iterator[ProviderChunk]:
        """
        Docstring for stream_chat

        :param self: Description
        :param messages: Description
        :type messages: List[Message]
        :param tools: Description
        :type tools: Optional[List[BaseTool]]
        :param kwargs: Description
        :type kwargs: Any
        :return: Description
        :rtype: Iterator[ProviderChunk]
        """
        system_instruction, contents = self._convert_messages(messages)
        config_kwargs = {**kwargs}
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if tools:
            config_kwargs["tools"] = self._convert_tools(tools)

        config = types.GenerateContentConfig(**config_kwargs)

        start_time = time.time()
        builder_kwargs = {
            "model": self.model,
            "contents": contents,
            "tools": tools or [],
        }

        try:
            for chunk in self.client.models.generate_content_stream(
                model=self.model, contents=contents, config=config
            ):
                # Yield text content
                if chunk.text:
                    yield ProviderChunk(content=chunk.text)

                # Yield function Calls
                if chunk.function_calls:
                    for i, fc in enumerate(chunk.function_calls):
                        yield ProviderChunk(
                            tool_call_done=ToolCall(
                                id=f"call_{fc.name}_{int(time.time() * 1000)}_{i}",
                                name=fc.name or "unknown",
                                arguments=dict(fc.args) if fc.args else {},
                            )
                        )
                # Final metadata chunk
                # Note: usage_metadata may not be available in all streaming chunks
                yield self._build_final_chunk(builder_kwargs, start_time)
        except genai_errors.APIError as e:
            raise ProviderError(
                f"Google API streaming error: {e}",
                provider="google",
            ) from e

    def _build_metadata(self, response, builder_kwargs, start_time):
        """Build Metadata from Google response."""
        usage = response.usage_metadata
        input_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
        output_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0

        synthetic_response = {
            "id": "google_response",
            "model": self.model,
            "status": "completed",
            "usage": type(
                "Usage",
                (),
                {"input_tokens": input_tokens, "output_tokens": output_tokens},
            )(),
        }

        builder = MetadataBuilder(
            provider_name="google",
            request_kwargs=builder_kwargs,
            start_time=start_time,
        )

        return builder.with_response_obj(
            type("obj", (object,), synthetic_response)()
        ).build()

    def _build_final_chunk(self, builder_kwargs, start_time):
        """Build final metadata chunk for streaming."""
        synthetic_response = {
            "id": "google_stream",
            "model": self.model,
            "status": "completed",
            "usage": type("Usage", (), {"input_tokens": 0, "output_tokens": 0})(),
        }

        builder = MetadataBuilder(
            provider_name="google",
            request_kwargs=builder_kwargs,
            start_time=start_time,
        )

        metadata = builder.with_response_obj(
            type("obj", (object,), synthetic_response)()
        ).build()

        return ProviderChunk(final_metadata=metadata)

    def get_context_window(self) -> int:
        if self._model_context_window:
            return self._model_context_window
        for model_prefix, size in MODEL_CONTEXT_WINDOWS.items():
            if model_prefix in self.model:
                return size
        return 4096  # Default for unknown models
