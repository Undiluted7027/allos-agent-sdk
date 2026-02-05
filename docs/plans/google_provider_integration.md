# Google Provider Integration Plan

## SDK Reference

**Package**: `google-genai` (version 1.61.0+, Python >=3.10)
**Import**: `from google import genai` and `from google.genai import types`
**Docs**: https://googleapis.github.io/python-genai/

---

## 1. Client Initialization

### Gemini Developer API (API Key)
```python
from google import genai
client = genai.Client(api_key='GEMINI_API_KEY')
# Or auto-detect from GEMINI_API_KEY or GOOGLE_API_KEY env var:
client = genai.Client()
```

### Vertex AI (GCP Project)
```python
client = genai.Client(
    vertexai=True,
    project='your-project-id',
    location='us-central1'  # Required for Vertex AI
)
# Or auto-detect from GOOGLE_GENAI_USE_VERTEXAI=true, GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION
```

---

## 2. Content Generation

### Non-Streaming
```python
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents='Why is the sky blue?',
    config=types.GenerateContentConfig(
        system_instruction='You are helpful.',
        temperature=0.7,
        max_output_tokens=4096,
    )
)
print(response.text)  # Access generated text
```

### Streaming
```python
for chunk in client.models.generate_content_stream(
    model='gemini-2.5-flash',
    contents='Tell me a story',
    config=types.GenerateContentConfig(...)
):
    print(chunk.text, end='')  # Each chunk has .text
```

---

## 3. Message/Content Format

### Content Structure
```python
# Simple string (auto-converted to user content)
contents = 'Hello'

# Explicit Content object
contents = types.Content(
    role='user',  # or 'model' for assistant
    parts=[types.Part.from_text('Hello')]
)

# Multi-turn conversation (list of Content)
contents = [
    types.Content(role='user', parts=[types.Part.from_text('Hi')]),
    types.Content(role='model', parts=[types.Part.from_text('Hello!')]),
    types.Content(role='user', parts=[types.Part.from_text('How are you?')]),
]
```

### System Instruction
System prompt is passed via `GenerateContentConfig.system_instruction`, NOT as a message:
```python
config = types.GenerateContentConfig(
    system_instruction='You are a helpful assistant.'
)
```

---

## 4. Function Calling (Tools)

### Define FunctionDeclaration
```python
from google.genai import types

weather_func = types.FunctionDeclaration(
    name='get_weather',
    description='Get weather for a location',
    parameters={
        'type': 'object',
        'properties': {
            'location': {
                'type': 'string',
                'description': 'City name, e.g. Boston, MA'
            },
            'unit': {
                'type': 'string',
                'enum': ['celsius', 'fahrenheit'],
                'description': 'Temperature unit'
            }
        },
        'required': ['location']
    }
)

# Wrap in Tool object
tool = types.Tool(function_declarations=[weather_func])
```

### Send Request with Tools
```python
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents='What is the weather in Boston?',
    config=types.GenerateContentConfig(
        tools=[tool]
    )
)
```

### Check for Function Calls in Response
```python
# Method 1: Direct access (shortcut)
if response.function_calls:
    fc = response.function_calls[0]
    print(fc.name)  # 'get_weather'
    print(fc.args)  # {'location': 'Boston, MA'}

# Method 2: Through candidates
for candidate in response.candidates:
    for part in candidate.content.parts:
        if hasattr(part, 'function_call') and part.function_call:
            fc = part.function_call
            print(fc.name, fc.args)
```

**IMPORTANT**: When model returns function_call, `response.text` may be None or empty.

### Send Function Result Back
```python
# Execute your function
result = get_weather(location='Boston, MA')  # Your actual function

# Create function response part
function_response = types.Part.from_function_response(
    name='get_weather',  # Must match the function name called
    response={'result': result}  # Wrap result in dict
)

# Build conversation with function result
contents = [
    types.Content(role='user', parts=[types.Part.from_text('What is the weather in Boston?')]),
    response.candidates[0].content,  # Model's function call response
    types.Content(role='user', parts=[function_response]),  # Function result (role='user')
]

# Get final response
final_response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=contents,
    config=types.GenerateContentConfig(tools=[tool])
)
print(final_response.text)  # Natural language response using the function result
```

---

## 5. Response Structure

### GenerateContentResponse
```python
response.text                    # str - Generated text (may be None if function_call)
response.candidates              # List[Candidate] - Response candidates
response.function_calls          # List[FunctionCall] - Shortcut to function calls
response.usage_metadata          # UsageMetadata - Token counts
```

### Candidate
```python
candidate = response.candidates[0]
candidate.content                # Content object
candidate.content.role           # 'model'
candidate.content.parts          # List[Part]
candidate.finish_reason          # FinishReason enum
candidate.safety_ratings         # List[SafetyRating]
```

### Part (polymorphic)
```python
part.text                        # str - If text part
part.function_call               # FunctionCall - If function call
part.function_call.name          # str - Function name
part.function_call.args          # dict - Function arguments
```

### UsageMetadata
```python
response.usage_metadata.prompt_token_count      # int - Input tokens
response.usage_metadata.candidates_token_count  # int - Output tokens
response.usage_metadata.total_token_count       # int - Total tokens
# Optional fields:
response.usage_metadata.cached_content_token_count
response.usage_metadata.tool_use_prompt_token_count
response.usage_metadata.thoughts_token_count
```

---

## 6. Streaming with Function Calls

```python
config = types.GenerateContentConfig(
    tools=[tool],
    tool_config=types.ToolConfig(
        function_calling_config=types.FunctionCallingConfig(
            mode=types.FunctionCallingConfigMode.AUTO,
        )
    )
)

for chunk in client.models.generate_content_stream(
    model='gemini-2.5-flash',
    contents='What is the weather?',
    config=config
):
    if chunk.text:
        print(chunk.text, end='')
    if chunk.function_calls:
        for fc in chunk.function_calls:
            print(f'Function call: {fc.name}({fc.args})')
```

---

## 7. Error Handling

### Error Classes (from google.genai.errors)
```python
from google.genai import errors

# Hierarchy:
# APIError (base)
#   ├── ClientError (4xx errors)
#   └── ServerError (5xx errors)
# ValueError subclasses:
#   ├── UnknownFunctionCallArgumentError
#   ├── UnsupportedFunctionError
#   ├── FunctionInvocationError
#   └── UnknownApiResponseError
```

### Error Attributes
```python
try:
    response = client.models.generate_content(...)
except errors.ClientError as e:
    print(e.code)      # HTTP status code (e.g., 429, 401, 400)
    print(e.message)   # Error message
    print(e.status)    # Error status string
    print(e.details)   # Full response JSON
except errors.ServerError as e:
    print(e.code)      # 5xx status code
except errors.APIError as e:
    # Catch-all for API errors
    pass
```

### Common Errors
- `ClientError 400 INVALID_ARGUMENT` - Bad request/input
- `ClientError 401 UNAUTHENTICATED` - Auth failure
- `ClientError 403 PERMISSION_DENIED` - No access
- `ClientError 404 NOT_FOUND` - Model not found
- `ClientError 429 RESOURCE_EXHAUSTED` - Rate limit
- `ServerError 500 INTERNAL` - Server error
- `ServerError 503 UNAVAILABLE` - Service unavailable

---

## 8. Model Context Windows

```python
MODEL_CONTEXT_WINDOWS = {
    'gemini-2.5-flash': 1048576,      # 1M tokens
    'gemini-2.5-pro': 1048576,        # 1M tokens
    'gemini-2.0-flash': 1048576,      # 1M tokens
    'gemini-1.5-flash': 1048576,      # 1M tokens
    'gemini-1.5-pro': 2097152,        # 2M tokens
    'gemini-1.0-pro': 32768,          # 32K tokens
}
```

---

## 9. Implementation Mapping (Allos → Google)

### Message Conversion
| Allos MessageRole | Google Role | Notes |
|-------------------|-------------|-------|
| SYSTEM | N/A | → `config.system_instruction` |
| USER | 'user' | → `Content(role='user', parts=[Part.from_text()])` |
| ASSISTANT | 'model' | → `Content(role='model', parts=[...])` |
| TOOL | 'user' | → `Content(role='user', parts=[Part.from_function_response()])` |

### Tool Conversion
| Allos ToolParameter.type | Google Schema type |
|--------------------------|-------------------|
| 'string' | 'string' |
| 'integer' | 'integer' |
| 'number' | 'number' |
| 'boolean' | 'boolean' |
| 'array' | 'array' |
| 'object' | 'object' |

### Response Conversion
| Google | Allos |
|--------|-------|
| `response.text` | `ProviderResponse.content` |
| `response.function_calls[i].name` | `ToolCall.name` |
| `response.function_calls[i].args` | `ToolCall.arguments` |
| `response.usage_metadata.prompt_token_count` | `Metadata.usage.input_tokens` |
| `response.usage_metadata.candidates_token_count` | `Metadata.usage.output_tokens` |

### ToolCall ID Generation
Google doesn't provide tool call IDs. Generate them:
```python
tool_call_id = f"call_{function_call.name}_{index}"
```

---

## 10. Files to Modify

### allos/providers/google.py (CREATE)
Main provider implementation.

### allos/providers/__init__.py (MODIFY)
Add dynamic import:
```python
try:
    from . import google
except (ImportError, AttributeError):
    logger.debug("Skipped optional provider: google")
```

### allos/providers/metadata.py (MODIFY)
Add Google-specific metadata:
```python
class ProviderSpecificGoogle(BaseModel):
    vertexai: bool = False
    project: Optional[str] = None
    location: Optional[str] = None

class ProviderSpecific(BaseModel):
    # ... existing fields ...
    google: Optional[ProviderSpecificGoogle] = None
```

### pyproject.toml
Already has: `google = ["google-genai>=1.47.0"]`

---

## 11. Environment Variable Validation (Framework Change)

### Problem

Google provider needs to support two authentication modes:
- **Gemini API**: `GOOGLE_API_KEY` or `GEMINI_API_KEY` (either works)
- **Vertex AI**: `GOOGLE_CLOUD_PROJECT` + optionally `GOOGLE_CLOUD_LOCATION`

The current `BaseProvider.env_var` is a single string, which can't express:
- OR logic (either `GOOGLE_API_KEY` or `GEMINI_API_KEY`)
- AND logic (both `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` for Vertex)
- Multiple valid configurations (Gemini API vs Vertex AI)

### Solution: Add `check_env_config()` Classmethod

Add a classmethod to `BaseProvider` that each provider can override. Default implementation uses existing `env_var` for backward compatibility.

### Changes to allos/providers/base.py

```python
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Tuple

# ... existing code ...

class BaseProvider(ABC):
    """Abstract base class for all LLM providers."""

    env_var: Optional[str] = None

    def __init__(self, model: str, **kwargs: Any):
        self.model = model
        self.provider_specific_kwargs = kwargs

    @classmethod
    def check_env_config(cls) -> Tuple[bool, str]:
        """Check if environment is properly configured for this provider.

        Override this method in subclasses that need complex env var logic
        (e.g., multiple valid configurations, OR/AND combinations).

        Returns:
            Tuple of (is_configured, display_message) where:
            - is_configured: True if the provider can be used
            - display_message: Human-readable status for CLI display
              Examples: "OPENAI_API_KEY (Set)", "GOOGLE_API_KEY (Not Set)",
                       "Vertex AI (PROJECT=Set, LOCATION=Set)"
        """
        if cls.env_var is None:
            return (True, "N/A (No key required)")
        if cls.env_var in os.environ:
            return (True, f"{cls.env_var} (Set)")
        return (False, f"{cls.env_var} (Not Set)")

    # ... rest of existing abstract methods ...
```

### Changes to allos/providers/registry.py

```python
# Add new method to ProviderRegistry class

class ProviderRegistry:
    # ... existing methods ...

    @classmethod
    def check_provider_env(cls, provider_name: str) -> Tuple[bool, str]:
        """Check environment configuration for a provider.

        This method delegates to the provider's check_env_config() classmethod,
        which allows each provider to implement its own validation logic.

        Args:
            provider_name: Name of the provider (e.g., "google", "openai")

        Returns:
            Tuple of (is_configured, display_message)
        """
        # Check aliases first
        if provider_name in OPENAI_COMPATIBLE_PROVIDERS:
            config = OPENAI_COMPATIBLE_PROVIDERS[provider_name]
            env_var = config.get("env_var")
            if env_var is None:
                return (True, "N/A (No key required)")
            if env_var in os.environ:
                return (True, f"{env_var} (Set)")
            return (False, f"{env_var} (Not Set)")

        # Check registered providers
        if provider_name in _provider_registry:
            provider_class = _provider_registry[provider_name]
            return provider_class.check_env_config()

        return (False, "Provider not found")
```

### Changes to allos/cli/main.py

Replace direct `env_var` checks with `check_provider_env()`:

```python
# In the provider status display function

for p in providers:
    # Use the new unified check method
    is_configured, var_display = ProviderRegistry.check_provider_env(p)

    # Special Case: Native Ollama (still needs running check)
    if p == "ollama":
        OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        if ollama_running(OLLAMA_URL):
            status = "[green]Ready[/]"
        else:
            status = "[yellow]Ollama not running[/]"
    elif is_configured:
        status = "[green]Ready[/]"
    else:
        status = "[red]Missing Key[/]"

    table.add_row(p, status, var_display)
```

### Changes to allos/cli/utils.py

Update `validate_api_key()` to use the new method:

```python
def validate_api_key(
    provider: str, api_key: Optional[str] = None
) -> Tuple[bool, str]:
    """Validates that the API key is available for the given provider.

    Args:
        provider: The provider name.
        api_key: Optional API key passed directly.

    Returns:
        A tuple of (is_valid, message) where:
        - is_valid: True if API key is available, False otherwise.
        - message: Status message (env var name if missing, empty if valid).
    """
    # If API key passed directly, it's valid
    if api_key:
        return (True, "")

    # Special case: Ollama doesn't need API key, just needs to be running
    if provider == "ollama":
        if ollama_running("http://localhost:11434"):
            return (True, "")
        return (False, "Ollama server not running")

    # Use provider's env check
    is_configured, message = ProviderRegistry.check_provider_env(provider)
    if is_configured:
        return (True, "")
    return (False, message)
```

### Implementation in allos/providers/google.py

```python
@provider("google")
class GoogleProvider(BaseProvider):
    env_var = "GOOGLE_API_KEY"  # Primary for simple Gemini API case

    @classmethod
    def check_env_config(cls) -> Tuple[bool, str]:
        """Check for Gemini API key or Vertex AI configuration.

        Supports two authentication modes:
        1. Gemini Developer API: GOOGLE_API_KEY or GEMINI_API_KEY
        2. Vertex AI: GOOGLE_CLOUD_PROJECT (+ optional GOOGLE_CLOUD_LOCATION)
        """
        import os

        # Check Gemini API keys (OR logic)
        google_key = os.environ.get("GOOGLE_API_KEY")
        gemini_key = os.environ.get("GEMINI_API_KEY")

        if google_key:
            return (True, "GOOGLE_API_KEY (Set)")
        if gemini_key:
            return (True, "GEMINI_API_KEY (Set)")

        # Check Vertex AI configuration
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        location = os.environ.get("GOOGLE_CLOUD_LOCATION")
        use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() == "true"

        if project:
            loc_status = "Set" if location else "us-central1"
            if use_vertex:
                return (True, f"Vertex AI (PROJECT=Set, LOCATION={loc_status})")
            # Project set but USE_VERTEXAI not set - still valid, SDK auto-detects
            return (True, f"GOOGLE_CLOUD_PROJECT (Set)")

        # Nothing configured
        return (False, "GOOGLE_API_KEY or GOOGLE_CLOUD_PROJECT (Not Set)")

    # ... rest of implementation ...
```

### Summary of Changes

| File | Change |
|------|--------|
| `allos/providers/base.py` | Add `check_env_config()` classmethod with default implementation |
| `allos/providers/registry.py` | Add `check_provider_env()` that delegates to provider classmethod |
| `allos/cli/main.py` | Replace `get_env_var_name()` usage with `check_provider_env()` |
| `allos/cli/utils.py` | Update `validate_api_key()` to use `check_provider_env()` |
| `allos/providers/google.py` | Override `check_env_config()` with Gemini/Vertex logic |

### Backward Compatibility

- Existing providers continue to work unchanged (default implementation uses `env_var`)
- `get_env_var_name()` can remain for any code that still needs just the env var name
- No changes required to existing provider implementations unless they need complex logic

---

## 12. Provider Implementation Skeleton

```python
# allos/providers/google.py

import time
from typing import Any, Dict, Iterator, List, Optional

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

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

MODEL_CONTEXT_WINDOWS = {
    'gemini-2.5-flash': 1048576,
    'gemini-2.5-pro': 1048576,
    'gemini-2.0-flash': 1048576,
    'gemini-1.5-flash': 1048576,
    'gemini-1.5-pro': 2097152,
    'gemini-1.0-pro': 32768,
}


@provider("google")
class GoogleProvider(BaseProvider):
    env_var = "GOOGLE_API_KEY"

    @classmethod
    def check_env_config(cls) -> tuple[bool, str]:
        """Check for Gemini API key or Vertex AI configuration."""
        import os

        # Check Gemini API keys (OR logic)
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

        try:
            if vertexai:
                self.client = genai.Client(
                    vertexai=True,
                    project=project,
                    location=location,
                )
            else:
                self.client = genai.Client(api_key=api_key)
        except Exception as e:
            raise ProviderError(
                f"Failed to initialize Google client: {e}",
                provider="google",
            ) from e

    @staticmethod
    def _convert_messages(
        messages: List[Message],
    ) -> tuple[Optional[str], List[types.Content]]:
        """Convert Allos messages to Google format.

        Returns (system_instruction, contents).
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
                        parts=[types.Part.from_text(msg.content or "")],
                    )
                )
            elif msg.role == MessageRole.ASSISTANT:
                parts = []
                if msg.content:
                    parts.append(types.Part.from_text(msg.content))
                for tc in msg.tool_calls:
                    # Reconstruct function call part
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
        """Convert Allos tools to Google FunctionDeclaration format."""
        function_declarations = []

        for tool in tools:
            properties = {}
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
                parameters={
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            )
            function_declarations.append(func_decl)

        return [types.Tool(function_declarations=function_declarations)]

    @staticmethod
    def _parse_response(
        response,
    ) -> tuple[Optional[str], List[ToolCall]]:
        """Parse Google response to Allos format."""
        content = response.text  # May be None if function_call
        tool_calls = []

        if response.function_calls:
            for i, fc in enumerate(response.function_calls):
                tool_calls.append(
                    ToolCall(
                        id=f"call_{fc.name}_{i}",
                        name=fc.name,
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
                model=self.model,
                contents=contents,
                config=config,
            )

            metadata = self._build_metadata(response, builder_kwargs, start_time)
            content, tool_calls = self._parse_response(response)

            return ProviderResponse(
                content=content,
                tool_calls=tool_calls,
                metadata=metadata,
            )

        except genai_errors.ClientError as e:
            raise ProviderError(
                f"Google API client error ({e.code}): {e.message}",
                provider="google",
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
                model=self.model,
                contents=contents,
                config=config,
            ):
                # Yield text content
                if chunk.text:
                    yield ProviderChunk(content=chunk.text)

                # Yield function calls
                if chunk.function_calls:
                    for i, fc in enumerate(chunk.function_calls):
                        yield ProviderChunk(
                            tool_call_done=ToolCall(
                                id=f"call_{fc.name}_{i}",
                                name=fc.name,
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
        for model_prefix, size in MODEL_CONTEXT_WINDOWS.items():
            if model_prefix in self.model:
                return size
        return 32768  # Default for unknown models
```

---

## 13. Test Plan

### Unit Tests (tests/unit/providers/test_google_provider.py)
```python
@patch("allos.providers.google.genai.Client")
class TestGoogleProvider:
    def test_init_gemini_api(self, MockClient):
        """Init with API key (Gemini Developer API)."""

    def test_init_vertex_ai(self, MockClient):
        """Init with vertexai=True, project, location."""

    def test_convert_messages_system(self, MockClient):
        """System message → system_instruction."""

    def test_convert_messages_user_assistant(self, MockClient):
        """USER/ASSISTANT → Content with correct roles."""

    def test_convert_messages_tool_result(self, MockClient):
        """TOOL message → function_response part."""

    def test_convert_tools(self, MockClient):
        """BaseTool → FunctionDeclaration."""

    def test_parse_response_text(self, MockClient):
        """Parse text-only response."""

    def test_parse_response_function_call(self, MockClient):
        """Parse response with function_calls."""

    def test_chat_basic(self, MockClient):
        """Basic chat without tools."""

    def test_chat_with_tools(self, MockClient):
        """Chat with tool calling."""

    def test_stream_chat_text(self, MockClient):
        """Streaming text response."""

    def test_stream_chat_function_call(self, MockClient):
        """Streaming with function calls."""

    def test_error_client_error(self, MockClient):
        """ClientError → ProviderError."""

    def test_error_server_error(self, MockClient):
        """ServerError → ProviderError."""

    def test_context_window(self, MockClient):
        """Context window lookup."""
```

---

## 14. Sources

- [Google Gen AI SDK Documentation](https://googleapis.github.io/python-genai/)
- [GitHub: googleapis/python-genai](https://github.com/googleapis/python-genai)
- [Gemini API Function Calling](https://ai.google.dev/gemini-api/docs/function-calling)
- [Vertex AI Function Calling](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling)
- [Gemini vs Vertex AI Migration](https://ai.google.dev/gemini-api/docs/migrate-to-cloud)
- [google-genai on PyPI](https://pypi.org/project/google-genai/)
- [SDK Error Classes](https://github.com/googleapis/python-genai/blob/main/google/genai/errors.py)
