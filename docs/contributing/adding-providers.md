# Guide: Adding a New Provider

The Allos SDK is designed to be extensible. Adding a new LLM provider is a straightforward process. This guide will walk you through the steps.

Let's assume we want to add a provider for a fictional "NexusAI".

## Step 1: Create the Provider File

Create a new file in `allos/providers/nexusai.py`.

## Step 2: Implement the Provider Class

Inside the new file, create a class that inherits from `BaseProvider` and implement the required abstract methods: `chat`, `stream_chat`, and `get_context_window`.

```python
# allos/providers/nexusai.py

from typing import List, Optional, Any
from allos.providers import BaseProvider, Message, ProviderResponse, provider
from allos.utils.errors import ProviderError
# Assume a fictional 'nexusai' client library
import nexusai

@provider("nexusai") # The decorator that registers the provider
class NexusAIProvider(BaseProvider):
    """Example provider template for Allos."""

    env_var = "NEXUSAI_API_KEY"

    @classmethod
    def check_env_config(cls) -> Tuple[bool, str]:
        """Return (is_configured, status_message) for CLI readiness checks."""
        if cls.env_var and os.getenv(cls.env_var):
            return (True, f"{cls.env_var} (Set)")
        return (False, f"{cls.env_var} (Not Set)")

    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs: Any):
        super().__init__(model, **kwargs)
        try:
            # Initialize the provider-specific client
            self.client = nexusai.Client(api_key=api_key)
        except Exception as e:
            raise ProviderError(f"Failed to initialize NexusAI client: {e}", "nexusai")

    def chat(self, messages: List[Message], **kwargs: Any) -> ProviderResponse:
        """
        Main method to interact with the NexusAI API.
        """
        start_time = time.time()
        try:
            # raw_response = self.client.chat(...)
            raw_response = self._mock_sync_response()

            content, tool_calls = self._parse_sync(raw_response)

            metadata = (
                MetadataBuilder(
                    provider_name="nexusai",
                    request_kwargs=kwargs,
                    start_time=start_time,
                )
                .with_response_obj(raw_response)
                .build()
            )

            return ProviderResponse(
                content=content,
                tool_calls=tool_calls,
                metadata=metadata,
                # Include this only if your provider needs it:
                # thought_signatures=...
            )
        except Exception as e:
            raise ProviderError(f"NexusAI API error: {e}", provider="nexusai") from e

    def stream_chat(self, messages: List[Message], **kwargs: Any) -> Iterator[ProviderChunk]:
        """
        Main method to interact with the NexusAI API (Streaming).
        """
        start_time = time.time()
        raw_final_response: Any = None

        try:
            # stream = self.client.stream_chat(...)
            stream = self._mock_stream_response()

            for event in stream:
                raw_final_response = event.raw_response
                if event.text_delta:
                    yield ProviderChunk(content=event.text_delta)

                # Optional for providers that emit reasoning-state continuity artifacts:
                # if event.thought_signature and event.tool_call_id:
                #     yield ProviderChunk(thought_signatures={event.tool_call_id: event.thought_signature})

                # Optional tool streaming pattern:
                # if event.tool_call_done:
                #     yield ProviderChunk(tool_call_done=ToolCall(...))

            if raw_final_response is not None:
                metadata = (
                    MetadataBuilder(
                        provider_name="nexusai",
                        request_kwargs=kwargs,
                        start_time=start_time,
                    )
                    .with_response_obj(raw_final_response)
                    .build()
                )
                yield ProviderChunk(final_metadata=metadata)

        except Exception as e:
            yield ProviderChunk(error=f"NexusAI streaming error: {e}")

    def get_context_window(self) -> int:
        """
        Return the context window size for the model.
        """
        # Return a known value or look it up
        if self.model == "nexus-pro":
            return 100000
        return 8000

    # Add private helper methods for conversion and parsing
    def _convert_messages(self, messages: List[Message]) -> List[Dict]:
        # ... your implementation ...
        pass

    def _convert_tools(self, tools: List[Tools]) -> List[Dict]:
        # ... your implementation ...
        pass
```

## Step 3: Register the Provider

The most important step is registering the provider so the `ProviderRegistry` can find it.

In `allos/providers/__init__.py`, add a side-effect import for your new provider file.

Registration can be conditional for optional dependencies (same pattern as `allos/providers/__init__.py`).

```python
# allos/providers/__init__.py

# ... other imports

# Add your new provider to the list of imports
from . import openai
from . import anthropic
from . import nexusai # Add this line
```

## Step 4: Add Tests

-   Add comprehensive unit tests for your provider in `tests/unit/test_nexusai_provider.py`. Use mocking to avoid making real API calls.
-   Add integration tests in `tests/integration/test_nexusai_provider.py`. Mark them with the `run_integration_tests` decorator.

## Advanced: Supporting Thought Signatures

Some LLM providers (like Google Gemini 3.x) use "thought signatures" - encrypted reasoning state that must be preserved across conversation turns.

See `allos/providers/google.py` for a complete reference implementation.

Congratulations! You've successfully extended the Allos SDK with a new provider.
