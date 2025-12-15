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
        # 1. Convert Allos Messages to the format NexusAI expects.
        nexus_messages = self._convert_messages(messages)

        try:
            # 2. Call the NexusAI API.
            response = self.client.generate(model=self.model, messages=nexus_messages)

            # 3. Parse the NexusAI response back into a standard Allos ProviderResponse.
            return self._parse_response(response)

        except nexusai.APIError as e:
            # 4. Handle provider-specific errors and wrap them in ProviderError.
            raise ProviderError(f"NexusAI API error: {e}", "nexusai") from e

    def stream_chat(self, messages: List[Message], **kwargs: Any) -> Iterator[ProviderChunk]:
        """
        Main method to interact with the NexusAI API (Streaming).
        """
        nexus_messages = self._convert_messages(messages)

        try:
            # Assume the client supports a streaming method
            stream = self.client.stream(model=self.model, messages=nexus_messages)

            for event in stream:
                # Convert provider events to Allos ProviderChunks
                if event.token:
                    yield ProviderChunk(content=event.token)
                elif event.is_done:
                    # In a real implementation, you would build the final metadata here
                    yield ProviderChunk(final_metadata=None)

    def get_context_window(self) -> int:
        """
        Return the context window size for the model.
        """
        # Return a known value or look it up
        if self.model == "nexus-pro":
            return 100000
        return 8000

    # Add private helper methods for conversion and parsing
    def _convert_messages(self, messages: List[Message]) -> List[dict]:
        # ... your implementation ...
        pass

    def _parse_response(self, response) -> ProviderResponse:
        # ... your implementation ...
        pass
```

## Step 3: Register the Provider

The most important step is registering the provider so the `ProviderRegistry` can find it.

In `allos/providers/__init__.py`, add a side-effect import for your new provider file.

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

Congratulations! You've successfully extended the Allos SDK with a new provider.
