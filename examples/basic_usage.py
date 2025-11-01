# examples/basic_usage.py

"""
A basic example demonstrating how to get a simple response from an LLM provider.

This script shows the fundamental steps:
1. Load environment variables.
2. Get a provider instance from the ProviderRegistry.
3. Create a list of messages.
4. Call the .chat() method and print the response.

To run this example:
1. Install the required dependencies: `uv pip install "allos-agent-sdk[openai]" python-dotenv`
2. Create a .env file and add your OPENAI_API_KEY.
3. Run the script: `python examples/basic_usage.py`
"""

from dotenv import load_dotenv

from allos.providers import Message, MessageRole, ProviderRegistry
from allos.utils.errors import AllosError

# Load API keys from your .env file
load_dotenv()


def main():
    """Main function to run the basic usage example."""
    try:
        # 1. Get the OpenAI provider from the registry
        print("Initializing OpenAI provider...")
        provider = ProviderRegistry.get_provider("openai", model="gpt-4o")
        print(f"Successfully initialized provider for model: {provider.model}")

        # 2. Define the conversation payload
        messages = [
            Message(role=MessageRole.SYSTEM, content="You are a helpful assistant."),
            Message(
                role=MessageRole.USER,
                content="Explain the greek term allos in context of an LLM provider agnostic SDK.",
            ),
        ]
        print("\nSending request to OpenAI...")

        # 3. Call the unified .chat() method
        response = provider.chat(messages)

        # 4. Print the result
        print("\nReceived response:")
        print(f"  Content: {response.content}")
        print("\n--- Metadata ---")
        import pprint

        pprint.pprint(response.metadata)
        print("------------------")

    except AllosError as e:
        print(f"\nAn error occurred: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
