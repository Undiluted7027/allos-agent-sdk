# examples/provider_switching.py

"""
An example demonstrating the core provider-switching capability of the Allos SDK.

This script runs the exact same prompt through both the OpenAI and Anthropic
providers, showcasing the unified `.chat()` interface.

To run this example:
1. Install all provider dependencies: `uv pip install "allos-agent-sdk[all]" python-dotenv`
2. Create a .env file and add both your OPENAI_API_KEY and ANTHROPIC_API_KEY.
3. Run the script: `python examples/provider_switching.py`
"""

from dotenv import load_dotenv

from allos.providers import Message, MessageRole, ProviderRegistry
from allos.utils.errors import AllosError

# Load API keys from your .env file
load_dotenv()

# The conversation is defined once and reused for both providers
MESSAGES = [
    Message(
        role=MessageRole.USER, content="Write a short, three-line poem about the moon."
    )
]

PROVIDERS_TO_TEST = {
    "openai": "gpt-4o",
    "anthropic": "claude-3-haiku-20240307",
}


def main():
    """Main function to run the provider switching example."""
    print("--- Demonstrating Provider Switching ---")

    for provider_name, model_name in PROVIDERS_TO_TEST.items():
        print(f"\n--- Testing Provider: {provider_name.upper()} ---")
        try:
            # Get the provider from the registry
            provider = ProviderRegistry.get_provider(provider_name, model=model_name)

            # The .chat() call is identical for every provider
            response = provider.chat(MESSAGES)

            print(f"Model: {model_name}")
            print("Response:")
            # We add indentation to make the poem stand out
            if response.content:
                indented_content = "\n".join(
                    [f"  {line}" for line in response.content.strip().split("\n")]
                )
                print(indented_content)
            else:
                print("  No content returned.")

        except AllosError as e:
            print(f"  An error occurred with provider '{provider_name}': {e}")
        except Exception as e:
            print(f"  An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
