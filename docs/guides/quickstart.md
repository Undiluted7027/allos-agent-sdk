# 5-Minute Quickstart

Get your first LLM response using the Allos SDK in under 5 minutes. This guide will show you how to use both OpenAI and Anthropic with the exact same code structure.

## Prerequisites

1.  **Install Allos** with OpenAI and Anthropic support:
    ```bash
    uv pip install "allos-agent-sdk[all]" python-dotenv
    ```
2.  **Create a `.env` file** in your project directory with your API keys:
    ```env
    # .env
    OPENAI_API_KEY="your_openai_api_key_here"
    ANTHROPIC_API_KEY="your_anthropic_api_key_here"
    ```

## Your First Script

Create a file named `quickstart.py` and paste the following code into it.

```python
# quickstart.py
from dotenv import load_dotenv

from allos.providers import Message, MessageRole, ProviderRegistry

# Load API keys from your .env file
load_dotenv()

def run_provider_test(provider_name: str, model: str):
    """
    Initializes a provider and gets a simple response.
    """
    print(f"--- Testing Provider: {provider_name.upper()} ---")
    try:
        # 1. Get the provider from the registry
        provider = ProviderRegistry.get_provider(provider_name, model=model)

        # 2. Create the message payload
        messages = [
            Message(role=MessageRole.USER, content="Hello! Write a one-sentence greeting.")
        ]

        # 3. Get the response using the unified .chat() method
        response = provider.chat(messages)

        print(f"Model: {model}")
        print(f"Response: {response.content}\n")

    except Exception as e:
        print(f"An error occurred: {e}\n")


if __name__ == "__main__":
    # Run the test for OpenAI
    run_provider_test("openai", model="gpt-4o")

    # Run the exact same test for Anthropic
    run_provider_test("anthropic", model="claude-3-haiku-20240307")

```

## Run It!

Execute the script from your terminal:

```bash
python quickstart.py
```

### Expected Output

You will see a response from both providers, demonstrating the seamless switching capability of Allos!

```text
--- Testing Provider: OPENAI ---
Model: gpt-4o
Response: Hello there! It's a pleasure to connect with you.

--- Testing Provider: ANTHROPIC ---
Model: claude-3-haiku-20240307
Response: Hello! I hope you're having a wonderful day.
```

Congratulations! you have successfully used the Allos SDK to interact with multiple LLM providers.
