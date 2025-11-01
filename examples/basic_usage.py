from dotenv import load_dotenv

from allos.providers.base import Message, MessageRole
from allos.providers.registry import ProviderRegistry

load_dotenv()

# OpenAI
openai = ProviderRegistry.get_provider("openai", model="gpt-4")
response_openai = openai.chat([Message(role=MessageRole.USER, content="Hello")])

print(response_openai)

# Anthropic
anthropic = ProviderRegistry.get_provider("anthropic", model="claude-sonnet-4-5")
response_anthropic = anthropic.chat([Message(role=MessageRole.USER, content="Hello")])

print(response_anthropic)
