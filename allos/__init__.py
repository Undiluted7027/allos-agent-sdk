# allos/__init__.py

"""
Allos Agent SDK: The LLM-Agnostic Agentic Framework.

This is the main entry point for the `allos` package.
It exposes the primary components for building and running AI agents.
"""

from .__version__ import __version__
from .agent import Agent, AgentConfig
from .context import ConversationContext
from .providers import ProviderRegistry
from .tools import BaseTool, ToolRegistry, tool
from .utils.errors import AllosError

__all__ = [
    "__version__",
    "Agent",
    "AgentConfig",
    "ConversationContext",
    "ProviderRegistry",
    "ToolRegistry",
    "BaseTool",
    "tool",
    "AllosError",
]
