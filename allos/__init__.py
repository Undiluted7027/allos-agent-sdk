# allos/__init__.py

"""
Allos Agent SDK: The LLM-Agnostic Agentic Framework.

This is the main entry point for the `allos` package.
It exposes the primary components for building and running AI agents.
"""

from .__version__ import __version__
from .utils.errors import AllosError

# The following will be uncommented and added as we build them
# from .agent.agent import Agent
# from .agent.config import AgentConfig
# from .tools.base import BaseTool
# from .tools.decorator import tool

__all__ = [
    "__version__",
    "AllosError",
    # "Agent",
    # "AgentConfig",
    # "BaseTool",
    # "tool",
]
