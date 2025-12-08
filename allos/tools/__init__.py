# allos/tools/__init__.py

"""Defines the abstraction layer and pluggable architecture for agent capabilities.

This module provides the core framework for defining, registering, and managing
"tools" — the discrete capabilities that an Allos agent can use to interact with
its environment. It allows the agent to move beyond simple text generation to
perform actions like reading files, executing code, or searching the web.

The architecture is designed to be extensible, mirroring the provider system:
 - `BaseTool`: An abstract class that defines the contract all tool
   implementations must follow, specifying a name, description, parameters,
   a permission level, and an `execute` method.
 - `ToolRegistry`: A factory class that discovers and instantiates tools on demand.
 - `@tool` decorator: A simple mechanism for new tool classes to register
   themselves automatically with the registry.

This design allows developers to easily create and add new, custom capabilities
to any agent.
"""

# --- Side-effect imports to register the providers ---
# These imports are for side-effects: they trigger the @tool decorator
# in each file, populating the ToolRegistry.
from . import execution, filesystem  # noqa: F401
from .base import BaseTool, ToolParameter, ToolPermission
from .registry import ToolRegistry, tool

__all__ = [
    "BaseTool",
    "ToolParameter",
    "ToolPermission",
    "ToolRegistry",
    "tool",
]
