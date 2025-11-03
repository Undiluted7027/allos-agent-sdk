# allos/tools/__init__.py

"""
The `tools` module contains the abstraction layer and concrete implementations
for tools that an Allos agent can use to interact with its environment.
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
