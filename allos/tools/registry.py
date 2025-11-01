# allos/tools/registry.py

"""
A registry for discovering and instantiating tool classes.
"""

from typing import Dict, List, Type

from ..utils.errors import ToolNotFoundError
from .base import BaseTool

# The global registry dictionary mapping tool names to their classes
_tool_registry: Dict[str, Type[BaseTool]] = {}


def tool(cls: Type[BaseTool]) -> Type[BaseTool]:
    """
    A class decorator to register a new tool.

    Usage:
        @tool
        class MyCustomTool(BaseTool):
            ...
    """
    if not issubclass(cls, BaseTool):
        raise TypeError("Registered class must be a subclass of BaseTool.")

    tool_name = cls.name
    if tool_name in _tool_registry:
        raise ValueError(f"Tool '{tool_name}' is already registered.")

    _tool_registry[tool_name] = cls
    return cls


class ToolRegistry:
    """
    A factory class for creating tool instances.
    """

    @classmethod
    def get_tool(cls, name: str) -> BaseTool:
        """
        Get an instance of a registered tool.

        Args:
            name: The name of the tool (e.g., "read_file").

        Returns:
            An instance of the requested tool.

        Raises:
            ToolNotFoundError: If the tool is not registered.
        """
        if name not in _tool_registry:
            raise ToolNotFoundError(tool_name=name)

        tool_class = _tool_registry[name]
        return tool_class()

    @classmethod
    def list_tools(cls) -> List[str]:
        """
        List the names of all registered tools.
        """
        return sorted(_tool_registry.keys())

    @classmethod
    def get_all_tools(cls) -> List[BaseTool]:
        """
        Get instances of all registered tools.
        """
        return [cls.get_tool(name) for name in cls.list_tools()]
