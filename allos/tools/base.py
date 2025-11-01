# allos/tools/base.py
"""Base classes and data structures for tools. This is just to satisfy type checkers and Linters for conftest and testing infrastructure."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolParameter:
    """Defines a parameter for a tool."""

    name: str
    type: str
    description: str
    required: bool = False


class BaseTool(ABC):
    """Abstract base class for all tools."""

    name: str = "base_tool"
    description: str = "This is a base tool."
    parameters: list[ToolParameter] = []

    @abstractmethod
    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """Executes the tool with the given arguments."""
        raise NotImplementedError
