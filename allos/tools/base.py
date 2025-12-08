# allos/tools/base.py

"""Base classes and data structures for all tools in the Allos SDK.

This module defines the abstract interface that all tool implementations must follow,
ensuring they are interchangeable and can be correctly registered and utilized
by the agent core.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List

from ..utils.errors import ToolError


class ToolPermission(str, Enum):
    """Enumeration for tool execution permissions."""

    ALWAYS_ALLOW = "always_allow"
    ALWAYS_DENY = "always_deny"
    ASK_USER = "ask_user"


@dataclass
class ToolParameter:
    """Represents a single parameter for a tool.

    Attributes:
        name: The name of the parameter.
        type: The data type of the parameter (e.g., "string", "number", "boolean").
        description: A clear description of what the parameter is for.
        required: A boolean indicating if the parameter is mandatory.
    """

    name: str
    type: str
    description: str
    required: bool = False


class BaseTool(ABC):
    """Abstract base class for all tools.

    All tool implementations must inherit from this class, define the required
    class attributes, and implement the `execute` method.
    """

    # --- Required attributes for all subclasses ---
    name: str = "base_tool"
    description: str = "A base tool that does nothing."
    parameters: List[ToolParameter] = []
    permission: ToolPermission = ToolPermission.ASK_USER

    @abstractmethod
    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        """Executes the tool's logic with the given arguments.

        This method must be implemented by all subclasses.

        Args:
            **kwargs: A dictionary of arguments for the tool, where keys
                      are the parameter names.

        Returns:
            A dictionary representing the result of the tool's execution.
            It is recommended to include a "status" or "success" key.
        """
        raise NotImplementedError

    def validate_arguments(self, arguments: Dict[str, Any]) -> None:
        """Validates the provided arguments against the tool's defined parameters.

        Raises:
            ToolError: If a required argument is missing or an argument
                       has an incorrect type.
        """
        required_params = {p.name for p in self.parameters if p.required}
        provided_args = set(arguments.keys())

        missing_args = required_params - provided_args
        if missing_args:
            raise ToolError(
                f"Missing required arguments for tool '{self.name}': "
                f"{', '.join(sorted(missing_args))}"
            )

        # Basic type checking can be added here in the future if needed,
        # but for now, we rely on the LLM to provide correct types.

    def to_provider_format(self, provider: str) -> Dict[str, Any]:
        """Converts the tool definition to a provider-specific format.

        Currently supports "openai" and "anthropic" formats.

        Args:
            provider: The name of the provider (e.g., "openai", "anthropic").

        Returns:
            A dictionary representing the tool in the provider's format.
        """
        param_schema: Dict[str, Any] = {
            "type": "object",
            "properties": {},
            "required": [],
        }

        for param in self.parameters:
            param_schema["properties"][param.name] = {
                "type": param.type,
                "description": param.description,
            }
            if param.required:
                param_schema["required"].append(param.name)

        if provider == "openai":
            param_schema["additionalProperties"] = False
            return {
                "type": "function",
                "name": self.name,
                "description": self.description,
                "parameters": param_schema,
                "strict": True,
            }
        elif provider == "anthropic":
            return {
                "name": self.name,
                "description": self.description,
                "input_schema": param_schema,
            }

        raise ValueError(f"Unsupported provider format requested: {provider}")

    def __repr__(self) -> str:
        """Provides a formal, unambiguous string representation of the tool instance.

        This representation is designed to be developer-friendly, closely resembling
        a constructor call. It includes the specific tool's class name and its
        registered `name`, making it invaluable for debugging, logging, and
        interactive inspection of an agent's capabilities.

        Returns:
            A string in the format 'ClassName(name='tool_name')'.
        """
        return f"{self.__class__.__name__}(name='{self.name}')"
