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
        self._check_required_arguments(arguments)
        self._validate_and_coerce_types(arguments)

    def _check_required_arguments(self, arguments: Dict[str, Any]) -> None:
        """Check that all required arguments are provided."""
        required_params = {p.name for p in self.parameters if p.required}
        provided_args = set(arguments.keys())
        missing_args = required_params - provided_args

        if missing_args:
            raise ToolError(
                f"Missing required arguments for tool '{self.name}': "
                f"{', '.join(sorted(missing_args))}"
            )

    def _validate_and_coerce_types(self, arguments: Dict[str, Any]) -> None:
        """Validate and coerce argument types."""
        param_map = {p.name: p for p in self.parameters}

        for name, value in arguments.items():
            if name not in param_map:
                continue  # Allow extra arguments from LLMs

            expected_type = param_map[name].type
            coerced_value = self._coerce_type(value, expected_type)
            arguments[name] = coerced_value

            if not self._is_valid_type(coerced_value, expected_type):
                raise ToolError(
                    f"Invalid type for argument '{name}' in tool '{self.name}'. "
                    f"Expected {expected_type}, but got {type(coerced_value).__name__}."
                )

    def _coerce_type(self, value: Any, expected_type: str) -> Any:
        """Attempt to coerce value to expected type."""
        if expected_type == "boolean" and isinstance(value, str):
            return self._coerce_boolean(value)
        if expected_type == "integer" and isinstance(value, str) and value.isdigit():
            return int(value)
        if expected_type == "number" and isinstance(value, str):
            try:
                return float(value)
            except (ValueError, TypeError):
                pass
        return value

    def _coerce_boolean(self, value: str) -> Any:
        """Coerce string to boolean."""
        lower_val = value.lower()
        if lower_val in ["true", "1"]:
            return True
        if lower_val in ["false", "0"]:
            return False
        return value

    def _is_valid_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type."""
        type_checks = {
            "string": lambda v: isinstance(v, str),
            "boolean": lambda v: isinstance(v, bool),
            "integer": lambda v: isinstance(v, int),
            "number": lambda v: isinstance(v, (int, float)),
        }
        return type_checks.get(expected_type, lambda v: True)(value)

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
