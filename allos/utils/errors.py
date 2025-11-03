# allos/utils/errors.py

"""Custom exception hierarchy for the Allos Agent SDK."""


class AllosError(Exception):
    """Base exception for all errors raised by the Allos SDK."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}: {self.message}"


class ConfigurationError(AllosError):
    """Raised for configuration-related errors."""


class ProviderError(AllosError):
    """Raised for errors related to LLM providers."""

    def __init__(self, message: str, provider: str):
        super().__init__(f"[{provider}] {message}")
        self.provider = provider


class ToolError(AllosError):
    """Base exception for tool-related errors."""


class ToolNotFoundError(ToolError):
    """Raised when a requested tool cannot be found."""

    def __init__(self, tool_name: str):
        super().__init__(f"Tool '{tool_name}' not found in the registry.")
        self.tool_name = tool_name


class ToolExecutionError(ToolError):
    """Raised when a tool fails during execution."""

    def __init__(self, tool_name: str, error: str):
        super().__init__(f"Error executing tool '{tool_name}': {error}")
        self.tool_name = tool_name
        self.original_error = error


class PermissionError(AllosError):
    """Raised when an agent action is denied by the user or security policy."""


class FileOperationError(AllosError):
    """Raised for errors during safe file operations."""


class ContextWindowExceededError(AllosError):
    """Raised when the conversation context exceeds the provider's model limit."""
