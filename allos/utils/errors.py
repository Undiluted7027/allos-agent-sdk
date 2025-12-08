# allos/utils/errors.py

"""Custom exception hierarchy for the Allos Agent SDK."""


class AllosError(Exception):
    """Base exception for all errors raised by the Allos SDK."""

    def __init__(self, message: str):
        """Initialize the exception with a custom error message.

        Args:
            message (str): The error message describing the exception.
        """
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        """Return a string representation of the exception.

        Returns:
            str: A formatted string containing the exception class name and message.
                 Format: "{ClassName}: {message}"
        """
        return f"{self.__class__.__name__}: {self.message}"


class ConfigurationError(AllosError):
    """Raised for configuration-related errors."""


class ProviderError(AllosError):
    """Raised for errors related to LLM providers."""

    def __init__(self, message: str, provider: str):
        """Initialize the exception with a formatted message and provider information.

        Args:
            message (str): The error message describing what went wrong.
            provider (str): The name of the provider that raised this exception.

        Returns:
            None
        """
        super().__init__(f"[{provider}] {message}")
        self.provider = provider


class ToolError(AllosError):
    """Base exception for tool-related errors."""


class ToolNotFoundError(ToolError):
    """Raised when a requested tool cannot be found."""

    def __init__(self, tool_name: str):
        """Initialize a ToolNotFoundError exception.

        Args:
            tool_name (str): The name of the tool that was not found in the registry.

        Raises:
            ToolNotFoundError: Exception indicating the specified tool is not registered.
        """
        super().__init__(f"Tool '{tool_name}' not found in the registry.")
        self.tool_name = tool_name


class ToolExecutionError(ToolError):
    """Raised when a tool fails during execution."""

    def __init__(self, tool_name: str, error: str):
        """Initialize a tool execution error.

        Args:
            tool_name (str): The name of the tool that failed to execute.
            error (str): The error message or description of what went wrong during execution.
        """
        super().__init__(f"Error executing tool '{tool_name}': {error}")
        self.tool_name = tool_name
        self.original_error = error


class PermissionError(AllosError):
    """Raised when an agent action is denied by the user or security policy."""


class FileOperationError(AllosError):
    """Raised for errors during safe file operations."""


class ContextWindowExceededError(AllosError):
    """Raised when the conversation context exceeds the provider's model limit."""
