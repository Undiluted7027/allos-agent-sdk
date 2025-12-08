# allos/utils/logging.py

"""Logging configuration for the Allos Agent SDK."""

import logging
from typing import Any, Optional, cast

from rich.logging import RichHandler

# Define a custom log level for agent-specific thoughts or plans
THOUGHT_LEVEL = 15
logging.addLevelName(THOUGHT_LEVEL, "THOUGHT")


class ThoughtLogger(logging.Logger):
    """Custom logger with THOUGHT log level support."""

    def thought(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log a message with level THOUGHT."""
        if self.isEnabledFor(THOUGHT_LEVEL):
            self._log(THOUGHT_LEVEL, message, args, **kwargs)


logging.setLoggerClass(ThoughtLogger)

# Configure the root logger for the application
logger = cast(ThoughtLogger, logging.getLogger("allos"))


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Set up logging for the SDK.

    Args:
        level: The minimum log level to display (e.g., "DEBUG", "INFO").
        log_file: Optional path to a file to write logs to.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)

    # Prevent adding handlers multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler with rich formatting
    console_handler = RichHandler(
        rich_tracebacks=True,
        tracebacks_show_locals=True,
        markup=True,
        log_time_format="[%X]",
    )
    console_handler.setLevel(log_level)
    logger.addHandler(console_handler)

    # File handler, if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

    # Silence noisy loggers from dependencies
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)

    logger.debug(f"Logger configured with level {level.upper()}")


# Set up a default logger on import
if not logger.hasHandlers():
    setup_logging()
