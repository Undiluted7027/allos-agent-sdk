# allos/utils/token_counter.py

"""Token counting and text truncation utilities."""

import tiktoken

from .logging import logger

# A rough approximation for token count when tiktoken is not applicable
# Assumes on average a token is ~4 characters
CHARS_PER_TOKEN = 4


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Counts the number of tokens in a given text string for a specific model.

    Uses `tiktoken` for supported models, otherwise falls back to a
    character-based approximation.

    Args:
        text: The text to analyze.
        model: The model name to use for tokenization.

    Returns:
        The estimated number of tokens.
    """
    if not text:
        return 0

    try:
        # Get the encoding for the specified model
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except KeyError:
        # If the model is not found in tiktoken, use a fallback
        logger.debug(
            f"Model '{model}' not found in tiktoken. Using char-based fallback."
        )
        return len(text) // CHARS_PER_TOKEN
    except Exception as e:
        logger.warning(
            f"An unexpected error occurred with tiktoken: {e}. Using fallback."
        )
        return len(text) // CHARS_PER_TOKEN


def truncate_text_by_tokens(text: str, max_tokens: int, model: str = "gpt-4") -> str:
    """Truncates text to a maximum number of tokens.

    Args:
        text: The text to truncate.
        max_tokens: The maximum number of tokens to allow.
        model: The model name to use for tokenization.

    Returns:
        The truncated text.
    """
    if count_tokens(text, model) <= max_tokens:
        return text

    try:
        encoding = tiktoken.encoding_for_model(model)
        tokens = encoding.encode(text)
        truncated_tokens = tokens[:max_tokens]
        return encoding.decode(truncated_tokens)
    except (KeyError, Exception):
        # Fallback to character-based truncation
        max_chars = max_tokens * CHARS_PER_TOKEN
        return text[:max_chars]
