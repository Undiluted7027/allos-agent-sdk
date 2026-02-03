# allos/utils/token_counter.py

"""Token counting and text truncation utilities."""

from functools import lru_cache
from typing import Optional

import tiktoken

from .logging import logger

# A rough approximation for token count when tiktoken is not applicable
# Assumes on average a token is ~4 characters
CHARS_PER_TOKEN = 4

# Mapping of Ollama model families to tiktoken encodings
# Most modern LLMs use BPE tokenizers similar to cl100k_base
OLLAMA_MODEL_TO_ENCODING = {
    # Llama family - uses similar tokenization to GPT-4
    "llama3": "cl100k_base",
    "llama3.1": "cl100k_base",
    "llama3.2": "cl100k_base",
    "llama2": "cl100k_base",
    "codellama": "cl100k_base",
    # Mistral family
    "mistral": "cl100k_base",
    "mixtral": "cl100k_base",
    # Qwen family
    "qwen": "cl100k_base",
    "qwen2": "cl100k_base",
    "qwen2.5": "cl100k_base",
    "qwen2.5-coder": "cl100k_base",
    "qwen3": "cl100k_base",
    # Google family
    "gemma": "cl100k_base",
    "gemma2": "cl100k_base",
    # Phi family
    "phi": "cl100k_base",
    "phi3": "cl100k_base",
    # DeepSeek family
    "deepseek": "cl100k_base",
    "deepseek-coder": "cl100k_base",
    "deepseek-coder-v2": "cl100k_base",
    # StarCoder family
    "starcoder": "cl100k_base",
    "starcoder2": "cl100k_base",
}


def _get_encoding_for_ollama_model(model: str) -> Optional[str]:
    """Map an Ollama model name to a tiktoken encoding.

    Args:
        model: The Ollama model name (e.g., 'llama3.1:8b', 'mistral:latest').

    Returns:
        The tiktoken encoding name, or None if no mapping found.
    """
    # Strip version/size tags (e.g., 'llama3.1:8b' -> 'llama3.1')
    base_model = model.split(":")[0].lower()

    # Try exact match first
    if base_model in OLLAMA_MODEL_TO_ENCODING:
        return OLLAMA_MODEL_TO_ENCODING[base_model]

    # Try prefix match (e.g., 'llama3.1-custom' matches 'llama3.1')
    for family, encoding in OLLAMA_MODEL_TO_ENCODING.items():
        if base_model.startswith(family):
            return encoding

    return None


@lru_cache(maxsize=32)
def _get_encoding(model: str) -> Optional[tiktoken.Encoding]:
    """Get tiktoken encoding for a model with caching.

    Args:
        model: The model name.

    Returns:
        The tiktoken Encoding object, or None if not found.
    """
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        # Check if it's an Ollama model
        ollama_encoding = _get_encoding_for_ollama_model(model)
        if ollama_encoding:
            logger.debug(
                f"Using '{ollama_encoding}' encoding for Ollama model '{model}'."
            )
            return tiktoken.get_encoding(ollama_encoding)
        return None


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Counts the number of tokens in a given text string for a specific model.

    Uses `tiktoken` for supported models (including Ollama model families),
    otherwise falls back to a character-based approximation.

    Args:
        text: The text to analyze.
        model: The model name to use for tokenization. Supports OpenAI models,
               Anthropic models, and Ollama model names (e.g., 'llama3.1:8b').

    Returns:
        The estimated number of tokens.
    """
    if not text:
        return 0

    try:
        # Get the encoding using our helper (handles Ollama models too)
        encoding = _get_encoding(model)
        if encoding:
            return len(encoding.encode(text))

        # No encoding found - use character-based fallback
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
        model: The model name to use for tokenization. Supports OpenAI models,
               Anthropic models, and Ollama model names (e.g., 'llama3.1:8b').

    Returns:
        The truncated text.
    """
    if count_tokens(text, model) <= max_tokens:
        return text

    try:
        # Get the encoding using our helper (handles Ollama models too)
        encoding = _get_encoding(model)
        if encoding:
            tokens = encoding.encode(text)
            truncated_tokens = tokens[:max_tokens]
            return encoding.decode(truncated_tokens)

        # No encoding found - fallback to character-based truncation
        max_chars = max_tokens * CHARS_PER_TOKEN
        return text[:max_chars]
    except Exception:
        # Fallback to character-based truncation
        max_chars = max_tokens * CHARS_PER_TOKEN
        return text[:max_chars]
