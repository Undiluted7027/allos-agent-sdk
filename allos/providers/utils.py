"""Utility functions for provider operations.

This module provides helper functions for provider connectivity and validation.
"""

from typing import cast

import requests


def ollama_running(OLLAMA_URL: str) -> bool:
    """Check if Ollama is running and accessible.

    Args:
        OLLAMA_URL: The URL of the Ollama server to check.

    Returns:
        True if Ollama is running and responding to requests, False otherwise.
    """
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=0.3)
        return cast(bool, r.status_code == 200)
    except requests.RequestException:
        return False
