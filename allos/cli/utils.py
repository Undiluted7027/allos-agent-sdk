"""Utility functions for CLI validation and configuration.

This module provides helper functions for validating and configuring AI provider
settings, including model selection and API key validation. It ensures that users
have properly configured providers and credentials before executing CLI commands.
"""

import os
from typing import Dict, Optional, Tuple, Union

from allos.providers.registry import ProviderRegistry
from allos.providers.utils import ollama_running


def determine_model(provider: str, model: Optional[str]) -> Optional[str]:
    """Select a default model if none is provided.

    Determines the appropriate default model for the given provider
    if no model is explicitly specified.

    Args:
        provider: The name of the AI provider (e.g., 'openai', 'anthropic').
        model: The model name specified by the user, if any.

    Returns:
        The model name to use: either the provided model or a provider-specific
        default. Returns None if no default exists for the provider.
    """
    if model is not None:
        return model
    default_model: Optional[str] = None
    if provider == "openai":
        default_model = "gpt-4o"
    elif provider == "anthropic":
        default_model = "claude-3-haiku-20240307"
    return default_model


def validate_api_key(provider: str, api_key: Optional[str]) -> Tuple[bool, str]:
    """Check if a valid API key is available for the provider.

    Validates that an API key exists either as a parameter or in the
    environment variables. Returns validation status and any missing
    environment variable name.

    Args:
        provider: The name of the AI provider.
        api_key: The API key provided by the user, if any.

    Returns:
        A tuple of (is_valid, missing_env_var) where:
        - is_valid: True if API key is available, False otherwise.
        - missing_env_var: Name of the environment variable if missing, empty string if valid.
    """
    if api_key:
        return (True, "")

    required_env_var = ProviderRegistry.get_env_var_name(provider)

    if required_env_var and required_env_var not in os.environ:
        if provider == "ollama":
            if not ollama_running("http://localhost:11434"):
                return (False, required_env_var)
            else:
                return (True, "")
        return (False, required_env_var)
    return (True, "")


def validate_model_and_api_key(
    provider: str,
    model: Optional[str],
    api_key: Optional[str],
) -> Dict[str, Dict[str, Union[bool, str]]]:
    """Validate both model and API key configurations.

    Performs comprehensive validation of the provider, model, and API key.
    Determines appropriate defaults and checks for required environment variables.

    Args:
        provider: The name of the AI provider.
        model: The model name specified by the user, if any.
        api_key: The API key provided by the user, if any.

    Returns:
        A dictionary with validation results containing:
        - 'determined_model': dict with 'check' (bool), 'message' (str), and optionally 'model' (str).
        - 'validate_api_key': dict with 'check' (bool) and optionally 'message' (str).
        Each check includes formatted error/info messages for CLI display.
    """
    result: Dict[str, Dict[str, Union[bool, str]]] = {}
    determined_model = determine_model(provider, model)
    if determined_model is None:
        result["determined_model"] = {
            "check": False,
            "message": f"[bold red]Error:[/] Model needs to be specified for {provider}. "
            "use the [bold]--model[/] option.",
        }
    else:
        if model is None:
            result["determined_model"] = {
                "check": True,
                "model": determined_model,
                "message": f"[dim]Model not specified, defaulting to '{determined_model}' for provider '{provider}'.[/dim]",
            }
        else:
            result["determined_model"] = {"check": True, "model": determined_model}
    api_key_validation = validate_api_key(provider, api_key)
    if not api_key_validation[0]:
        result["validate_api_key"] = {
            "check": False,
            "message": f"[bold red]Error:[/] API key not found. "
            f"Please set the [bold]{api_key_validation[1]}[/] environment variable "
            f"or use the [bold]--api-key[/] option.",
        }
    else:
        result["validate_api_key"] = {"check": True}

    return result
