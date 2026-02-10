"""Utility functions for CLI validation and configuration.

This module provides helper functions for validating and configuring AI provider
settings, including model selection and API key validation. It ensures that users
have properly configured providers and credentials before executing CLI commands.
"""

import os
from typing import Literal, Optional, Tuple

from pydantic import BaseModel, Field, field_validator
from rich.console import Console
from rich.panel import Panel

from allos.providers.registry import OPENAI_COMPATIBLE_PROVIDERS, ProviderRegistry
from allos.providers.utils import ollama_running


class ValidationResult(BaseModel):
    """Validation result for provider, model, and API key configuration."""

    success: bool = Field(..., description="Whether validation was successful")
    model: Optional[str] = Field(None, description="Name of the model to use")
    model_defaulted: bool = Field(
        False, description="Whether the default model name was used"
    )
    error: Optional[str] = Field(None, description="Error message if validation failed")
    error_type: Optional[Literal["model", "api_key"]] = Field(
        None, description="Type of validation error"
    )

    @field_validator("error_type")
    @classmethod
    def validate_error_type_consistency(cls, v: Optional[str], info) -> Optional[str]:
        """Ensure error_type is only set when success is False."""
        if info.data.get("success") and v is not None:
            raise ValueError("error_type must be None when success is True")
        if not info.data.get("success") and v is None:
            raise ValueError("error_type must be set when success is False")
        return v

    @field_validator("error")
    @classmethod
    def validate_error_message(cls, v: Optional[str], info) -> Optional[str]:
        """Ensure error message is only set when success is False."""
        if info.data.get("success") and v is not None:
            raise ValueError("error must be None when success is True")
        if not info.data.get("success") and not v:
            raise ValueError("error message required when success is False")
        return v


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
    elif provider == "google":
        default_model = "gemini-2.5-flash-lite"
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
        - env_var_or_error_msg: Clean env var name (e.g., "ANTHROPIC_API_KEY")
            or error description for special cases.
    """
    if api_key:
        return (True, "")

    # Check if provider requires auth at all
    if provider in OPENAI_COMPATIBLE_PROVIDERS:
        config = OPENAI_COMPATIBLE_PROVIDERS[provider]
        if not config.get("requires_auth", True):
            return (True, "")  # No API key required

    # Special case: Native Ollama doesn't need API key, just needs to be running
    if provider == "ollama":
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        if ollama_running(host):
            return (True, "")
        return (False, f"Ollama server not running at {host}")

    # Get env var name
    env_var = ProviderRegistry.get_env_var_name(provider)

    if env_var and env_var in os.environ:
        return (True, "")

    # Return just the clean env var name if available
    if env_var:
        return (False, env_var)

    # For complex providers (Google) without a simple env var
    is_configured, message = ProviderRegistry.check_provider_env(provider)
    if is_configured:
        return (True, "")

    return (False, message)


def validate_model_and_api_key(
    provider: str,
    model: Optional[str],
    api_key: Optional[str],
) -> ValidationResult:
    """Validate both model and API key configurations.

    Performs comprehensive validation of the provider, model, and API key.
    Determines appropriate defaults and checks for required environment variables.

    Args:
        provider: The name of the AI provider.
        model: The model name specified by the user, if any.
        api_key: The API key provided by the user, if any.

    Returns:
        ValidationResult object containing validation status and details.

    Raises:
        ValueError: If validation result is internally inconsistent.
    """
    # Validate model
    determined_model = determine_model(provider, model)

    if determined_model is None:
        return ValidationResult(
            success=False,
            model=None,
            model_defaulted=False,
            error=f"Model must be specified for {provider}",
            error_type="model",
        )

    # Validate API key
    api_key_valid, api_error = validate_api_key(provider, api_key)

    if not api_key_valid:
        return ValidationResult(
            success=False,
            model=determined_model,
            model_defaulted=(model is None),
            error=api_error,
            error_type="api_key",
        )

    return ValidationResult(
        success=True,
        model=determined_model,
        model_defaulted=(model is None),
        error=None,
        error_type=None,
    )


def display_validation_error(
    validation_result: ValidationResult,
    provider: str,
    console: Console,
) -> None:
    """Display validation errors in Rich Console Panel."""
    if validation_result.error_type == "model":
        console.print(
            Panel(
                f"[bold red]Model Configuration Error[/]\n\n"
                f"{validation_result.error}\n\n"
                f"Use the [bold cyan]--model[/] option to specify a model.",
                title="⚠️   Configuration Error",
                border_style="red",
            )
        )

    elif validation_result.error_type == "api_key" and validation_result.error:
        # Special handling for different provider types
        if provider == "google":
            message = (
                f"[bold red]Authentication Error[/]\n\n"
                f"{validation_result.error}\n\n"
                f"See documentation for Google provider setup."
            )
        elif "not running" in validation_result.error.lower():
            # Ollama special case
            message = (
                f"[bold red]Service Not Available[/]\n\n"
                f"{validation_result.error}\n\n"
                f"Start Ollama with: [bold cyan]ollama serve[/]"
            )
        else:
            # Standard API key error
            message = (
                f"[bold red]API Key Not Found[/]\n\n"
                f"The [bold cyan]{validation_result.error}[/] environment variable is not set.\n\n"
                f"Set it with: [bold cyan]export {validation_result.error}=your_key[/]\n"
                f"Or use: [bold cyan]--api-key your_key[/]"
            )
        console.print(
            Panel(message, title="⚠️   Configuration Error", border_style="red")
        )


def display_provider_info(
    validation_result: ValidationResult,
    provider: str,
    console: Console,
    stream: bool = False,
) -> None:
    """Display provider/model info message."""
    model = validation_result.model
    defaulted = validation_result.model_defaulted

    # Build consolidated message
    model_suffix = " (streaming)" if stream else ""
    default_prefix = " [dim](default)[/]" if defaulted else ""

    console.print(
        f"[dim]Using [cyan]{provider}[/] with model "
        f"[cyan]{model}[/]{default_prefix}{model_suffix}[/]"
    )
