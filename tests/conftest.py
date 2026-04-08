# tests/conftest.py

import json
import logging
import os
import sys
import unittest.mock as mock
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Generator, Union, cast

import pytest
from _pytest.logging import LogCaptureFixture

# Import this for better type hinting with the mocker fixture
from pytest_mock import MockerFixture

from allos.providers.base import BaseProvider, Message, ProviderResponse, ToolCall
from allos.providers.metadata import (
    Latency,
    Metadata,
    ModelConfiguration,
    ModelInfo,
    ProviderSpecific,
    QualitySignals,
    SdkInfo,
    ToolInfo,
    Usage,
)
from allos.providers.utils import ollama_running
from allos.tools.base import BaseTool
from allos.utils.token_counter import count_tokens

# Default models for providers
PROVIDER_MODELS = {
    "openai": "gpt-4o",
    "anthropic": "claude-haiku-4-5",
    "ollama": os.getenv("TEST_OLLAMA_MODEL", "qwen3:8b"),  # Reads env or defaults
    "chat_completions": "gpt-3.5-turbo",
    "google": "gemini-2.5-flash-lite",
    "cohere": "command-r7b-12-2024",
}

# This sets the env var before the coverage plugin finishes initialization
if sys.version_info < (3, 10):
    os.environ["OMIT_FOR_VERSION"] = "allos/providers/google.py"
else:
    os.environ["OMIT_FOR_VERSION"] = ""


def pytest_addoption(parser):
    """Adds command-line options for running specific test categories."""
    parser.addoption(
        "--run-e2e",
        action="store_true",
        default=False,
        help="Run only the end-to-end tests.",
    )
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run only the integration tests (requires API keys).",
    )
    parser.addoption(
        "--run-performance",
        action="store_true",
        default=False,
        help="Run only the performance tests (requires Ollama or other providers).",
    )


def run_e2e_tests(func):
    """Decorator to mark tests as end-to-end tests."""
    return pytest.mark.e2e(func)


def run_integration_tests(func):
    """Decorator to mark tests as integration tests."""
    return pytest.mark.integration(func)


def pytest_configure(config):
    """Registers custom markers for pytest."""
    config.addinivalue_line(
        "markers",
        "e2e: marks tests as end-to-end tests (mocked LLM, real tools/filesystem)",
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration (requires --run-integration and API keys)",
    )
    config.addinivalue_line(
        "markers",
        "performance: marks tests as performance tests (requires --run-performance)",
    )
    config.addinivalue_line(
        "markers", "requires_openai: marks tests as requiring an OpenAI API key"
    )
    config.addinivalue_line(
        "markers", "requires_anthropic: marks tests as requiring an Anthropic API key"
    )
    config.addinivalue_line(
        "markers", "requires_cohere: marks tests as requiring a Cohere API key"
    )
    config.addinivalue_line(
        "markers", "requires_ollama: marks tests as requiring ollama local client"
    )
    config.addinivalue_line(
        "markers", "requires_gemini: marks tests as requiring Gemini API key"
    )
    config.addinivalue_line(
        "markers",
        "requires_vertexai: marks tests as requiring Vertex AI authentication",
    )
    config.addinivalue_line("markers", "slow: marks tests as slow-running tests")
    config.addinivalue_line(
        "markers",
        "requires_python_310: marks tests requiring Python 3.10+ (Google provider)",
    )
    config.addinivalue_line(
        "markers",
        "skip_on_python_39: alias for requires_python_310",
    )


def _skip_integration_tests(items):
    """Skip integration tests unless explicitly requested."""
    skip_marker = pytest.mark.skip(
        reason="Integration tests require the --run-integration flag"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_marker)


def _skip_performance_tests(items):
    """Skip performance tests unless explicitly requested."""
    skip_marker = pytest.mark.skip(
        reason="Performance tests require the --run-performance flag"
    )
    for item in items:
        if "performance" in item.keywords:
            item.add_marker(skip_marker)


def _select_tests_by_flag(items, run_e2e, run_integration, run_performance):
    """Return the list of tests to run based on given flags."""
    selected, deselected = [], []

    for item in items:
        is_e2e = "e2e" in item.keywords
        is_integration = "integration" in item.keywords
        is_performance = "performance" in item.keywords

        if run_e2e and is_e2e:
            selected.append(item)
        elif run_integration and is_integration:
            selected.append(item)
        elif run_performance and is_performance:
            selected.append(item)
        else:
            deselected.append(item)

    return selected


def _check_vertexai_conf() -> bool:
    if os.getenv("GOOGLE_CLOUD_PROJECT"):
        return True

    # Check for service account file
    sa_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if sa_path and os.path.exists(sa_path):
        return True

    # Try to detect ADC creds
    try:
        import google.auth
        from google.auth.exceptions import DefaultCredentialsError

        _, project = google.auth.default()
        if project:
            return True
    except DefaultCredentialsError:
        pass
    return False


def _apply_integration_key_skips(items):
    """Skip integration tests that require missing API keys."""
    missing_keys = {
        "requires_openai": lambda: bool(os.getenv("OPENAI_API_KEY")),
        "requires_anthropic": lambda: bool(os.getenv("ANTHROPIC_API_KEY")),
        "requires_gemini": lambda: bool(os.getenv("GEMINI_API_KEY")),
        "requires_cohere": lambda: bool(os.getenv("COHERE_API_KEY")),
        "requires_ollama": ollama_running,
        "requires_vertexai": _check_vertexai_conf,
    }

    for item in items:
        if "integration" not in item.keywords and "performance" not in item.keywords:
            continue
        for marker_name, check in missing_keys.items():
            if marker_name in item.keywords and not check():
                item.add_marker(pytest.mark.skip(reason=f"{marker_name} not satisfied"))


def pytest_collection_modifyitems(config, items):
    """
    Selects or skips tests based on custom command-line flags.

    - If no flags are given, runs unit and e2e tests (skips integration and performance).
    - If --run-e2e is given, runs ONLY e2e tests.
    - If --run-integration is given, runs ONLY integration tests and
      provides skip messages if required API keys are missing.
    - If --run-performance is given, runs ONLY performance tests and
      provides skip messages if required services are not available.
    """
    run_e2e = config.getoption("--run-e2e")
    run_integration = config.getoption("--run-integration")
    run_performance = config.getoption("--run-performance")

    if not run_e2e and not run_integration and not run_performance:
        _skip_integration_tests(items)
        _skip_performance_tests(items)
        return

    selected = _select_tests_by_flag(items, run_e2e, run_integration, run_performance)

    if run_integration or run_performance:
        _apply_integration_key_skips(selected)

    items[:] = selected


@pytest.fixture(autouse=True)
def mock_api_keys(monkeypatch):
    """
    Automatically mock API keys for all tests to bypass CLI checks.
    Comment this function to use actual keys for E2E tests.
    """
    TEST_OPENAI_API_KEY = os.getenv("TEST_OPENAI_API_KEY", "test-openai-api-key")
    TEST_ANTHROPIC_API_KEY = os.getenv(
        "TEST_ANTHROPIC_API_KEY", "test-anthropic-api-key"
    )
    TEST_GEMINI_API_KEY = os.getenv("TEST_GEMINI_API_KEY", "test-gemini-api-key")
    TEST_GOOGLE_API_KEY = os.getenv("TEST_GOOGLE_API_KEY", "test-google-api-key")
    TEST_COHERE_API_KEY = os.getenv("TEST_COHERE_API_KEY", "test-cohere-api-key")
    TEST_OLLAMA_API_KEY = os.getenv("TEST_OLLAMA_API_KEY", "test-ollama-api-key")
    monkeypatch.setenv("OPENAI_API_KEY", TEST_OPENAI_API_KEY)
    monkeypatch.setenv("ANTHROPIC_API_KEY", TEST_ANTHROPIC_API_KEY)
    monkeypatch.setenv("GEMINI_API_KEY", TEST_GEMINI_API_KEY)
    monkeypatch.setenv("GOOGLE_API_KEY", TEST_GOOGLE_API_KEY)
    monkeypatch.setenv("COHERE_API_KEY", TEST_COHERE_API_KEY)
    monkeypatch.setenv("OLLAMA_API_KEY", TEST_OLLAMA_API_KEY)
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "")


@pytest.fixture
def work_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """
    Pytest fixture to create a temporary working directory for the agent.
    Each test function gets a unique, empty directory.
    """
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()
    original_cwd = Path.cwd()
    try:
        os.chdir(project_dir)
        yield project_dir
    finally:
        os.chdir(original_cwd)


@pytest.fixture
def configured_caplog(
    caplog: LogCaptureFixture,
) -> Generator[LogCaptureFixture, None, None]:
    """
    Fixture that configures caplog to capture DEBUG messages from the 'allos' logger.

    This allows tests to assert the content of DEBUG, INFO, WARNING, etc.,
    level logs emitted by the application.
    """
    # Use the context manager to temporarily set the log level for the 'allos' logger
    with caplog.at_level(logging.DEBUG, logger="allos"):
        yield caplog


@pytest.fixture
def mock_provider_factory(
    mocker: MockerFixture, mock_metadata_factory: Callable[..., Metadata]
) -> Callable[..., mock.MagicMock]:
    """
    Pytest fixture that returns a factory for creating mock LLM providers
    that dynamically generate metadata.
    """

    def _create_mock_provider(
        response_content: str = "",
        tool_calls: Union[list[ToolCall], None] = None,
    ) -> mock.MagicMock:
        mock_provider = mocker.MagicMock(spec=BaseProvider)

        def chat_side_effect(
            messages: list[Message], **kwargs: Any
        ) -> ProviderResponse:
            # 1. Calculate input tokens
            input_text = " ".join([msg.content or "" for msg in messages])
            input_tokens = count_tokens(input_text, model="gpt-4")

            # 2. Calculate output tokens from the mock response
            output_content = response_content or ""
            if tool_calls:
                # Approximate tokens for tool calls
                output_content += json.dumps([asdict(tc) for tc in tool_calls])
            output_tokens = count_tokens(output_content, model="gpt-4")

            # 3. Create the dynamic metadata
            dynamic_metadata = mock_metadata_factory(
                usage={
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                }
            )

            # 4. Return the complete ProviderResponse
            return ProviderResponse(
                content=response_content,
                tool_calls=tool_calls or [],
                metadata=dynamic_metadata,
            )

        mock_provider.chat.side_effect = chat_side_effect
        mock_provider.get_context_window.return_value = 32000
        return cast(mock.MagicMock, mock_provider)

    return _create_mock_provider


def get_available_provider_params():
    """
    Get pytest parametrize values for all available providers.

    Returns parameters as (provider_name, model_name), with Google conditionally skipped on Python 3.9.
    """
    import sys

    # Helper to construct param with model
    def make_param(name, marks, id_val, model_override=None):
        model = model_override or PROVIDER_MODELS.get(name)

        if model is None:
            raise ValueError(
                f"model was not provided or not configured as default for provider {name}."
            )
        return pytest.param(name, model, marks=marks, id=id_val)

    base_providers = [
        make_param("openai", pytest.mark.requires_openai, "openai"),
        make_param("anthropic", pytest.mark.requires_anthropic, "anthropic"),
        make_param("ollama", pytest.mark.requires_ollama, "ollama"),
        make_param("chat_completions", pytest.mark.requires_openai, "chat_completions"),
        make_param("cohere", pytest.mark.requires_cohere, "cohere"),
    ]

    if sys.version_info >= (3, 10):
        base_providers.append(
            make_param("google", pytest.mark.requires_gemini, "gemini")
        )
        base_providers.append(
            make_param("google", pytest.mark.requires_vertexai, "vertexai")
        )
    else:
        # Skip Google on older Python
        skip_google = [
            pytest.mark.skip("Google requires Python 3.10+"),
            pytest.mark.requires_gemini,
            pytest.mark.requires_vertexai,
        ]
        base_providers.append(make_param("google", skip_google, "google-py39-skip"))

    return base_providers


@pytest.fixture
def skip_if_python_39():
    """Skip test if running on Python 3.9."""
    import sys

    if sys.version_info < (3, 10):
        pytest.skip("Test requires Python 3.10+")


@pytest.fixture
def available_providers():
    """Return list of providers available on current Python version."""
    import sys

    providers = ["openai", "anthropic", "ollama", "chat_completions", "cohere"]
    if sys.version_info >= (3, 10):
        providers.append("google")
    return providers


@pytest.fixture
def mock_tool_factory(mocker: MockerFixture) -> Callable[..., BaseTool]:
    """
    Pytest fixture that returns a factory for creating mock tools.
    This allows tests to simulate tool execution.
    """

    def _create_mock_tool(
        name: str,
        result: Any,
        side_effect: Union[Exception, None] = None,
    ) -> BaseTool:
        mock_tool = mocker.MagicMock(spec=BaseTool)
        mock_tool.name = name
        if side_effect:
            mock_tool.execute.side_effect = side_effect
        else:
            mock_tool.execute.return_value = result
        return cast(BaseTool, mock_tool)

    return _create_mock_tool


@pytest.fixture
def mock_metadata_factory() -> Callable[..., Metadata]:
    """Provides a factory for creating a baseline, valid Metadata object for tests."""

    def _create_metadata(**kwargs: Any) -> Metadata:
        from allos.providers.metadata import (
            Latency,
            Metadata,
            ModelConfiguration,
            ModelInfo,
            ProviderSpecific,
            QualitySignals,
            SdkInfo,
            ToolInfo,
            Usage,
        )

        # Extract usage kwargs if provided
        usage_kwargs = kwargs.pop("usage", {})

        # Extract model shortcuts if provided
        provider = kwargs.pop("provider", "mock")
        model_id = kwargs.pop("model_id", "mock-model")

        # Define the baseline structure with proper types
        base_metadata: dict[str, Any] = {
            "status": "success",
            "model": ModelInfo(
                provider=provider,
                model_id=model_id,
                configuration=ModelConfiguration(max_output_tokens=8192),
            ),
            "usage": Usage(
                input_tokens=usage_kwargs.get("input_tokens", 10),
                output_tokens=usage_kwargs.get("output_tokens", 20),
                total_tokens=usage_kwargs.get("input_tokens", 10)
                + usage_kwargs.get("output_tokens", 20),
            ),
            "latency": Latency(total_duration_ms=100),
            "tools": ToolInfo(tools_available=[]),
            "quality_signals": QualitySignals(
                finish_reason="stop"
            ),  # String, not MagicMock
            "provider_specific": ProviderSpecific(),
            "sdk": SdkInfo(sdk_version="test"),
        }

        # Allow overriding any top-level field
        base_metadata.update(kwargs)

        return Metadata(**base_metadata)

    return _create_metadata


@pytest.fixture
def mock_provider_instance(mocker):
    """
    Provides a MagicMock of a BaseProvider instance with default
    configurations needed for agent tests (e.g., context window size).
    """
    mock_instance = mocker.MagicMock(spec=BaseProvider)

    # Configure the essential methods that agent tests will call
    mock_instance.get_context_window.return_value = 8192

    return mock_instance


@pytest.fixture
def mock_get_provider(mocker, mock_provider_instance):
    """
    Mocks ProviderRegistry.get_provider to return a pre-configured
    mock provider instance.
    """
    # This single patch will affect all calls to ProviderRegistry.get_provider
    # across the entire test suite.
    return mocker.patch(
        "allos.agent.agent.ProviderRegistry.get_provider",
        return_value=mock_provider_instance,
    )


@pytest.fixture
def mock_metadata() -> Metadata:
    """Provides a default, valid Metadata object for tests."""
    return Metadata(
        status="success",
        model=ModelInfo(
            provider="mock",
            model_id="mock-model",
            configuration=ModelConfiguration(max_output_tokens=8192),
        ),
        usage=Usage(),
        latency=Latency(total_duration_ms=100),
        tools=ToolInfo(tools_available=[]),
        quality_signals=QualitySignals(),
        provider_specific=ProviderSpecific(),
        sdk=SdkInfo(sdk_version="test"),
    )


# Default model for ollama integration tests
# If you don't have this model, change it to something
# that supports thinking and tool calling (and is pulled).
@pytest.fixture
def default_ollama_model():
    return "qwen3:8b"
