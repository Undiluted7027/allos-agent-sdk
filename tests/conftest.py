# tests/conftest.py

import logging
import os
import unittest.mock as mock
from pathlib import Path
from typing import Any, Callable, Generator, Union

import pytest
from _pytest.logging import LogCaptureFixture

# Import this for better type hinting with the mocker fixture
from pytest_mock import MockerFixture

from allos.providers.base import BaseProvider, ProviderResponse, ToolCall
from allos.tools.base import BaseTool


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
        "markers", "requires_openai: marks tests as requiring an OpenAI API key"
    )
    config.addinivalue_line(
        "markers", "requires_anthropic: marks tests as requiring an Anthropic API key"
    )


def _skip_integration_tests(items):
    """Skip integration tests unless explicitly requested."""
    skip_marker = pytest.mark.skip(
        reason="Integration tests require the --run-integration flag"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_marker)


def _select_tests_by_flag(items, run_e2e, run_integration):
    """Return the list of tests to run based on given flags."""
    selected, deselected = [], []

    for item in items:
        is_e2e = "e2e" in item.keywords
        is_integration = "integration" in item.keywords

        if run_e2e and is_e2e:
            selected.append(item)
        elif run_integration and is_integration:
            selected.append(item)
        else:
            deselected.append(item)

    return selected


def _apply_integration_key_skips(items):
    """Skip integration tests that require missing API keys."""
    missing_keys = {
        "requires_openai": "OPENAI_API_KEY",
        "requires_anthropic": "ANTHROPIC_API_KEY",
    }

    for item in items:
        if "integration" not in item.keywords:
            continue
        for marker_name, key_name in missing_keys.items():
            if marker_name in item.keywords and not os.getenv(key_name):
                item.add_marker(
                    pytest.mark.skip(
                        reason=f"Requires the '{key_name}' environment variable"
                    )
                )


def pytest_collection_modifyitems(config, items):
    """
    Selects or skips tests based on custom command-line flags.

    - If no flags are given, runs unit and e2e tests (skips integration).
    - If --run-e2e is given, runs ONLY e2e tests.
    - If --run-integration is given, runs ONLY integration tests and
      provides skip messages if required API keys are missing.
    """
    run_e2e = config.getoption("--run-e2e")
    run_integration = config.getoption("--run-integration")

    if not run_e2e and not run_integration:
        _skip_integration_tests(items)
        return

    selected = _select_tests_by_flag(items, run_e2e, run_integration)

    if run_integration:
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
    monkeypatch.setenv("OPENAI_API_KEY", TEST_OPENAI_API_KEY)
    monkeypatch.setenv("ANTHROPIC_API_KEY", TEST_ANTHROPIC_API_KEY)


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
    mocker: MockerFixture,
) -> Callable[..., mock.MagicMock]:
    """
    Pytest fixture that returns a factory for creating mock LLM providers.
    This allows tests to simulate LLM responses without making real API calls.
    """

    def _create_mock_provider(
        response_content: str = "",
        tool_calls: Union[list[dict[str, Any]], None] = None,
    ) -> mock.MagicMock:
        mock_provider: mock.MagicMock = mocker.MagicMock(spec=BaseProvider)

        # Create typed ToolCall objects for a more realistic mock
        parsed_tool_calls = []
        if tool_calls:
            for i, tc in enumerate(tool_calls):
                parsed_tool_calls.append(
                    ToolCall(
                        id=tc.get("id", f"call_{i}"),
                        name=tc.get("name", "unknown_tool"),
                        arguments=tc.get("arguments", {}),
                    )
                )

        mock_response = ProviderResponse(
            content=response_content, tool_calls=parsed_tool_calls
        )
        mock_provider.chat.return_value = mock_response
        return mock_provider

    return _create_mock_provider


@pytest.fixture
def mock_tool_factory(mocker: MockerFixture) -> Callable[..., mock.MagicMock]:
    """
    Pytest fixture that returns a factory for creating mock tools.
    This allows tests to simulate tool execution.
    """

    def _create_mock_tool(
        name: str,
        result: Any,
        side_effect: Union[Exception, None] = None,
    ) -> mock.MagicMock:
        mock_tool: mock.MagicMock = mocker.MagicMock(spec=BaseTool)
        mock_tool.name = name
        if side_effect:
            mock_tool.execute.side_effect = side_effect
        else:
            mock_tool.execute.return_value = result
        return mock_tool

    return _create_mock_tool


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
