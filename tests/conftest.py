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

# This import will now work because we created the placeholder files
from allos.providers.base import BaseProvider, ProviderResponse, ToolCall
from allos.tools.base import BaseTool


def pytest_addoption(parser):
    """Adds the --run-integration command-line option to pytest."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests (requires OPENAI_API_KEY).",
    )


def run_integration_tests(func):
    """Decorator to mark tests as integration tests."""
    return pytest.mark.integration(func)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration (requires --run-integration and OPENAI_API_KEY)",
    )


def pytest_collection_modifyitems(config, items):
    """Skip integration tests unless explicitly enabled and key provided."""
    run_integration = config.getoption("--run-integration")
    api_key = os.getenv("OPENAI_API_KEY")

    if run_integration and api_key:
        return

    skip_marker = pytest.mark.skip(
        reason="Requires --run-integration flag and OPENAI_API_KEY environment variable"
    )

    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_marker)


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
