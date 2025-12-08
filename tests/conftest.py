# tests/conftest.py

import json
import logging
import os
import unittest.mock as mock
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Generator, Union, cast

import pytest
from _pytest.logging import LogCaptureFixture
from pydantic import BaseModel

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
from allos.tools.base import BaseTool
from allos.utils.token_counter import count_tokens


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


# @pytest.fixture(autouse=True)
# def mock_api_keys(monkeypatch):
#     """
#     Automatically mock API keys for all tests to bypass CLI checks.
#     Comment this function to use actual keys for E2E tests.
#     """
#     TEST_OPENAI_API_KEY = os.getenv("TEST_OPENAI_API_KEY", "test-openai-api-key")
#     TEST_ANTHROPIC_API_KEY = os.getenv(
#         "TEST_ANTHROPIC_API_KEY", "test-anthropic-api-key"
#     )
#     monkeypatch.setenv("OPENAI_API_KEY", TEST_OPENAI_API_KEY)
#     monkeypatch.setenv("ANTHROPIC_API_KEY", TEST_ANTHROPIC_API_KEY)


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
        # Define the baseline structure with proper types
        base_metadata: dict[str, Any] = {
            "status": "success",
            "model": ModelInfo(
                provider="mock",
                model_id="mock-model",
                configuration=ModelConfiguration(max_output_tokens=8192),
            ),
            "usage": Usage(input_tokens=0, output_tokens=0, total_tokens=0),
            "latency": Latency(total_duration_ms=100),
            "tools": ToolInfo(tools_available=[]),
            "quality_signals": QualitySignals(),
            "provider_specific": ProviderSpecific(),
            "sdk": SdkInfo(sdk_version="test"),
        }

        # Allow overriding any field
        for key, value in kwargs.items():
            # Special handling for nested Pydantic models
            if (
                key in base_metadata
                and isinstance(base_metadata[key], BaseModel)
                and isinstance(value, dict)
            ):
                original_model = cast(BaseModel, base_metadata[key])
                updated_model = original_model.model_copy(update=value)
                base_metadata[key] = updated_model
            else:
                base_metadata[key] = value

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
