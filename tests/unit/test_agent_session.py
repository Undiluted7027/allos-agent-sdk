# tests/unit/test_agent_session.py

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from allos.agent import Agent, AgentConfig
from allos.context import ConversationContext
from allos.utils.errors import AllosError


@pytest.fixture
def mock_get_tool(mocker):
    """Fixture to mock ToolRegistry.get_tool."""
    return mocker.patch("allos.agent.agent.ToolRegistry.get_tool")


class TestAgentSession:
    """Tests the successful save and load paths of the Agent."""

    @pytest.fixture
    def agent_instance(self, mock_get_provider, mock_get_tool):
        """Provides a basic agent instance with some context."""
        config = AgentConfig(
            provider_name="test-provider",
            model="test-model",
            tool_names=["test_tool"],
            auto_approve=True,
        )
        context = ConversationContext()
        context.add_user_message("Hello")
        return Agent(config, context)

    @patch("rich.console.Console.print")
    def test_save_session_success_path(
        self, mock_print, agent_instance, work_dir: Path
    ):
        """
        Test the successful save of a session, covering the json.dump and print calls.
        """
        session_file = work_dir / "session.json"
        agent_instance.save_session(session_file)

        # 1. Assert that the file was created and contains the correct data.
        assert session_file.exists()
        with open(session_file, "r") as f:
            data = json.load(f)

        assert data["config"]["provider_name"] == "test-provider"
        assert data["config"]["auto_approve"] is True
        assert len(data["context"]["messages"]) == 1
        assert data["context"]["messages"][0]["content"] == "Hello"

        # 2. Assert that the success message was printed to the console.
        mock_print.assert_any_call("[green]✅ Session saved successfully.[/green]")

    @patch("allos.agent.agent.ProviderRegistry.get_provider")
    @patch("allos.agent.agent.ToolRegistry.get_tool")
    def test_load_session_success_path(
        self, mock_get_tool, mock_get_provider, work_dir: Path
    ):
        """
        Test the successful loading of a session, covering the data extraction
        and object instantiation.
        """
        session_file = work_dir / "session.json"
        session_data = {
            "config": {
                "provider_name": "loaded-provider",
                "model": "loaded-model",
                "tool_names": ["loaded_tool"],
                "max_iterations": 5,
                "auto_approve": True,
            },
            "context": {"messages": [{"role": "user", "content": "Test"}]},
        }
        session_file.write_text(json.dumps(session_data))

        # --- Act ---
        loaded_agent = Agent.load_session(session_file)

        # --- Assert ---
        # 1. Assert that a new agent was created with the correct config.
        assert isinstance(loaded_agent, Agent)
        assert loaded_agent.config.provider_name == "loaded-provider"
        assert loaded_agent.config.max_iterations == 5
        assert loaded_agent.config.auto_approve is True

        # 2. Assert that the context was loaded correctly.
        assert len(loaded_agent.context) == 1
        assert loaded_agent.context.messages[0].content == "Test"

        # 3. Assert that the provider and tools were re-initialized.
        mock_get_provider.assert_called_once_with(
            "loaded-provider", model="loaded-model"
        )
        mock_get_tool.assert_called_once_with("loaded_tool")


class TestAgentSessionErrors:
    @pytest.fixture
    def agent_instance(self, mock_get_provider, mock_get_tool):
        """Provides a basic agent instance for session tests."""
        config = AgentConfig(provider_name="test", model="test")
        return Agent(config)

    @pytest.mark.parametrize(
        "exception_to_raise",
        [
            pytest.param(IOError("Permission denied"), id="IOError_on_save"),
            pytest.param(TypeError("Cannot serialize object"), id="TypeError_on_save"),
        ],
    )
    @patch("builtins.open")
    def test_save_session_raises_allos_error_on_failure(
        self, mock_open, exception_to_raise, agent_instance, work_dir
    ):
        """Test that save_session wraps IO/Type errors in AllosError."""
        # Configure the mock to raise an error when writing
        mock_open.side_effect = exception_to_raise

        filepath = work_dir / "session.json"

        with pytest.raises(AllosError) as excinfo:
            agent_instance.save_session(filepath)

        assert f"Failed to save session to '{filepath}'" in str(excinfo.value)

    @pytest.mark.parametrize(
        "file_content, exception_type, error_msg",
        [
            pytest.param(
                None, FileNotFoundError, "Failed to load session", id="file_not_found"
            ),
            pytest.param(
                '{"config": {}, "context": {}',
                json.JSONDecodeError,
                "Failed to load session",
                id="json_decode_error",
            ),
            pytest.param(
                '{"no_config_key": {}, "context": {}}',
                KeyError,
                "Failed to load session",
                id="key_error",
            ),
        ],
    )
    def test_load_session_raises_allos_error_on_failure(
        self, file_content, exception_type, error_msg, work_dir
    ):
        """Test that load_session wraps various errors in AllosError."""
        filepath = work_dir / "session.json"

        if file_content is not None:
            filepath.write_text(file_content)

        with pytest.raises(AllosError) as excinfo:
            Agent.load_session(filepath)

        assert error_msg in str(excinfo.value)
        # Check that the original exception is chained
        assert isinstance(excinfo.value.__cause__, exception_type)
