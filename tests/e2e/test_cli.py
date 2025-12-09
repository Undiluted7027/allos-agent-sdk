# tests/e2e/test_cli.py
import importlib  # noqa: F401
import json
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from allos.agent import Agent, AgentConfig
from allos.cli.main import main
from allos.providers.base import ProviderChunk
from allos.utils.errors import AllosError

pytestmark = pytest.mark.e2e


# --- Fixtures ---


@pytest.fixture
def runner() -> CliRunner:
    """Provides a CliRunner instance for invoking the CLI."""
    return CliRunner()


@pytest.fixture
def mock_agent_and_load_session(mocker):
    """
    Mocks the Agent class, its instance, and its load_session classmethod
    across all relevant modules where it is imported.
    """
    # Create a mock instance that has the necessary attributes
    mock_instance = MagicMock(spec=Agent)
    mock_instance.config = AgentConfig(
        provider_name="mock", model="mock"
    )  # Give it a config

    # Patch the Agent class in both modules where it is imported and used
    patcher_main = mocker.patch("allos.cli.main.Agent", return_value=mock_instance)
    patcher_interactive = mocker.patch(
        "allos.cli.interactive.Agent", return_value=mock_instance
    )

    # Also patch the classmethod on both mocked classes
    patcher_main.load_session.return_value = mock_instance
    patcher_interactive.load_session.return_value = mock_instance

    # Yield the class and instance mocks for assertions
    yield {
        "class_main": patcher_main,
        "class_interactive": patcher_interactive,
        "instance": mock_instance,
    }


# --- Test Cases ---


def test_cli_help(runner: CliRunner):
    """Test that the main --help command works."""
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Usage: allos [OPTIONS] [PROMPT]..." in result.output
    assert "--list-providers" in result.output


def test_list_providers_command(runner: CliRunner):
    """Test the `list-providers` command."""
    result = runner.invoke(main, ["--list-providers"])
    assert result.exit_code == 0
    assert "openai" in result.output
    assert "anthropic" in result.output


def test_list_tools_command(runner: CliRunner):
    """Test the `list-tools` command."""
    result = runner.invoke(main, ["--list-tools"])
    assert result.exit_code == 0
    assert "read_file" in result.output
    assert "shell_exec" in result.output
    assert "(ask_user)" in result.output  # Check for permission display


def test_active_providers_command(runner: CliRunner, monkeypatch):
    """
    Test the --active-providers command prints a table with correct statuses.
    This covers the if/elif/else branches in print_active_providers.
    """
    # 1. Set a key for one provider (Ready)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    # 2. Ensure another provider key is unset (Missing Key)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    result = runner.invoke(main, ["--active-providers"])
    assert result.exit_code == 0

    # Check for table headers and content
    assert "Active Providers Configuration" in result.output
    # OpenAI should be Ready
    assert "openai" in result.output
    assert "Ready" in result.output

    # Anthropic should be Missing Key (if it appears in list)
    # Note: Rich tables might wrap or format differently, but the text should be present
    assert "anthropic" in result.output
    # ollama_compat should be Manual Config
    assert "ollama_compat" in result.output
    assert "Manual Config Required" in result.output


def test_run_command_max_tokens(runner: CliRunner, mock_agent_and_load_session):
    """Test that --max-tokens is parsed and passed to config."""
    mock_agent_class = mock_agent_and_load_session["class_main"]

    result = runner.invoke(main, ["run", "prompt", "--max-tokens", "123"])

    assert result.exit_code == 0
    mock_agent_class.assert_called_once()
    config = mock_agent_class.call_args.args[0]
    assert config.max_tokens == 123


def test_run_command_no_tools(runner: CliRunner, mock_agent_and_load_session):
    """Test that --no-tools sets the flag in config."""
    mock_agent_class = mock_agent_and_load_session["class_main"]

    result = runner.invoke(main, ["run", "prompt", "--no-tools"])

    assert result.exit_code == 0
    mock_agent_class.assert_called_once()
    config = mock_agent_class.call_args.args[0]
    assert config.no_tools is True


def test_run_command_base_url_and_api_key(
    runner: CliRunner, mock_agent_and_load_session
):
    """Test that base-url and api-key flags are passed to config."""
    mock_agent_class = mock_agent_and_load_session["class_main"]

    result = runner.invoke(
        main,
        ["run", "prompt", "--base-url", "http://test.url", "--api-key", "secret-key"],
    )

    assert result.exit_code == 0
    config = mock_agent_class.call_args.args[0]
    assert config.base_url == "http://test.url"
    assert config.api_key == "secret-key"


def test_cli_no_arguments_shows_help(runner: CliRunner):
    """
    Test that running the command with no prompt or flags shows the help message.
    This covers the final `else` block in `main`.
    """
    result = runner.invoke(main, [])
    assert result.exit_code == 0
    assert "Usage: allos [OPTIONS] [PROMPT]..." in result.output
    assert "Run an agent to execute a given PROMPT." in result.output


@pytest.mark.parametrize(
    "mistyped_flag", ["interactive", "list-tools", "provider", "help"]
)
def test_cli_mistyped_flag_as_prompt_warns_and_exits(
    runner: CliRunner, mock_agent_and_load_session, mistyped_flag: str
):
    """
    Test that using a known flag name as a prompt shows an error and exits,
    preventing the agent from running.
    """
    mock_agent_instance = mock_agent_and_load_session["instance"]

    result = runner.invoke(main, [mistyped_flag])

    assert result.exit_code == 0
    assert "Did you mean to use" in result.output
    assert f"'--{mistyped_flag}'" in result.output
    mock_agent_instance.run.assert_not_called()


def test_single_word_prompt_aborted_by_user(
    runner: CliRunner, mock_agent_and_load_session
):
    """
    Test that a single-word prompt asks for confirmation and is aborted by the user.
    """
    mock_agent_instance = mock_agent_and_load_session["instance"]
    # The 'input' parameter simulates the user typing 'n' and pressing Enter.
    result = runner.invoke(main, ["hbdhbdvh"], input="n\n")

    assert result.exit_code == 0
    assert "Do you want to proceed?" in result.output
    assert "Aborted." in result.output
    mock_agent_instance.run.assert_not_called()


def test_single_word_prompt_confirmed_by_user(
    runner: CliRunner, mock_agent_and_load_session
):
    """
    Test that a single-word prompt asks for confirmation and is executed.
    """
    mock_agent_instance = mock_agent_and_load_session["instance"]
    # The 'input' parameter simulates the user typing 'y' and pressing Enter.
    result = runner.invoke(main, ["hbdhbdvh"], input="y\n")

    assert result.exit_code == 0
    assert "Do you want to proceed?" in result.output
    mock_agent_instance.run.assert_called_once_with("hbdhbdvh")


@pytest.mark.parametrize(
    "prompt_args",
    [
        ["multi", "word", "prompt"],  # Simulates `allos multi word prompt`
        ["multi word prompt"],  # Simulates `allos "multi word prompt"`
    ],
)
def test_multi_word_prompt_executes_directly(
    runner: CliRunner, mock_agent_and_load_session, prompt_args
):
    """
    Test that multi-word prompts (quoted or not) execute directly without confirmation.
    """
    mock_agent_instance = mock_agent_and_load_session["instance"]
    result = runner.invoke(main, prompt_args)

    assert result.exit_code == 0
    # The confirmation prompt should NOT be in the output
    assert "Do you want to proceed?" not in result.output
    mock_agent_instance.run.assert_called_once_with("multi word prompt")


class TestCliRunCommand:
    """Tests specifically for the `run` command."""

    def test_run_command_parsing(self, runner: CliRunner, mock_agent_and_load_session):
        """Test that the `run` command correctly parses all its arguments."""
        mock_agent_class = mock_agent_and_load_session["class_main"]
        mock_agent_instance = mock_agent_and_load_session["instance"]
        result = runner.invoke(
            main,
            [
                "--provider",
                "anthropic",
                "--model",
                "claude-test",
                "--tool",
                "read_file",
                "--tool",
                "write_file",
                "--auto-approve",
                "This is a test prompt.",
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0

        mock_agent_class.assert_called_once()
        config_arg = mock_agent_class.call_args.args[0]

        assert config_arg.provider_name == "anthropic"
        assert config_arg.model == "claude-test"
        assert set(config_arg.tool_names) == {"read_file", "write_file"}
        assert config_arg.auto_approve is True

        # Check that agent.run was called with the correct prompt
        mock_agent_instance.run.assert_called_once_with("This is a test prompt.")

    def test_run_command_missing_api_key(self, runner: CliRunner, monkeypatch):
        """Test that the CLI exits gracefully if an API key is missing."""
        # Temporarily remove the key from the environment for this test
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        result = runner.invoke(main, ["run", "test prompt"])

        assert (
            result.exit_code == 0
        )  # The command itself doesn't fail, it prints an error
        assert "Error" in result.output
        assert "API key not found" in result.output
        assert "Please set the " in result.output
        assert "OPENAI_API_KEY" in result.output

    def test_run_command_session_management(
        self, runner: CliRunner, mock_agent_and_load_session, work_dir: Path
    ):
        """Test that the --session flag correctly loads and saves sessions."""
        mock_agent_class = mock_agent_and_load_session["class_main"]
        mock_agent_instance = mock_agent_and_load_session["instance"]

        session_file = work_dir / "cli_session.json"
        session_file_str = str(session_file)

        # --- Test Loading ---
        dummy_config: Dict[str, Any] = {
            "provider_name": "openai",
            "model": "gpt-4o",
            "tool_names": [],
            "max_iterations": 10,
            "auto_approve": False,
        }
        dummy_context: Dict[str, List[Dict[str, Any]]] = {"messages": []}
        session_file.write_text(
            json.dumps({"config": dummy_config, "context": dummy_context})
        )

        runner.invoke(main, ["--session", session_file_str, "run", "prompt"])

        mock_agent_class.load_session.assert_called_once_with(session_file_str)
        mock_agent_instance.save_session.assert_called_once_with(session_file_str)

        # --- Test Saving (when file doesn't exist initially) ---
        mock_agent_class.load_session.reset_mock()
        mock_agent_instance.save_session.reset_mock()

        new_session_file = work_dir / "new_session.json"
        new_session_file_str = str(new_session_file)

        runner.invoke(main, ["--session", new_session_file_str, "run", "prompt"])

        mock_agent_class.load_session.assert_not_called()
        mock_agent_class.assert_called_once()
        mock_agent_instance.save_session.assert_called_once_with(new_session_file_str)

    def test_run_command_tools_override_session(
        self, runner: CliRunner, mock_agent_and_load_session, work_dir: Path
    ):
        """Test that --tool flags override the tools from a loaded session."""
        mock_agent_instance = mock_agent_and_load_session["instance"]
        mock_agent_instance.config.tool_names = [
            "read_file"
        ]  # Config from loaded session

        session_file = work_dir / "session.json"
        session_file.touch()

        runner.invoke(
            main,
            [
                "--session",
                str(session_file),
                "--tool",
                "write_file",  # New tool from CLI
                "run",
                "prompt",
            ],
        )

        # Check that the config on the loaded instance was updated
        assert mock_agent_instance.config.tool_names == ["write_file"]

    def test_run_command_session_management_with_overrides(
        self, runner: CliRunner, mock_agent_and_load_session, work_dir: Path
    ):
        """
        Test that loading a session AND providing new flags correctly updates the config.
        This covers all the 'if' branches in _update_agent_config.
        """
        mock_agent_class = mock_agent_and_load_session["class_main"]
        mock_agent_instance = mock_agent_and_load_session["instance"]

        # Prepare a dummy session file
        session_file = work_dir / "cli_session.json"
        dummy_config = {"provider_name": "openai", "model": "old", "tool_names": []}
        session_file.write_text(json.dumps({"config": dummy_config, "context": {}}))

        # Run with ALL override flags
        runner.invoke(
            main,
            [
                "--session",
                str(session_file),
                "run",
                "prompt",
                "--base-url",
                "http://new.url",
                "--api-key",
                "new-key",
                "--max-tokens",
                "99",
                "--no-tools",
            ],
        )

        # Verify load_session was called
        mock_agent_class.load_session.assert_called_once_with(str(session_file))

        # Verify the instance attributes were updated (via _update_agent_config)
        assert mock_agent_instance.config.base_url == "http://new.url"
        assert mock_agent_instance.config.api_key == "new-key"
        assert mock_agent_instance.config.max_tokens == 99
        assert mock_agent_instance.config.no_tools is True
        assert mock_agent_instance.tools == []

    def test_run_command_auto_approve_shows_warning(
        self, runner: CliRunner, mock_agent_and_load_session
    ):
        """Test that --auto-approve prints a warning to the console."""
        mock_agent_instance = mock_agent_and_load_session["instance"]
        mock_agent_instance.config.auto_approve = True
        result = runner.invoke(main, ["--auto-approve", "this is a prompt"])

        assert result.exit_code == 0
        assert "Auto-approve is enabled" in result.output

        # # Verify the config was set correctly
        # mock_agent_class.assert_called_once()
        # config_arg = mock_agent_class.call_args.args[0]
        # assert config_arg.auto_approve is True

    @pytest.mark.parametrize(
        "exception, title",
        [
            (AllosError("Test agent error"), "Agent Error"),
            (Exception("Generic test error"), "System Error"),
        ],
    )
    def test_run_command_handles_exceptions(
        self, runner: CliRunner, mock_agent_and_load_session, exception, title
    ):
        """Test that errors from agent.run() are caught and displayed."""
        mock_agent_instance = mock_agent_and_load_session["instance"]
        mock_agent_instance.run.side_effect = exception

        result = runner.invoke(main, ["run", "prompt"])

        assert result.exit_code == 0  # CLI should exit gracefully
        assert title in result.output
        assert str(exception) in result.output

    @pytest.mark.order(-1)  # Runs last to avoid corrupting the shared state
    def test_dotenv_import_error_is_handled(self, runner: CliRunner, monkeypatch):
        """
        Test that the CLI runs without error if python-dotenv is not installed
        by forcing a module reload after patching sys.modules.
        """
        # Step 1: Force Python to forget that 'dotenv' was ever imported.
        monkeypatch.setitem(sys.modules, "dotenv", None)

        # Step 2: Force Python to forget that our CLI module was ever imported.
        # This is CRITICAL. It ensures the module will be fully re-executed from the top.
        if "allos.cli.main" in sys.modules:
            monkeypatch.delitem(sys.modules, "allos.cli.main")

        # Step 3: Now, import the module. Since we've cleared the cache,
        # Python will re-run the file from the top. The `from dotenv import...`
        # line will be executed, and because 'dotenv' is now None in sys.modules,
        # it will raise an ImportError, which is then caught by the 'except' block.
        from allos.cli.main import main as reloaded_main

        # Step 4: Run the reloaded main function.
        result = runner.invoke(reloaded_main, ["--help"])

        # Step 5: Assert that the program ran successfully, proving the 'except' block was hit.
        assert result.exit_code == 0
        assert "Usage: allos" in result.output


class TestCliInteractiveCommand:
    """Tests specifically for the `interactive` command and its helpers."""

    @patch("allos.cli.interactive._run_repl_loop")
    def test_interactive_command_creates_new_agent(
        self, mock_repl, runner: CliRunner, mock_agent_and_load_session
    ):
        """Test that `interactive` creates a new agent when no session file exists."""
        mock_agent_class = mock_agent_and_load_session["class_interactive"]

        result = runner.invoke(
            main,
            [
                "--interactive",
                "--provider",
                "anthropic",
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        # Assert that a NEW agent was created via the constructor
        mock_agent_class.assert_called_once()
        # Assert that the load_session classmethod was NOT called
        mock_agent_class.load_session.assert_not_called()
        # Assert that the main REPL loop was entered
        mock_repl.assert_called_once()

    @patch("allos.cli.interactive._run_repl_loop")
    def test_interactive_command_loads_session(
        self, mock_repl, runner: CliRunner, mock_agent_and_load_session, work_dir: Path
    ):
        """Test that `interactive` loads an agent when a session file exists."""
        mock_agent_class = mock_agent_and_load_session["class_interactive"]

        # Create a dummy session file so the "load" path is taken
        session_file = work_dir / "interactive.json"
        session_file.touch()

        result = runner.invoke(
            main,
            [
                "--interactive",
                "--provider",
                "anthropic",
                "--session",
                str(session_file),
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        # Assert that the constructor was NOT called
        mock_agent_class.assert_not_called()
        # Assert that the load_session classmethod WAS called
        mock_agent_class.load_session.assert_called_once_with(str(session_file))
        # Assert that the main REPL loop was entered
        mock_repl.assert_called_once()

    @patch(
        "allos.cli.interactive._load_or_create_agent",
        side_effect=AllosError("Init failed"),
    )
    def test_interactive_initialization_error(
        self, mock_load_or_create, runner: CliRunner
    ):
        """Test that an error during agent init is caught and displayed."""
        result = runner.invoke(main, ["--interactive"])

        assert result.exit_code == 0
        assert "Failed to initialize agent" in result.output
        assert "Init failed" in result.output

    @patch("allos.cli.interactive._run_repl_loop")
    def test_interactive_tools_override_session(
        self,
        mock_repl,
        runner: CliRunner,
        mock_agent_and_load_session,
        work_dir: Path,
    ):
        """Test that --tool flags override the tools from a loaded session in interactive mode."""
        mock_agent_instance = mock_agent_and_load_session["instance"]
        mock_agent_instance.config.tool_names = ["read_file"]  # From loaded session

        session_file = work_dir / "session.json"
        session_file.touch()

        runner.invoke(
            main,
            ["--interactive", "--session", str(session_file), "--tool", "write_file"],
        )

        assert mock_agent_instance.config.tool_names == ["write_file"]

    @patch("allos.cli.interactive.console.input")
    def test_run_repl_loop(self, mock_input, mock_agent_and_load_session):
        """Test the REPL loop's input handling and exit commands."""
        mock_agent_instance = mock_agent_and_load_session["instance"]
        from allos.cli.interactive import _run_repl_loop

        # Simulate user typing "hello", "quit"
        mock_input.side_effect = ["hello", "quit"]

        _run_repl_loop(mock_agent_instance)

        # Agent.run should be called once with "hello"
        mock_agent_instance.run.assert_called_once_with("hello")

    @patch("allos.cli.interactive.console.print")  # Mock print to check output
    def test_save_session_error_handling(self, mock_print, mock_agent_and_load_session):
        """Test that an error during session save is caught and displayed."""
        mock_agent_instance = mock_agent_and_load_session["instance"]
        from rich.panel import Panel

        from allos.cli.interactive import _save_session

        mock_agent_instance.save_session.side_effect = AllosError("Disk full")

        _save_session(mock_agent_instance, "test.json")

        # Assert that `print` was called at least once.
        mock_print.assert_called()

        # Get the arguments of the first call to `print`.
        call_args, _ = mock_print.call_args

        # Ensure at least one argument was passed.
        assert len(call_args) > 0

        # The argument should be a Panel instance.
        printed_object = call_args[0]
        assert isinstance(printed_object, Panel)

        # Now, inspect the attributes of the Panel that was actually printed.
        assert printed_object.title == "Save Error"
        assert printed_object.border_style == "red"
        # We can check the content as a string.
        rendered = str(printed_object.renderable)
        assert "Failed to save session on exit" in rendered
        assert "Disk full" in rendered

    @patch("allos.cli.interactive.console.print")
    def test_print_panel(self, mock_print):
        """Test the _print_panel utility function."""
        from rich.panel import Panel

        from allos.cli.interactive import _print_panel

        _print_panel("Test message", "Test Title", "green")

        mock_print.assert_called_once()
        # Check that the argument to print was a Panel instance with the correct properties
        arg = mock_print.call_args.args[0]
        assert isinstance(arg, Panel)
        assert "Test message" in str(arg.renderable)
        assert arg.title == "Test Title"
        assert arg.border_style == "green"

    @patch("allos.cli.interactive.console.input")
    def test_run_repl_loop_ignores_empty_input(
        self, mock_input, mock_agent_and_load_session
    ):
        """Test that the REPL loop continues without action on empty/whitespace input."""
        mock_agent_instance = mock_agent_and_load_session["instance"]
        from allos.cli.interactive import _run_repl_loop

        # Simulate user hitting Enter, then typing spaces, then quitting
        mock_input.side_effect = ["", "   ", "exit"]

        _run_repl_loop(mock_agent_instance)

        # CRITICAL ASSERTION: The agent's run method should never have been called
        mock_agent_instance.run.assert_not_called()

    @pytest.mark.parametrize(
        "exception, expected_title",
        [
            (AllosError("Tool failed"), "Agent Error"),
            (ValueError("Unexpected value"), "System Error"),
        ],
    )
    @patch("allos.cli.interactive.console.input")
    @patch("allos.cli.interactive._print_panel")  # Target the specific helper
    def test_run_repl_loop_handles_exceptions(
        self,
        mock_print_panel,
        mock_input,
        exception,
        expected_title,
        mock_agent_and_load_session,
    ):
        """Test that exceptions during agent.run in the REPL are caught and displayed."""
        mock_agent_instance = mock_agent_and_load_session["instance"]
        from allos.cli.interactive import _run_repl_loop

        # Simulate user typing a prompt that will cause an error, then quitting
        mock_input.side_effect = ["do a failing task", "quit"]
        mock_agent_instance.run.side_effect = exception

        _run_repl_loop(mock_agent_instance)

        # Assert that our custom panel printer was called
        mock_print_panel.assert_called_once()
        # Inspect the arguments it was called with
        call_args, _ = mock_print_panel.call_args
        message, title, color = call_args

        assert str(exception) in message
        assert title == expected_title
        assert color == "red"

    @pytest.mark.parametrize("interrupt_exception", [KeyboardInterrupt, EOFError])
    @patch("allos.cli.interactive.console.input")
    def test_run_repl_loop_exits_gracefully_on_interrupt(
        self, mock_input, interrupt_exception, mock_agent_and_load_session
    ):
        """
        Test that the REPL loop breaks cleanly on KeyboardInterrupt or EOFError
        without calling the agent's run method.
        """
        mock_agent_instance = mock_agent_and_load_session["instance"]
        from allos.cli.interactive import _run_repl_loop

        # Configure the mock input to raise the specified exception when called
        mock_input.side_effect = interrupt_exception

        # Call the loop. We expect it to catch the exception and break,
        # so the test itself should not see the exception.
        _run_repl_loop(mock_agent_instance)

        # Assert that the input function was called once, triggering the exception
        mock_input.assert_called_once()

        # CRITICAL ASSERTION: Assert that the agent's run method was never called,
        # proving the loop broke before reaching that line.
        mock_agent_instance.run.assert_not_called()

    @patch("allos.cli.interactive._run_repl_loop")
    def test_interactive_command_auto_approve_shows_warning(
        self, mock_repl, runner: CliRunner, mock_agent_and_load_session
    ):
        """Test that the auto-approve warning is shown in interactive mode."""
        mock_agent_class = mock_agent_and_load_session["class_interactive"]
        mock_agent_instance = mock_agent_and_load_session["instance"]

        # Configure the mock instance BEFORE the CLI is invoked.
        # The instance returned by both Agent() and Agent.load_session() will have this config.
        mock_agent_instance.config.auto_approve = True

        result = runner.invoke(
            main,
            ["-i", "--auto-approve"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0
        # The config is passed during creation, so let's check that
        config_arg = mock_agent_class.call_args.args[0]
        assert config_arg.auto_approve is True

        # Now, check the output and the repl call
        assert "Auto-approve is enabled" in result.output
        mock_repl.assert_called_once()

    def test_interactive_overrides_session_config(
        self, runner: CliRunner, mock_agent_and_load_session, work_dir: Path
    ):
        """
        Test that interactive mode correctly overrides session config with CLI flags.
        This covers all branches in _override_agent_config.
        """
        mock_agent_instance = mock_agent_and_load_session["instance"]

        # 1. Setup a dummy session file
        session_file = work_dir / "session.json"
        session_file.touch()

        # 2. Run interactive with ALL override flags
        result = runner.invoke(
            main,
            [
                "--interactive",
                "--session",
                str(session_file),
                "--base-url",
                "http://override.url",
                "--api-key",
                "override-key",
                "--max-tokens",
                "500",
                "--no-tools",
                "--auto-approve",
            ],
            input="exit\n",  # Exit immediately
        )

        assert result.exit_code == 0

        # 3. Verify the overrides were applied to the agent config
        assert mock_agent_instance.config.base_url == "http://override.url"
        assert mock_agent_instance.config.api_key == "override-key"
        assert mock_agent_instance.config.max_tokens == 500
        assert mock_agent_instance.config.no_tools is True
        assert mock_agent_instance.config.auto_approve is True

        # Verify tools list was cleared (no_tools logic)
        assert mock_agent_instance.tools == []

    @patch("allos.cli.interactive._run_repl_loop")
    def test_interactive_no_tools_message(
        self, mock_repl, runner: CliRunner, mock_agent_and_load_session
    ):
        """Test that the 'Tools are disabled' message is printed."""
        mock_agent_instance = mock_agent_and_load_session["instance"]
        mock_agent_instance.config.no_tools = True

        result = runner.invoke(main, ["--interactive"])

        assert result.exit_code == 0
        assert "Tools are disabled for this session" in result.output


class TestCliStreamCommand:
    """Tests for the `--stream` flag and `run_agent_stream` function."""

    def test_main_dispatches_to_run_agent_stream(
        self, runner: CliRunner, mock_agent_and_load_session
    ):
        """Test that the --stream flag correctly calls run_agent_stream."""
        mock_agent_class = mock_agent_and_load_session["class_main"]
        mock_agent_instance = mock_agent_and_load_session["instance"]

        # Test with the flag
        result = runner.invoke(main, ["--stream", "This is a test prompt"])

        assert result.exit_code == 0

        mock_agent_class.assert_called_once()
        mock_agent_instance.stream_run.assert_called_once_with("This is a test prompt")
        mock_agent_instance.run.assert_not_called()
        assert "Streaming Response" in result.output

    def test_stream_command_all_chunk_types(
        self, runner: CliRunner, mock_agent_and_load_session
    ):
        """Test that run_agent_stream correctly prints different chunk types."""
        mock_agent_instance = mock_agent_and_load_session["instance"]
        mock_agent_instance.stream_run.return_value = iter(
            [
                ProviderChunk(content="Hello"),
                ProviderChunk(tool_call_start={"name": "search"}),
                ProviderChunk(content="World"),
                ProviderChunk(error="Something went wrong."),
            ]
        )

        result = runner.invoke(main, ["--stream", "This is a test prompt"])

        assert result.exit_code == 0
        mock_agent_instance.stream_run.assert_called_once_with("This is a test prompt")

        # Check the captured output for the formatted strings
        assert "Hello" in result.output
        assert "World" in result.output
        assert "Calling Tool" in result.output
        assert "Stream Error: Something went wrong." in result.output

    def test_stream_command_saves_session(
        self, runner: CliRunner, mock_agent_and_load_session, work_dir: Path
    ):
        """Test that --session flag works with --stream."""
        mock_agent_class = mock_agent_and_load_session["class_main"]
        mock_agent_instance = mock_agent_and_load_session["instance"]

        session_file = work_dir / "stream_session.json"

        result = runner.invoke(
            main, ["--stream", "--session", str(session_file), "This is a test prompt"]
        )

        assert result.exit_code == 0

        mock_agent_class.assert_called_once()
        mock_agent_instance.save_session.assert_called_once_with(str(session_file))

    @pytest.mark.parametrize(
        "exception, expected_output",
        [
            (AllosError("Stream failed"), "An error occurred:"),
            (ValueError("Unexpected stream error"), "An unexpected error occurred:"),
        ],
    )
    def test_stream_command_handles_exceptions(
        self, runner: CliRunner, mock_agent_and_load_session, exception, expected_output
    ):
        """Test that exceptions from stream_run are caught and printed."""
        mock_agent_instance = mock_agent_and_load_session["instance"]
        mock_agent_instance.stream_run.side_effect = exception

        result = runner.invoke(main, ["--stream", "This is a test prompt"])

        assert result.exit_code == 0
        assert expected_output in result.output
        assert str(exception) in result.output

    def test_stream_error_handling(
        self, runner: CliRunner, mock_agent_and_load_session
    ):
        """Test handling of stream errors."""
        mock_agent_instance = mock_agent_and_load_session["instance"]

        chunks = [
            ProviderChunk(content="Start"),
            ProviderChunk(error="Stream connection failed"),
        ]
        mock_agent_instance.stream_run.return_value = iter(chunks)

        result = runner.invoke(main, ["--stream", "This is a test prompt"])

        assert result.exit_code == 0
        assert "Start" in result.output
        assert "Stream Error: Stream connection failed" in result.output

    def test_stream_command_auto_approve_warning(
        self, runner: CliRunner, mock_agent_and_load_session
    ):
        """Test that the auto-approve warning is shown in stream mode."""
        mock_agent_instance = mock_agent_and_load_session["instance"]
        mock_agent_instance.config.auto_approve = True
        mock_agent_instance.stream_run.return_value = iter([])

        result = runner.invoke(
            main, ["--stream", "--auto-approve", "This is a test prompt"]
        )

        assert "Auto-approve is enabled" in result.output

    def test_stream_command_api_key_from_flag(
        self, runner: CliRunner, mock_agent_and_load_session
    ):
        """Test that validation succeeds when API key is passed via --api-key flag"""
        mock_agent_class = mock_agent_and_load_session["class_main"]
        mock_agent_instance = mock_agent_and_load_session["instance"]

        result = runner.invoke(
            main, ["--stream", "--api-key", "DUMMY_KEY", "This is a test prompt"]
        )

        assert result.exit_code == 0

        mock_agent_class.assert_called_once()
        mock_agent_instance.stream_run.assert_called_once_with("This is a test prompt")

    def test_stream_command_api_key_missing(
        self, runner: CliRunner, mock_agent_and_load_session, monkeypatch
    ):
        """Test that validation fails when API key and env var are both missing"""
        mock_agent_class = mock_agent_and_load_session["class_main"]
        mock_agent_instance = mock_agent_and_load_session["instance"]

        # Remove the API key environment variable if it exists
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        result = runner.invoke(
            main,
            [
                "--provider",
                "anthropic",
                "--model",
                "claude-test",
                "--stream",
                "This is a test prompt",
            ],
        )

        # Should exit successfully but not call the agent
        assert result.exit_code == 0

        # Verify error message was printed
        assert "Error:" in result.output
        assert "API key not found" in result.output
        assert "ANTHROPIC_API_KEY" in result.output

        # Agent should NOT be called because validation failed
        mock_agent_class.assert_not_called()
        mock_agent_instance.stream_run.assert_not_called()

    def test_stream_command_api_key_from_env(
        self, runner: CliRunner, mock_agent_and_load_session, monkeypatch
    ):
        """Test that validation succeeds when API key is in environment variable"""
        mock_agent_class = mock_agent_and_load_session["class_main"]
        mock_agent_instance = mock_agent_and_load_session["instance"]

        # Set the API key environment variable
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-from-env")

        result = runner.invoke(main, ["--stream", "This is a test prompt"])

        assert result.exit_code == 0

        # No error message should be printed
        assert "Error:" not in result.output
        assert "API key not found" not in result.output

        # Agent SHOULD be called
        mock_agent_class.assert_called_once()
        mock_agent_instance.stream_run.assert_called_once_with("This is a test prompt")
