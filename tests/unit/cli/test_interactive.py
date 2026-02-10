# tests/unit/cli/test_interactive.py

"""Tests for the interactive REPL session functionality."""

from unittest.mock import patch

from allos.cli.interactive import _execute_streaming, _handle_repl_command


class TestHandleReplCommand:
    """Tests for REPL command handling."""

    def test_handle_help_command(self):
        """Test /help command displays help panel.
        Covers line 275 in interactive.py.
        """
        streaming_state, end_flag = _handle_repl_command("/help", False, False)

        # Help command should not change streaming state or end flag
        assert streaming_state is None
        assert end_flag is False

    def test_handle_stream_status_command(self):
        """Test /stream command shows current status.
        Covers line 289 in interactive.py.
        """
        # Test with streaming enabled
        streaming_state, end_flag = _handle_repl_command("/stream", True, False)

        assert streaming_state is None
        assert end_flag is False

        # Test with streaming disabled
        streaming_state, end_flag = _handle_repl_command("/stream", False, False)

        assert streaming_state is None
        assert end_flag is False

    def test_handle_stream_on_command_when_disabled(self):
        """Test /stream on command enables streaming.
        Covers lines 300-302 in interactive.py.
        """
        # Streaming currently disabled (False)
        streaming_state, end_flag = _handle_repl_command("/stream on", False, False)

        # Should enable streaming
        assert streaming_state is True
        assert end_flag is False

    def test_handle_stream_on_command_when_already_enabled(self):
        """Test /stream on command when already enabled does nothing."""
        # Streaming currently enabled (True)
        streaming_state, end_flag = _handle_repl_command("/stream on", True, False)

        # Should not change state (already enabled)
        assert streaming_state is None
        assert end_flag is False

    def test_handle_stream_off_command_when_enabled(self):
        """Test /stream off command disables streaming.
        Covers lines 304-306 in interactive.py.
        """
        # Streaming currently enabled (True)
        streaming_state, end_flag = _handle_repl_command("/stream off", True, False)

        # Should disable streaming
        assert streaming_state is False
        assert end_flag is False

    def test_handle_stream_off_command_when_already_disabled(self):
        """Test /stream off command when already disabled does nothing."""
        # Streaming currently disabled (False)
        streaming_state, end_flag = _handle_repl_command("/stream off", False, False)

        # Should not change state (already disabled)
        assert streaming_state is None
        assert end_flag is False

    def test_handle_unknown_stream_subcommand(self):
        """Test /stream with unknown subcommand.
        Covers lines 308-310 in interactive.py.
        """
        streaming_state, end_flag = _handle_repl_command(
            "/stream unknown", False, False
        )

        # Should not change state
        assert streaming_state is None
        assert end_flag is False


class TestReplLoopBranches:
    """Tests for REPL loop execution branches."""

    @patch("allos.cli.interactive.console")
    def test_streaming_enabled_execution_branch(self, mock_console, mock_get_provider):
        """Test that streaming_enabled=True calls _execute_streaming.
        Covers line 249 in interactive.py (streaming branch).
        """
        from allos.agent import Agent, AgentConfig
        from allos.providers.base import ProviderChunk

        # Create agent config (will use mocked provider from conftest)
        config = AgentConfig(provider_name="openai", model="gpt-4o", no_tools=True)
        agent = Agent(config)

        # Mock the stream_run to return simple chunks
        mock_chunks = [
            ProviderChunk(content="Hello"),
            ProviderChunk(content=" World"),
        ]

        with patch.object(agent, "stream_run", return_value=iter(mock_chunks)):
            _execute_streaming(agent, "test prompt")

        # Verify console.print was called for content
        assert mock_console.print.called

    @patch("allos.cli.interactive.console")
    def test_streaming_tool_call_start(self, mock_console, mock_get_provider):
        """Test streaming execution with tool_call_start chunk.
        Covers line 328 in interactive.py.
        """
        from allos.agent import Agent, AgentConfig
        from allos.providers.base import ProviderChunk

        config = AgentConfig(provider_name="openai", model="gpt-4o", no_tools=True)
        agent = Agent(config)

        # Mock stream with tool call
        mock_chunks = [
            ProviderChunk(tool_call_start={"id": "call_1", "name": "read_file"}),
        ]

        with patch.object(agent, "stream_run", return_value=iter(mock_chunks)):
            _execute_streaming(agent, "test prompt")

        # Verify tool call message was printed
        assert mock_console.print.called

    @patch("allos.cli.interactive.console")
    def test_streaming_error_chunk(self, mock_console, mock_get_provider):
        """Test streaming execution with error chunk.
        Covers line 332 in interactive.py.
        """
        from allos.agent import Agent, AgentConfig
        from allos.providers.base import ProviderChunk

        config = AgentConfig(provider_name="openai", model="gpt-4o", no_tools=True)
        agent = Agent(config)

        # Mock stream with error
        mock_chunks = [
            ProviderChunk(error="Something went wrong"),
        ]

        with patch.object(agent, "stream_run", return_value=iter(mock_chunks)):
            _execute_streaming(agent, "test prompt")

        # Verify error message was printed
        assert mock_console.print.called


class TestRunReplLoop:
    """Tests for the main REPL loop."""

    @patch("allos.cli.interactive.console")
    @patch("allos.cli.interactive._handle_repl_command")
    @patch("allos.cli.interactive._execute_sync")
    def test_repl_loop_streaming_state_change(
        self, mock_execute_sync, mock_handle_command, mock_console, mock_get_provider
    ):
        """Test that streaming_state changes are applied.
        Covers line 242 in interactive.py.
        """
        from allos.agent import Agent, AgentConfig
        from allos.cli.interactive import _run_repl_loop

        config = AgentConfig(provider_name="openai", model="gpt-4o", no_tools=True)
        agent = Agent(config)

        # Simulate user input sequence: /stream on, then quit
        mock_console.input.side_effect = ["/stream on", "/quit"]

        # Mock _handle_repl_command to return state change for /stream on
        # and end_flag for /quit
        def handle_command_side_effect(cmd, current_state, end):
            if cmd == "/stream on":
                return (True, False)  # Enable streaming
            elif cmd == "/quit":
                return (None, True)  # End session
            return (None, False)

        mock_handle_command.side_effect = handle_command_side_effect

        # Run the REPL loop
        _run_repl_loop(agent, stream=False)

        # Verify _handle_repl_command was called twice
        assert mock_handle_command.call_count == 2

    @patch("allos.cli.interactive.console")
    @patch("allos.cli.interactive._execute_streaming")
    def test_repl_loop_executes_streaming_when_enabled(
        self, mock_execute_streaming, mock_console, mock_get_provider
    ):
        """Test that streaming mode executes _execute_streaming.
        Covers line 249 in interactive.py.
        """
        from allos.agent import Agent, AgentConfig
        from allos.cli.interactive import _run_repl_loop

        config = AgentConfig(provider_name="openai", model="gpt-4o", no_tools=True)
        agent = Agent(config)

        # Simulate user input: a prompt, then quit
        mock_console.input.side_effect = ["tell me a story", "/quit"]

        # Run REPL loop with streaming enabled
        _run_repl_loop(agent, stream=True)

        # Verify _execute_streaming was called for the prompt
        mock_execute_streaming.assert_called_once_with(agent, "tell me a story")
