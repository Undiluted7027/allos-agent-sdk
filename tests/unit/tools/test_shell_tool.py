# tests/unit/tools/test_shell_tool.py

import sys

import pytest

from allos.tools.execution.shell import DANGEROUS_COMMANDS, ShellExecuteTool


class TestShellExecuteTool:
    def setup_method(self):
        self.tool = ShellExecuteTool()

    def test_execute_simple_command_success(self):
        """Test executing a simple, successful command like 'echo'."""
        result = self.tool.execute(command="echo 'Hello, Allos!'")

        assert result["status"] == "success"
        assert result["return_code"] == 0
        assert "Hello, Allos!" in result["stdout"]
        assert result["stderr"] == ""

    def test_execute_command_with_error(self):
        """Test a command that produces an error, like 'ls' on a non-existent file."""
        # This command will write to stderr and have a non-zero return code
        result = self.tool.execute(command="ls non_existent_file_xyz")

        assert result["status"] == "success"  # The tool executed successfully
        assert result["return_code"] != 0
        assert (
            "No such file or directory" in result["stderr"]
            or "cannot access" in result["stderr"]
        )  # Cross-platform check
        assert result["stdout"] == ""

    def test_execute_timeout(self):
        """Test that a long-running command is correctly timed out."""
        # Use a reliable cross-platform way to create a long-running process.
        # This calls the current Python interpreter to run a short script that sleeps.
        python_executable = sys.executable
        sleep_command = f'"{python_executable}" -c "import time; time.sleep(2)"'

        result = self.tool.execute(command=sleep_command, timeout=1)

        assert result["status"] == "error"
        assert "timed out after 1 seconds" in result["message"]
        assert result["return_code"] == -1

    @pytest.mark.parametrize("dangerous_command", DANGEROUS_COMMANDS)
    def test_block_dangerous_commands(self, dangerous_command):
        """Test that all defined dangerous commands are blocked."""
        # Test with arguments to ensure only the command itself is checked
        result = self.tool.execute(command=f"{dangerous_command} -rf /")

        assert result["status"] == "error"
        assert "is on the blocklist" in result["message"]
        assert result["return_code"] == -1
        assert "Execution denied" in result["stderr"]

    def test_command_not_found(self):
        """Test executing a command that does not exist on the system."""
        result = self.tool.execute(command="a_completely_fake_command_xyz")

        assert result["status"] == "error"
        assert "Command not found" in result["message"]
        assert result["return_code"] == -1

    def test_empty_command_fails(self):
        """Test that providing an empty or whitespace-only command fails."""
        result_empty = self.tool.execute(command="")
        assert result_empty["status"] == "error"
        assert "Cannot execute an empty command" in result_empty["message"]

        result_space = self.tool.execute(command="   ")
        assert result_space["status"] == "error"
        assert "Cannot execute an empty command" in result_space["message"]

    def test_missing_command_argument(self):
        """Test calling execute without the 'command' kwarg."""
        result = self.tool.execute(timeout=10)  # Missing 'command'
        assert result["status"] == "error"
        assert "The 'command' argument is required" in result["message"]

    def test_shlex_parsing_error(self):
        """
        Test that a command with invalid syntax (e.g., an unclosed quote)
        is handled correctly.
        """
        # An unclosed quote will cause shlex.split to raise a ValueError
        malformed_command = "echo 'Hello, Allos!"
        result = self.tool.execute(command=malformed_command)

        assert result["status"] == "error"
        assert "Failed to parse command" in result["message"]
        # Check for the specific error from shlex
        assert "No closing quotation" in result["message"]

    def test_unexpected_subprocess_error(self, mocker):
        """
        Test that a generic, unexpected exception during subprocess.run is
        caught and handled.
        """
        # Use mocker to make subprocess.run raise an unexpected error
        mocker.patch(
            "subprocess.run",
            side_effect=Exception("A simulated unexpected OS error"),
        )

        result = self.tool.execute(command="some_valid_command")

        assert result["status"] == "error"
        assert "An unexpected error occurred" in result["message"]
        assert "A simulated unexpected OS error" in result["stderr"]
        assert result["return_code"] == -1
