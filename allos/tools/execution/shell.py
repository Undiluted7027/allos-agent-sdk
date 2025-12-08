# allos/tools/execution/shell.py

"""Implements a tool for executing shell commands with a strong focus on security.

This module provides the `ShellExecuteTool`, which allows an agent to run commands
in a subprocess on the host system. This is a powerful capability that enables
the agent to interact with system utilities, manage files, and run scripts.

To mitigate the significant risks associated with shell access, this tool
incorporates several critical safety measures:
 - Command Parsing: Uses `shlex.split` to safely parse command strings,
   preventing command injection vulnerabilities where a single argument could be
   interpreted as multiple commands.
 - Dangerous Command Blocklist: Maintains a list of high-risk commands
   (e.g., `rm`, `sudo`, `mkfs`) that are explicitly forbidden from execution.
 - Non-Interactive Execution: Commands are run in a non-interactive session
   with a timeout, preventing them from hanging or prompting for user input.
 - Permission Control: The tool defaults to `ToolPermission.ASK_USER`,
   requiring explicit user consent before any command is executed, unless
   overridden by the agent's configuration.
"""

import shlex
import subprocess
from typing import Any, Dict

from ...utils.logging import logger
from ..base import BaseTool, ToolParameter, ToolPermission
from ..registry import tool

# A simple blocklist of commands that are too dangerous for an agent to run.
# This is not exhaustive and should be seen as a first layer of defense.
DANGEROUS_COMMANDS = [
    "sudo",
    "su",
    "rm",
    "mv",
    "mkfs",
    "shutdown",
    "reboot",
    "kill",
    "pkill",
    "userdel",
    "groupdel",
    "chmod",
    "chown",
    "visudo",
]

DEFAULT_TIMEOUT = 60  # seconds


@tool
class ShellExecuteTool(BaseTool):
    """A tool for executing shell commands."""

    name: str = "shell_exec"
    description: str = (
        "Executes a shell command in a secure, non-interactive environment. "
        "Captures and returns the standard output, standard error, and return code."
    )
    permission: ToolPermission = ToolPermission.ASK_USER

    parameters: list[ToolParameter] = [
        ToolParameter(
            name="command",
            type="string",
            description="The shell command to execute.",
            required=True,
        ),
        ToolParameter(
            name="timeout",
            type="integer",
            description=f"Optional. The timeout for the command in seconds. Defaults to {DEFAULT_TIMEOUT}s.",
            required=False,
        ),
    ]

    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        """Executes the shell command.

        Args:
            **kwargs: Must contain 'command' and may optionally contain 'timeout'.

        Returns:
            A dictionary with the command's stdout, stderr, and return code.
        """
        command = kwargs.get("command")
        timeout = kwargs.get("timeout", DEFAULT_TIMEOUT)

        if not command:
            return {
                "status": "error",
                "message": "Cannot execute an empty command. The 'command' argument is required.",
            }

        try:
            # Use shlex to safely parse the command string
            args = shlex.split(command)
        except ValueError as e:
            return {
                "status": "error",
                "message": f"Failed to parse command: {e}",
            }

        if not args:
            return {"status": "error", "message": "Cannot execute an empty command."}

        # --- Security Check: Dangerous Command Detection ---
        if args[0] in DANGEROUS_COMMANDS:
            logger.warning(f"Blocked dangerous command attempt: '{command}'")
            return {
                "status": "error",
                "message": f"Command '{args[0]}' is on the blocklist and cannot be executed.",
                "return_code": -1,
                "stdout": "",
                "stderr": "Execution denied for security reasons.",
            }

        try:
            process = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,  # We handle the return code manually
            )

            return {
                "status": "success",
                "return_code": process.returncode,
                "stdout": process.stdout,
                "stderr": process.stderr,
            }
        except FileNotFoundError:
            return {
                "status": "error",
                "message": f"Command not found: '{args[0]}'",
                "return_code": -1,
                "stdout": "",
                "stderr": f"The command '{args[0]}' could not be found on the system path.",
            }
        except subprocess.TimeoutExpired:
            return {
                "status": "error",
                "message": f"Command timed out after {timeout} seconds.",
                "return_code": -1,
                "stdout": "",
                "stderr": f"Timeout of {timeout}s exceeded.",
            }
        except Exception as e:
            logger.error(f"An unexpected error occurred during shell execution: {e}")
            return {
                "status": "error",
                "message": f"An unexpected error occurred: {e}",
                "return_code": -1,
                "stdout": "",
                "stderr": str(e),
            }
