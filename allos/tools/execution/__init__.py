# allos/tools/execution/__init__.py

"""Provides tools for executing code and system commands on the host machine.

This package contains some of the most powerful capabilities an agent can possess:
the ability to interact directly with the operating system. The tools herein,
such as `ShellExecuteTool`, allow the agent to run shell commands, execute
scripts, and interact with other processes, forming a critical bridge between
the agent's reasoning and real-world action.

Given the inherent risks of such operations, a primary design consideration
for this package is security. Tools are built with safety mechanisms, such
as input sanitization and blocklisting of dangerous commands, to prevent
unintended or malicious actions by the agent.

Due to their powerful nature, these tools often default to requiring explicit user
permission (`ToolPermission.ASK_USER`) before execution.
"""

from .shell import ShellExecuteTool

__all__ = ["ShellExecuteTool"]
