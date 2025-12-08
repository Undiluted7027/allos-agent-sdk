# allos/tools/filesystem/__init__.py

"""Provides a suite of fundamental tools for interacting with the local filesystem.

This package equips an agent with the essential capabilities to perceive its
environment (by reading files and listing directories) and to enact changes
(by writing, creating, or editing files). These tools are the primary means by
which an agent can create code, save data, and manage its own workspace.

The central design philosophy for all tools in this package is **security**.
Every tool is built upon the SDK's safe path utilities, which strictly confine
all file operations to the agent's designated working directory. This design is
critical for preventing path traversal attacks and ensuring the agent cannot
access or modify sensitive system files.

The imports within this module serve to register each tool with the central
`ToolRegistry`, making them available for use by the agent.
"""

# These imports are for side-effects, to ensure the tools are registered
from .directory import ListDirectoryTool
from .edit import FileEditTool
from .read import FileReadTool
from .write import FileWriteTool

__all__ = ["FileReadTool", "FileWriteTool", "FileEditTool", "ListDirectoryTool"]
