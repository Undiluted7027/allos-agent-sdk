# allos/tools/filesystem/write.py

"""Implements a tool for creating, overwriting, or appending to files.

This module provides the `FileWriteTool`, a powerful "action" capability that
enables an agent to create artifacts, save its work, or modify its environment.
It is the primary tool for tasks such as writing code, generating reports, or
saving configuration files.

Given that this tool can alter the filesystem, its design is centered on safety
and user control:
 - Functionality: It supports creating new files, completely overwriting
   existing files, or appending content to the end of a file.
 - Path Security: It uses the `safe_write_file` utility to ensure that all
   write operations are strictly confined to the agent's designated working
   directory, preventing the agent from modifying system files or other
   sensitive data.
 - Permission Control: As a potentially destructive operation, the tool
   defaults to `ToolPermission.ASK_USER`. This acts as a critical safety check,
   requiring explicit consent from a human operator before the agent can make
   any changes to the filesystem.
"""

from typing import Any, Dict

from ...utils.errors import FileOperationError
from ...utils.file_utils import safe_write_file
from ..base import BaseTool, ToolParameter, ToolPermission
from ..registry import tool


@tool
class FileWriteTool(BaseTool):
    """A tool for writing content to a file."""

    name: str = "write_file"
    description: str = (
        "Writes content to a specified file. "
        "If the file does not exist, it will be created. "
        "If it exists, its contents will be overwritten."
    )
    permission: ToolPermission = ToolPermission.ASK_USER

    parameters: list[ToolParameter] = [
        ToolParameter(
            name="path",
            type="string",
            description="The relative path to the file to be written.",
            required=True,
        ),
        ToolParameter(
            name="content",
            type="string",
            description="The content to write into the file.",
            required=True,
        ),
        ToolParameter(
            name="append_mode",
            type="boolean",
            description="Whether to overwite file or append. Overwrites by default.",
            required=False,
        ),
    ]

    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        """Executes the file write operation.

        Args:
            **kwargs: Must contain 'path' and 'content'.

        Returns:
            A dictionary indicating the status of the operation.
        """
        # Extract arguments from kwargs
        path = kwargs.get("path")
        content = kwargs.get("content")
        append_mode = kwargs.get("append_mode", False)
        if path is None or content is None:
            return {
                "status": "error",
                "message": "Both 'path' and 'content' arguments are required.",
            }
        try:
            safe_write_file(path, content, base_dir=".", append_mode=append_mode)
            action = "appended to" if append_mode else "wrote"
            return {
                "status": "success",
                "message": f"Successfully {action} {len(content)} bytes to '{path}'.",
            }
        except FileOperationError as e:
            return {"status": "error", "message": str(e)}
