# allos/tools/filesystem/read.py

"""
A tool for allowing the agent to write or create files, with the crucial `ASK_USER` permission set.
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
        """
        Executes the file write operation.

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
