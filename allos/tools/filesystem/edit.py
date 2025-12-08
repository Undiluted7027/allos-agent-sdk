# allos/tools/filesystem/edit.py

"""Implements a tool for performing a controlled, atomic file editing operation.

This module provides the `FileEditTool`, which gives an agent the capability to
make a precise, targeted modification to a file. It is designed not as a
general-purpose text editor, but as a safe and predictable find-and-replace
utility, ideal for tasks like updating a configuration value or correcting a
specific line of code.

The tool's design philosophy prioritizes safety and predictability over flexibility:
 - Strict Uniqueness Constraint: The tool is intentionally designed to only
   succeed if the target string (`find_string`) appears exactly once in the file.
   This prevents accidental, widespread changes and forces the agent to be highly
   specific, making the operation more predictable and less error-prone.
 - Atomic Operation: The tool reads the entire file into memory, performs the
   replacement, and then writes the modified content back. This ensures that
   the file is never left in a partially modified or corrupted state.
 - Path Security: It inherits the security of the core file utilities,
   preventing the agent from editing files outside its designated working directory.
 - Default User-Approval: As a file modification tool, it defaults to
   `ToolPermission.ASK_USER`, ensuring a human operator must approve potentially
   destructive changes.
"""

from typing import Any, Dict, cast

from ...utils.errors import FileOperationError
from ...utils.file_utils import safe_read_file, safe_write_file
from ..base import BaseTool, ToolParameter, ToolPermission
from ..registry import tool


@tool
class FileEditTool(BaseTool):
    """A tool for performing a find-and-replace operation on a file."""

    name: str = "edit_file"
    description: str = (
        "Replaces a specific string in a file with a new string. "
        "The operation will fail if the 'find_string' does not appear exactly once in the file."
    )
    permission: ToolPermission = ToolPermission.ASK_USER

    parameters: list[ToolParameter] = [
        ToolParameter(
            name="path",
            type="string",
            description="The relative path to the file to be edited.",
            required=True,
        ),
        ToolParameter(
            name="find_string",
            type="string",
            description="The exact string to search for in the file. Must be unique.",
            required=True,
        ),
        ToolParameter(
            name="replace_with",
            type="string",
            description="The string to replace the 'find_string' with.",
            required=True,
        ),
    ]

    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        """Executes the file edit operation.

        Args:
            **kwargs: Must contain 'path', 'find_string', and 'replace_with'.

        Returns:
            A dictionary indicating the status of the operation.
        """
        path = cast(str, kwargs.get("path"))
        find_string = cast(str, kwargs.get("find_string"))
        replace_with = cast(str, kwargs.get("replace_with"))

        if not all([path, find_string, replace_with]):
            return {
                "status": "error",
                "message": "The 'path', 'find_string', and 'replace_with' arguments are all required.",
            }

        try:
            # Atomic operation: read the file first
            content = safe_read_file(path, base_dir=".")

            # Uniqueness validation
            count = content.count(find_string)
            if count == 0:
                return {
                    "status": "error",
                    "message": f"The 'find_string' was not found in the file '{path}'. Edit failed.",
                }
            if count > 1:
                return {
                    "status": "error",
                    "message": f"The 'find_string' appeared {count} times in the file '{path}'. Edit failed because it was not unique.",
                }

            # Perform the replacement and write the file back
            new_content = content.replace(find_string, replace_with, 1)
            safe_write_file(path, new_content, base_dir=".")

            return {
                "status": "success",
                "message": f"Successfully edited '{path}'.",
            }

        except FileOperationError as e:
            return {"status": "error", "message": str(e)}
