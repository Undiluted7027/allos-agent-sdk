# allos/tools/filesystem/directory.py

from pathlib import Path
from typing import Any, Dict

from ...utils.errors import FileOperationError
from ...utils.file_utils import (
    get_target_directory,
    list_directory_non_recursive,
    list_directory_recursive,
    validate_directory,
)
from ..base import BaseTool, ToolParameter, ToolPermission
from ..registry import tool


@tool
class ListDirectoryTool(BaseTool):
    """A tool for listing the contents of a directory."""

    name: str = "list_directory"
    description: str = (
        "Lists the contents of a specified directory. "
        "Allows for recursive listing and can include hidden files."
    )
    permission: ToolPermission = ToolPermission.ALWAYS_ALLOW

    parameters: list[ToolParameter] = [
        ToolParameter(
            name="path",
            type="string",
            description="The relative path to the directory to list. Defaults to the current directory.",
            required=False,
        ),
        ToolParameter(
            name="recursive",
            type="boolean",
            description="If true, lists contents of subdirectories recursively. Defaults to false.",
            required=False,
        ),
        ToolParameter(
            name="show_hidden",
            type="boolean",
            description="If true, includes files and directories starting with a dot ('.'). Defaults to false.",
            required=False,
        ),
    ]

    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Executes the directory listing operation.

        Args:
            **kwargs: May contain 'path', 'recursive', and 'show_hidden'.

        Returns:
            A dictionary containing the list of directory contents.
        """
        base_dir = Path(".").resolve()
        target_dir = get_target_directory(base_dir, kwargs.get("path", "."))
        recursive = kwargs.get("recursive", False)
        show_hidden = kwargs.get("show_hidden", False)

        try:
            validate_directory(base_dir, target_dir)

            if recursive:
                contents = list_directory_recursive(base_dir, target_dir, show_hidden)
            else:
                contents = list_directory_non_recursive(
                    base_dir, target_dir, show_hidden
                )

            return {
                "status": "success",
                "contents": contents,
                "message": f"Successfully listed contents of '{target_dir.relative_to(base_dir)}'.",
            }
        except FileOperationError as e:
            return {"status": "error", "message": str(e)}
