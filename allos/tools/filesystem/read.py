# allos/tools/filesystem/read.py

"""Implements a tool for safely reading the contents of a file.

This module provides the `FileReadTool`, another essential "sensory" capability
that allows an agent to gather detailed information from its local file
environment. It is the primary means for an agent to inspect code, read data,
or review the output of its own work.

Key design considerations for this tool include:
 - Security: It is built upon the `safe_read_file` utility, which enforces
   strict path validation to prevent traversal attacks. This ensures the agent
   can only read files within its designated working directory.
 - Flexibility: The tool supports reading an entire file or specifying a
   precise range of lines, allowing for both broad inspection and focused data
   extraction.
 - Safety by Default: As a read-only, non-destructive operation, this tool
   defaults to `ToolPermission.ALWAYS_ALLOW`. This provides a frictionless
   experience for the agent when it needs to gather information without
   requiring constant user intervention.
"""

from typing import Any, Dict, Optional

from ...utils.errors import FileOperationError
from ...utils.file_utils import safe_read_file
from ..base import BaseTool, ToolParameter, ToolPermission
from ..registry import tool


@tool
class FileReadTool(BaseTool):
    """A tool for reading the contents of a file."""

    name: str = "read_file"
    description: str = (
        "Reads the content of a specified file. "
        "Optionally, a specific range of lines can be read."
    )
    permission: ToolPermission = ToolPermission.ALWAYS_ALLOW

    parameters: list[ToolParameter] = [
        ToolParameter(
            name="path",
            type="string",
            description="The relative path to the file to be read.",
            required=True,
        ),
        ToolParameter(
            name="start_line",
            type="integer",
            description="Optional. The 1-based index of the starting line to read.",
            required=False,
        ),
        ToolParameter(
            name="end_line",
            type="integer",
            description="Optional. The 1-based index of the ending line to read.",
            required=False,
        ),
        ToolParameter(
            name="inclusive",
            type="boolean",
            description="Optional. Whether `end_line` is to be included in output or not.",
            required=False,
        ),
    ]

    def execute(
        self,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Executes the file read operation.

        Args:
            **kwargs: Must contain 'path', and may optionally contain
                      'start_line', 'end_line', and 'inclusive'.

        Returns:
            A dictionary containing the file content and line range info.
        """
        # Extract arguments from kwargs
        path = kwargs.get("path")
        start_line = kwargs.get("start_line")
        end_line = kwargs.get("end_line")
        inclusive = kwargs.get("inclusive", False)

        if not path:
            return {"status": "error", "message": "The 'path' argument is required."}
        try:
            # For simplicity, we assume the agent's working directory is the current directory.
            # TODO: This will be managed by the Agent Core later.
            content = safe_read_file(path, base_dir=".")
            lines = content.splitlines()
            total_lines = len(lines)

            # Determine line slice indices
            if start_line is not None or end_line is not None:
                start_index, end_index = self._compute_line_indices(
                    start_line, end_line, inclusive, total_lines
                )

                if start_index >= end_index:
                    return {
                        "status": "success",
                        "content": "",
                        "message": f"Invalid line range: start_line ({start_line}) must be less than end_line ({end_line}).",
                        "total_lines": total_lines,
                    }

                content = "\n".join(lines[start_index:end_index])
                message = f"Successfully read lines {start_index + 1} to {min(end_index, total_lines)} from '{path}'."
            else:
                message = f"Successfully read the entire file '{path}'."

            return {
                "status": "success",
                "content": content,
                "message": message,
                "total_lines": total_lines,
            }

        except FileOperationError as e:
            return {"status": "error", "message": str(e)}

    def _compute_line_indices(
        self,
        start_line: Optional[int],
        end_line: Optional[int],
        inclusive: bool,
        total_lines: int,
    ) -> tuple[int, int]:
        """Compute 0-based slice indices for the requested line range.

        Args:
            start_line: 1-based start line (optional)
            end_line: 1-based end line (optional)
            inclusive: Whether to include end_line in the output
            total_lines: Total lines in the file

        Returns:
            start_index, end_index suitable for slicing `lines[start_index:end_index]`
        """
        # Convert 1-based to 0-based
        start_index = (start_line - 1) if start_line is not None else 0
        end_index = end_line if end_line is not None else total_lines

        if end_line is not None and not inclusive:
            end_index -= 1

        # Clamp indices to valid range
        start_index = max(0, start_index)
        end_index = min(total_lines, end_index)

        return start_index, end_index
