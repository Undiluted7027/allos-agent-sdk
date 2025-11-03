# allos/utils/file_utils.py

"""Secure file system utilities for the Allos Agent SDK."""

import os
from pathlib import Path
from typing import Union

from .errors import FileOperationError

DEFAULT_MAX_FILE_SIZE = 1_000_000  # 1 MB


def is_safe_path(base_dir: Path, target_path_str: Union[str, Path]) -> bool:
    """
    Securely verify that a target path stays within a trusted base directory,
    even in the presence of symlinks or traversal attempts.

    Works across Linux, macOS, and Windows.

    Args:
        base_dir: The root directory that is considered safe.
        target_path_str: The path to check, relative to the base_dir.

    Returns:
        True if the path is safe, False otherwise.
    """
    # Explicitly check for null bytes before any path operations.
    # Some Python/OS versions might truncate the path at the null byte
    # instead of raising an error, leading to a security vulnerability.
    if "\0" in str(target_path_str):
        return False
    try:
        safe_root = base_dir.resolve(strict=True)

        # Combine and resolve fully (resolving symlinks safely)
        target_path = (safe_root / target_path_str).resolve(strict=False)

        # Ensure final path is still within base_dir after resolving symlinks
        return target_path.is_relative_to(safe_root)

    except (OSError, ValueError, RuntimeError):
        return False


def safe_read_file(
    path: str, base_dir: str, max_size: int = DEFAULT_MAX_FILE_SIZE
) -> str:
    """
    Reads a file after validating the path is safe and the file is not too large.

    Args:
        path: The relative path to the file.
        base_dir: The working directory of the agent.
        max_size: The maximum file size in bytes to read.

    Returns:
        The content of the file.

    Raises:
        FileOperationError: If the path is unsafe, the file doesn't exist,
                            is too large, or cannot be decoded.
    """
    base_path = Path(base_dir)
    target_path = base_path / path

    if not is_safe_path(base_path, target_path):
        raise FileOperationError(
            f"Path '{path}' is outside the safe working directory."
        )

    if not target_path.exists():
        raise FileOperationError(f"File not found: '{path}'")

    if not target_path.is_file():
        raise FileOperationError(f"Path is not a file: '{path}'")

    file_size = target_path.stat().st_size
    if file_size > max_size:
        raise FileOperationError(
            f"File '{path}' is too large ({file_size} bytes). "
            f"Maximum allowed size is {max_size} bytes."
        )

    try:
        return target_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        raise FileOperationError(f"Could not decode file '{path}' as UTF-8.") from e
    except Exception as e:
        raise FileOperationError(f"Failed to read file '{path}': {e}") from e


def safe_write_file(
    path: str, content: str, base_dir: str, append_mode: bool = False
) -> None:
    """
    Writes content to a file after validating the path is safe.

    Args:
        path: The relative path to the file.
        content: The content to write.
        base_dir: The working directory of the agent.
        append_mode: Whether to append to the file instead of overwriting.

    Raises:
        FileOperationError: If the path is unsafe or writing fails.
    """
    base_path = Path(base_dir)
    target_path = base_path / path

    if not is_safe_path(base_path, (base_path / path).parent):
        raise FileOperationError(
            f"Path '{path}' is outside the safe working directory."
        )

    try:
        # Ensure parent directories exist
        target_path.parent.mkdir(parents=True, exist_ok=True)
        (
            target_path.write_text(content, encoding="utf-8", errors="strict")
            if not append_mode
            else target_path.write_text(
                target_path.read_text(encoding="utf-8") + content, encoding="utf-8"
            )
        )
    except Exception as e:
        raise FileOperationError(f"Failed to write to file '{path}': {e}") from e


def get_target_directory(base_dir: Path, path_str: str) -> Path:
    """Compute the absolute target directory path."""
    return (base_dir / path_str).resolve()


def validate_directory(base_dir: Path, target_dir: Path) -> None:
    """Validate that the target directory is safe and exists."""
    if not is_safe_path(base_dir, target_dir):
        raise FileOperationError(
            f"Path '{target_dir}' is outside the safe working directory."
        )
    if not target_dir.exists():
        raise FileOperationError(f"Directory not found: '{target_dir}'")
    if not target_dir.is_dir():
        raise FileOperationError(f"Path is not a directory: '{target_dir}'")


def list_directory_recursive(
    base_dir: Path, target_dir: Path, show_hidden: bool
) -> list[str]:
    """List all contents recursively."""
    contents = []
    for root, dirs, files in os.walk(target_dir):
        if not show_hidden:
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            files = [f for f in files if not f.startswith(".")]

        current_path = Path(root).relative_to(base_dir)
        for name in sorted(dirs):
            contents.append(f"{(current_path / name).as_posix()}/")
        for name in sorted(files):
            contents.append(str((current_path / name).as_posix()))
    return contents


def list_directory_non_recursive(
    base_dir: Path, target_dir: Path, show_hidden: bool
) -> list[str]:
    """List contents of the directory without recursion."""
    contents = []
    for entry in sorted(target_dir.iterdir()):
        if not show_hidden and entry.name.startswith("."):
            continue
        relative_path = entry.relative_to(base_dir)
        if entry.is_dir():
            contents.append(f"{relative_path.as_posix()}/")
        else:
            contents.append(str(relative_path.as_posix()))
    return contents
