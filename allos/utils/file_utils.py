# allos/utils/file_utils.py

"""Secure file system utilities for the Allos Agent SDK."""

from pathlib import Path

from .errors import FileOperationError

DEFAULT_MAX_FILE_SIZE = 1_000_000  # 1 MB


def is_safe_path(base_dir: Path, target_path: Path) -> bool:
    """
    Check if a target path is safely within a base directory.

    Prevents directory traversal attacks (e.g., `../../...`).

    Args:
        base_dir: The root directory that is considered safe.
        target_path: The path to check.

    Returns:
        True if the path is safe, False otherwise.
    """
    try:
        # Resolve both paths to their absolute form to prevent symlink tricks
        resolved_base = base_dir.resolve(strict=True)
        resolved_target = target_path.resolve(strict=True)
        # Check if the resolved target path is a subpath of the resolved base path
        return resolved_target.is_relative_to(resolved_base)
    except (FileNotFoundError, RuntimeError):
        # strict=True raises FileNotFoundError if path doesn't exist
        # is_relative_to can raise RuntimeError on Windows with different drives
        # In these cases, we can check the unresolved path as a fallback
        # for operations like writing a new file.
        try:
            resolved_target = (base_dir / target_path).resolve()
            return resolved_target.is_relative_to(base_dir.resolve())
        except Exception:
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


def safe_write_file(path: str, content: str, base_dir: str) -> None:
    """
    Writes content to a file after validating the path is safe.

    Args:
        path: The relative path to the file.
        content: The content to write.
        base_dir: The working directory of the agent.

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
        target_path.write_text(content, encoding="utf-8")
    except Exception as e:
        raise FileOperationError(f"Failed to write to file '{path}': {e}") from e
