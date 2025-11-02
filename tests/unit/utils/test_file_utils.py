# tests/unit/utils/test_file_utils.py
from pathlib import Path

import pytest

from allos.utils.errors import FileOperationError
from allos.utils.file_utils import (
    is_safe_path,
    safe_read_file,
    safe_write_file,
    validate_directory,
)

# --- Tests for is_safe_path ---


def test_is_safe_path_handles_non_existent_paths_gracefully(work_dir: Path):
    """
    Test the fallback logic in is_safe_path for a path that doesn't exist yet.
    This covers the FileNotFoundError/RuntimeError except block.
    """
    # This path is safe, but doesn't exist, triggering the `except` block
    non_existent_path = work_dir / "new_dir" / "new_file.txt"
    assert is_safe_path(work_dir, non_existent_path)


def test_is_safe_path_with_invalid_path_component_fails(work_dir: Path):
    """
    Test that is_safe_path returns False when a path contains illegal characters,
    triggering the final generic exception handler.
    """
    # A NULL byte is an invalid character in a path.
    invalid_path_str = "some_dir/\0/some_file.txt"

    # The function should catch the low-level error (ValueError/OSError) and return False.
    assert is_safe_path(work_dir, invalid_path_str) is False


# --- Tests for safe_read_file ---


def test_safe_read_file_that_is_a_directory_fails(work_dir: Path):
    """Test that safe_read_file raises an error if the path is a directory."""
    (work_dir / "a_directory").mkdir()
    with pytest.raises(FileOperationError, match="Path is not a file"):
        safe_read_file("a_directory", str(work_dir))


def test_safe_read_file_too_large_fails(work_dir: Path):
    """Test that safe_read_file fails if the file exceeds the max_size."""
    large_file = work_dir / "large.txt"
    large_file.write_text("a" * 11)  # 11 bytes

    with pytest.raises(FileOperationError, match="is too large"):
        safe_read_file("large.txt", str(work_dir), max_size=10)


def test_safe_read_file_unicode_error_fails(work_dir: Path):
    """Test that a UnicodeDecodeError is wrapped in FileOperationError."""
    # Write bytes that are not valid UTF-8 (e.g., from latin-1)
    bad_encoding_file = work_dir / "bad.txt"
    bad_encoding_file.write_bytes(b"\xff")

    with pytest.raises(FileOperationError, match="Could not decode file"):
        safe_read_file("bad.txt", str(work_dir))


def test_safe_read_file_generic_exception(work_dir: Path, mocker):
    """Test that a generic exception during read is wrapped."""
    (work_dir / "a.txt").write_text("hello")
    # Mock the underlying read_text to simulate a disk error, etc.
    mocker.patch("pathlib.Path.read_text", side_effect=IOError("Disk is full"))

    with pytest.raises(FileOperationError, match="Failed to read file"):
        safe_read_file("a.txt", str(work_dir))


# --- Tests for safe_write_file ---


def test_safe_write_file_generic_exception(work_dir: Path, mocker):
    """Test that a generic exception during write is wrapped."""
    # Mock the underlying write_text to simulate a permission error, etc.
    mocker.patch("pathlib.Path.write_text", side_effect=OSError("Permission denied"))

    with pytest.raises(FileOperationError, match="Failed to write to file"):
        safe_write_file("new.txt", "content", str(work_dir))


def test_validate_directory_non_existent_fails(work_dir: Path):
    """
    Test that validate_directory raises FileOperationError if the directory does not exist.
    """
    # Define a path that is inside the safe work_dir but does not exist
    non_existent_dir = work_dir / "this_dir_does_not_exist"

    # Assert that the specific error is raised
    with pytest.raises(FileOperationError, match="Directory not found"):
        validate_directory(work_dir, non_existent_dir)


@pytest.mark.parametrize(
    "exception_to_raise",
    [
        pytest.param(OSError("Simulated OS Error"), id="os_error"),
        pytest.param(ValueError("Simulated Value Error"), id="value_error"),
        pytest.param(RuntimeError("Simulated Runtime Error"), id="runtime_error"),
    ],
)
def test_is_safe_path_catches_exceptions_and_returns_false(
    work_dir: Path, mocker, exception_to_raise
):
    """
    Test that is_safe_path's main try-except block correctly catches
    low-level errors and returns False, indicating an unsafe path.
    """
    # We mock `pathlib.Path.resolve` to raise the desired exception
    mocker.patch("pathlib.Path.resolve", side_effect=exception_to_raise)

    # The actual path doesn't matter since resolve is mocked
    is_it_safe = is_safe_path(work_dir, "some/path.txt")

    assert is_it_safe is False
