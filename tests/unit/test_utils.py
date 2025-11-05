# tests/unit/test_utils.py

from pathlib import Path

import pytest

# Test if the package is importable
import allos
from allos.__version__ import __version__ as internal_version
from allos.utils import (
    AllosError,
    ProviderError,
    count_tokens,
    is_safe_path,
    logger,
)


def test_pytest_is_working():
    """A simple sanity check to confirm pytest runs."""
    assert True


def test_allos_package_is_importable():
    """Verify that the main package can be imported."""
    assert allos.__name__ == "allos"


def test_version_is_exposed_and_correct():
    """
    Tests that the package version is correctly exposed on the top-level import
    and matches the internal version number.
    """
    assert allos.__version__ == internal_version
    assert isinstance(allos.__version__, str)
    # Check that the version for MVP is correct
    assert allos.__version__ == "0.0.2-dev"


def test_custom_error_hierarchy():
    """Test that our custom errors inherit from the base AllosError."""
    try:
        raise ProviderError("Test message", provider="test_provider")
    except AllosError as e:
        assert isinstance(e, ProviderError)
        assert "test_provider" in str(e)
    except Exception:
        pytest.fail("ProviderError was not caught by AllosError")


def test_logger_is_configured():
    """Check if the default logger is set up on import."""
    assert logger.name == "allos"
    assert logger.hasHandlers() is True


def test_safe_path_utility(work_dir: Path):
    """
    Test the is_safe_path utility using the work_dir fixture.
    The work_dir fixture provides a temporary, safe base directory.
    """
    # Safe paths within the working directory
    safe_file = work_dir / "safe_file.txt"
    safe_subdir_file = work_dir / "subdir" / "safe_file.txt"
    assert is_safe_path(work_dir, safe_file) is True
    assert is_safe_path(work_dir, safe_subdir_file) is True

    # Unsafe paths (directory traversal)
    unsafe_path = work_dir / ".." / "unsafe_file.txt"
    assert is_safe_path(work_dir, unsafe_path) is False

    # Path to a different, unrelated directory
    unrelated_path = Path("/tmp/some_other_dir")
    assert is_safe_path(work_dir, unrelated_path) is False


def test_token_counter():
    """Test the token counting utility with both a known and fallback model."""
    # Test with a model known by tiktoken
    text = "hello world"
    # gpt-4 tokenizer sees "hello" and " world" as two separate tokens
    assert count_tokens(text, model="gpt-4") == 2

    # Test with a model not in tiktoken to trigger the fallback
    text_long = "This is a sentence with eight words."
    # Fallback is len(text) // 4 = 36 // 4 = 9
    assert count_tokens(text_long, model="a-fake-model-name") == 9
