# tests/unit/cli/test_logo.py


from allos.cli.logo import _format_version


def test_format_version_normal():
    """Test that a standard version string is correctly formatted and padded."""
    version = "0.1.0"
    formatted = _format_version(version, max_width=12)
    assert formatted == "v0.1.0      "
    assert len(formatted) == 12


def test_format_version_exact_width():
    """Test a version string that exactly fits the max width."""
    version = "10.20.30"  # "v10.20.30" is 9 chars
    formatted = _format_version(version, max_width=9)
    assert formatted == "v10.20.30"
    assert len(formatted) == 9


def test_format_version_truncates_long_string():
    """
    Test that a version string longer than max_width is truncated with an ellipsis.
    This specifically covers the `return version_str[: max_width - 1] + "…"` line.
    """
    long_version = "0.1.0-alpha.1+build.12345"
    # "v" + long_version is 28 chars long

    formatted = _format_version(long_version, max_width=12)

    # Expected: "v0.1.0-alph…" (11 chars + 1 ellipsis = 12)
    assert formatted == "v0.1.0-alph…"
    assert len(formatted) == 12


def test_format_version_empty_string():
    """Test that an empty version string is handled gracefully."""
    version = ""
    formatted = _format_version(version, max_width=12)
    assert formatted == "v           "
    assert len(formatted) == 12
