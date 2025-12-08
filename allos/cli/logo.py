# allos/cli/logo.py

"""Contains the ASCII art logo and banner for the Allos Agent SDK CLI."""

from .. import __version__


def _format_version(version: str, max_width: int = 12) -> str:
    """Format version string to fit within max_width."""
    version_str = f"v{version}"
    if len(version_str) > max_width:
        # Truncate with ellipsis if too long
        return version_str[: max_width - 1] + "…"
    return version_str.ljust(max_width)


# Using an f-string to dynamically insert the version number
LOGO_BANNER = f"""
╔════════════════════════════════════════════════════════════════╗
║  █████╗ ██╗     ██╗      ██████╗ ███████╗                      ║
║ ██╔══██╗██║     ██║     ██╔═══██╗██╔════╝                      ║
║ ███████║██║     ██║     ██║   ██║███████╗                      ║
║ ██╔══██║██║     ██║     ██║   ██║╚════██║                      ║
║ ██║  ██║███████╗███████╗╚██████╔╝███████║   AGENT SDK          ║
║ ╚═╝  ╚═╝╚══════╝╚══════╝ ╚═════╝ ╚══════╝                      ║
╠════════════════════════════════════════════════════════════════╣
║           The LLM-Agnostic Agentic Framework                   ║
║        Build AI Agents Without Vendor Lock-In  •  {_format_version(__version__)} ║
╚════════════════════════════════════════════════════════════════╝
"""
