# allos/cli/logo.py

"""
Contains the ASCII art logo and banner for the Allos Agent SDK CLI.
"""

from .. import __version__

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
║        Build AI Agents Without Vendor Lock-In  •  v{__version__}       ║
╚════════════════════════════════════════════════════════════════╝
"""
