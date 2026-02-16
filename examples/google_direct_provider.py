# examples/google_direct_provider.py

"""Google advanced configuration via direct provider initialization.

Use Agent + AgentConfig for standard agent workflows.
Use ProviderRegistry.get_provider("google", ...) directly when you need
Google-specific options like vertexai/project/location/credentials.

Usage (Gemini API):
  uv pip install "allos-agent-sdk[google]" python-dotenv
  export GOOGLE_API_KEY="..."
  python examples/google_direct_provider.py

Usage (Vertex AI):
  export GOOGLE_CLOUD_PROJECT="your-project-id"
  export GOOGLE_APPLICATION_CREDENTIALS="/path/to/sa.json"
  python examples/google_direct_provider.py
"""

import os
import sys
from typing import Any, Dict

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from allos.providers import Message, MessageRole, ProviderRegistry

console = Console()


def _gemini_api_demo() -> None:
    """Run a Gemini API mode example."""
    provider = ProviderRegistry.get_provider(
        "google",
        model="gemini-2.5-flash",
        api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
    )
    response = provider.chat(
        [Message(role=MessageRole.USER, content="In one sentence, explain vector DBs.")]
    )
    console.print(Panel(response.content or "", title="Gemini API", border_style="cyan"))


def _vertex_ai_demo() -> None:
    """Run a Vertex AI mode example with advanced kwargs."""
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project:
        console.print(
            "[yellow]GOOGLE_CLOUD_PROJECT not set. Skipping Vertex AI demo.[/]"
        )
        return

    kwargs: Dict[str, Any] = {
        "vertexai": True,
        "project": project,
        "location": os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
    }
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if credentials_path:
        kwargs["credentials_path"] = credentials_path

    provider = ProviderRegistry.get_provider(
        "google",
        model="gemini-2.5-flash",
        **kwargs,
    )
    response = provider.chat(
        [
            Message(
                role=MessageRole.USER,
                content="Briefly explain when to use Vertex AI over API-key mode.",
            )
        ]
    )
    console.print(Panel(response.content or "", title="Vertex AI", border_style="green"))


def main() -> None:
    """Run direct Google provider examples."""
    load_dotenv()

    if sys.version_info < (3, 10):
        console.print("[red]Google provider requires Python 3.10+.[/red]")
        return

    console.print(
        Panel.fit(
            "[bold]Google: Agent vs Direct Provider[/]\n"
            "This file demonstrates direct provider usage for advanced Google kwargs.",
            border_style="blue",
        )
    )

    if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
        _gemini_api_demo()
    else:
        console.print(
            "[yellow]No GOOGLE_API_KEY/GEMINI_API_KEY found. Skipping Gemini API demo.[/]"
        )

    _vertex_ai_demo()


if __name__ == "__main__":
    main()
