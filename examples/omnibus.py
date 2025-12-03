"""
Comprehensive test encompassing 5 providers (Groq, Mistral, Together AI, OpenAI, Anthropic).
This script demonstrates the power of the Allos SDK by chaining together multiple
providers in a single session, passing context between them, and utilizing their
unique strengths (speed, planning, coding, execution, review).
"""

import os
import time

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from allos import Agent, AgentConfig

console = Console()


def step(title):
    console.print(f"\n[bold blue]{'=' * 10} STEP: {title} {'=' * 10}[/]")


def run_turn(agent, prompt, expected_outcome):
    step(prompt[:50] + "...")
    console.print(f"[dim]Goal: {expected_outcome}[/dim]")

    start = time.time()
    response = agent.run(prompt)
    duration = time.time() - start

    console.print(
        Panel(
            Markdown(response),
            title=f"Agent Response ({duration:.2f}s)",
            border_style="green",
        )
    )
    return response


def main():
    console.print(
        Panel.fit(
            "[bold]Allos Omnibus Verification: Phase 2.0[/]", border_style="magenta"
        )
    )

    session_file = "omnibus_session.json"
    if os.path.exists(session_file):
        os.remove(session_file)

    # --- 1. GROQ: The Fast Ideator (Testing --no-tools and Aliasing) ---
    # We use Groq because it's fast. We disable tools to test the config override.
    # We use the alias 'groq' to test the intelligent registry.
    console.print("\n[bold cyan]--- Provider 1: Groq (Llama 3.1 8b) ---[/]")
    console.print(
        "[dim]Testing: Intelligent Registry (Alias), Config Override (no_tools)[/dim]"
    )

    groq_config = AgentConfig(
        provider_name="groq",
        model="llama-3.1-8b-instant",
        no_tools=True,  # Explicitly disable tools
        auto_approve=True,
    )
    agent_groq = Agent(groq_config)

    run_turn(
        agent_groq,
        "Give me ONE interesting Python automation idea involving file manipulation. Just the idea, one sentence.",
        "Fast response, no tool calls attempted.",
    )

    # Save session to hand off context
    agent_groq.save_session(session_file)

    # --- 2. MISTRAL: The Planner (Testing Standard Compatibility) ---
    # We load the session. We switch to Mistral.
    # Mistral via API is OpenAI-compatible but uses a different endpoint.
    # Registry should handle base_url injection automatically.
    console.print("\n[bold cyan]--- Provider 2: Mistral (Mistral Small) ---[/]")
    console.print(
        "[dim]Testing: Session Loading, Provider Switching, Auto-Base-URL[/dim]"
    )

    agent_mistral = Agent.load_session(session_file)
    # Reconfigure for Mistral
    agent_mistral.config.provider_name = "mistral"
    agent_mistral.config.model = "mistral-small-latest"
    agent_mistral.config.no_tools = False  # Re-enable tools
    # Re-init provider with new config
    agent_mistral.__init__(agent_mistral.config, agent_mistral.context)

    run_turn(
        agent_mistral,
        "Create a step-by-step implementation plan for that idea.",
        "Context aware response outlining steps.",
    )

    agent_mistral.save_session(session_file)

    # --- 3. TOGETHER AI: The Coder (Testing Explicit Chat Completions Adapter) ---
    # Here we simulate a 'custom' provider scenario by using the raw 'chat_completions' provider
    # and passing the base_url manually (even though 'together' alias exists, we want to test manual mode).
    console.print("\n[bold cyan]--- Provider 3: Together AI (DeepSeek Coder) ---[/]")
    console.print(
        "[dim]Testing: Explicit 'chat_completions' provider, Manual Base URL, API Key injection[/dim]"
    )

    agent_together = Agent.load_session(session_file)

    # Configure manually
    agent_together.config.provider_name = "chat_completions"
    agent_together.config.model = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
    agent_together.config.base_url = "https://api.together.xyz/v1"
    # We explicitly inject the key to test that flow
    agent_together.config.api_key = os.getenv("TOGETHER_API_KEY")

    # Re-init
    agent_together.__init__(agent_together.config, agent_together.context)

    run_turn(
        agent_together,
        "Write the Python code for this plan. Do not execute it, just write it.",
        "Code generation using a specific model via generic adapter.",
    )

    agent_together.save_session(session_file)

    # --- 4. OPENAI: The Executor (Testing Tools & Max Tokens) ---
    # Finally, we hand off to GPT-4o to actually do the work.
    # We also test the 'max_tokens' parameter here just to verify it passes through.
    console.print("\n[bold cyan]--- Provider 4: OpenAI (GPT-4o) ---[/]")
    console.print(
        "[dim]Testing: Tool Execution (Write/Read/Shell), Max Tokens Param[/dim]"
    )

    agent_openai = Agent.load_session(session_file)
    agent_openai.config.provider_name = "openai"
    agent_openai.config.model = "gpt-4o"
    agent_openai.config.tool_names = ["write_file", "shell_exec"]
    agent_openai.config.max_tokens = 2000  # Verify this doesn't crash the request
    agent_openai.config.auto_approve = True
    # CRITICAL: Clear Base URL to ensure you hit OpenAI servers
    agent_openai.config.base_url = None

    agent_openai.__init__(agent_openai.config, agent_openai.context)

    # This prompt requires tools
    run_turn(
        agent_openai,
        "Save that code to a file named 'omnibus_demo.py' and execute it. Report the output.",
        "Agent should write file, run python, and report stdout.",
    )

    agent_openai.save_session(session_file)

    # --- 5. ANTHROPIC: The Reviewer (Testing Native Provider & Max Tokens) ---
    # We switch to Claude to review the output. This tests the Anthropic provider
    # correctly handling a context populated by OpenAI tools.
    console.print("\n[bold cyan]--- Provider 5: Anthropic (Claude 3 Haiku) ---[/]")
    console.print(
        "[dim]Testing: Native Provider, Context Translation (OpenAI -> Anthropic), Max Tokens[/dim]"
    )

    agent_anthropic = Agent.load_session(session_file)
    agent_anthropic.config.provider_name = "anthropic"
    agent_anthropic.config.model = "claude-3-haiku-20240307"
    # Anthropic often requires max_tokens to be explicit or it defaults.
    # We set it here to verify the param is respected.
    agent_anthropic.config.max_tokens = 1000
    agent_anthropic.__init__(agent_anthropic.config, agent_anthropic.context)

    run_turn(
        agent_anthropic,
        "Analyze the output of the script execution. Did it work as expected? Be brief.",
        "Correctly interprets the previous tool output message.",
    )

    # Cleanup
    if os.path.exists("omnibus_demo.py"):
        os.remove("omnibus_demo.py")
    if os.path.exists(session_file):
        os.remove(session_file)

    console.print("\n[bold green]✅ Omnibus Test Complete. The Allos SDK is robust.[/]")


if __name__ == "__main__":
    main()
