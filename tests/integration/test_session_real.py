# tests/integration/test_session_real.py
from pathlib import Path

import pytest

from allos import Agent, AgentConfig
from tests.conftest import get_available_provider_params

providers_to_test = get_available_provider_params()


def _provider_matches(expected: str, actual: str) -> bool:
    """Handle known provider-name normalization in metadata."""
    if expected == "chat_completions" and actual == "openai":
        return True
    return expected == actual


@pytest.mark.integration
@pytest.mark.parametrize("provider_name, model", providers_to_test)
def test_session_save_load_continue_real(
    provider_name: str, model: str, work_dir: Path
):
    """Real worklow: run -> save -> load -> continue conversation."""
    agent = Agent(
        AgentConfig(
            provider_name=provider_name,
            model=model,
            no_tools=True,
            max_iterations=3,
        )
    )

    first = agent.run("Say exactly: session test start.")
    assert first.strip() != ""
    assert agent.last_run_metadata is not None

    initial_context_size = len(agent.context.messages)
    session_path = work_dir / "real_session.json"
    agent.save_session(session_path)
    assert session_path.exists()

    loaded = Agent.load_session(session_path)
    assert loaded.config.provider_name == provider_name
    assert loaded.config.model == model
    assert len(loaded.context.messages) == initial_context_size

    second = loaded.run("Now summarize what you just said in 5 words.")
    assert second.strip() != ""
    assert loaded.last_run_metadata is not None
    assert len(loaded.context.messages) > initial_context_size

    metadata = loaded.last_run_metadata
    assert metadata is not None
    assert _provider_matches(provider_name, metadata.model.provider)
    assert metadata.usage.total_tokens >= 0
