from pathlib import Path

import pytest

from allos import Agent, AgentConfig
from allos.providers import MessageRole
from tests.conftest import get_available_provider_params

providers_to_test = get_available_provider_params()


def _provider_matches(expected: str, actual: str) -> bool:
    if expected == "chat_completions" and actual == "openai":
        return True
    return expected == actual


@pytest.mark.integration
@pytest.mark.parametrize("provider_name, model", providers_to_test)
def test_agent_run_tool_loop_real(provider_name: str, model: str, work_dir: Path):
    """Real integration: Agent.run executes a write->read tool loop and finalizes metadata."""
    filename = f"agent_tool_loop_{provider_name}.txt"
    token = "ALLOS_TOOL_LOOP_OK"

    agent = Agent(
        AgentConfig(
            provider_name=provider_name,
            model=model,
            tool_names=["write_file", "read_file"],
            auto_approve=True,
            no_tools=False,
            max_iterations=5,
        )
    )

    prompt = (
        f"You must call write_file to create '{filename}' with exact content '{token}'. "
        f"Then call read_file on '{filename}'. "
        f"After tool results, answer with one sentence containing '{token}'."
    )
    result = agent.run(prompt)

    assert isinstance(result, str)
    assert result.strip() != ""
    assert token.lower() in result.lower()

    file_path = work_dir / filename
    assert file_path.exists()
    file_text = file_path.read_text(encoding="utf-8")
    assert token in file_text

    metadata = agent.last_run_metadata
    assert metadata is not None
    assert _provider_matches(provider_name, metadata.model.provider)
    assert metadata.model.model_id != ""
    assert metadata.usage.input_tokens >= 0
    assert metadata.usage.output_tokens >= 0
    assert metadata.usage.total_tokens == (
        metadata.usage.input_tokens + metadata.usage.output_tokens
    )
    assert metadata.turns.total_turns >= 2
    assert len(metadata.turns.turn_history) == metadata.turns.total_turns
    assert metadata.tools.total_tool_calls >= 1
    assert len(metadata.tools.tool_calls) == metadata.tools.total_tool_calls
    assert any(detail.tool_name == "write_file" for detail in metadata.tools.tool_calls)
    assert all(
        detail.status in {"success", "error", None}
        for detail in metadata.tools.tool_calls
    )
    assert all(
        (detail.execution_time_ms is None) or (detail.execution_time_ms >= 0)
        for detail in metadata.tools.tool_calls
    )

    assert any(msg.role == MessageRole.TOOL for msg in agent.context.messages)
    assert any(
        msg.role == MessageRole.ASSISTANT and len(msg.tool_calls) > 0
        for msg in agent.context.messages
    )
