from pathlib import Path

import pytest

from allos import Agent, AgentConfig
from tests.conftest import get_available_provider_params

providers_to_test = get_available_provider_params()


def _provider_matches(expected: str, actual: str) -> bool:
    if expected == "chat_completions" and actual == "openai":
        return True
    return expected == actual


@pytest.mark.integration
@pytest.mark.parametrize("provider_name, model", providers_to_test)
def test_aggregate_metadata_consistency_real(
    provider_name: str, model: str, work_dir: Path
):
    """Real integration: aggregate metadata is internally consistent for tool loops."""
    filename = f"metadata_aggregate_{provider_name}.txt"
    token = "ALLOS_METADATA_AGGREGATE_OK"

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
        f"Use write_file to create '{filename}' with exact content '{token}', "
        f"then use read_file to verify it, then answer with one sentence containing '{token}'."
    )
    result = agent.run(prompt)

    assert isinstance(result, str)
    assert result.strip() != ""

    metadata = agent.last_run_metadata
    assert metadata is not None
    assert _provider_matches(provider_name, metadata.model.provider)

    turn_history = metadata.turns.turn_history
    assert metadata.turns.total_turns == len(turn_history)
    assert metadata.turns.total_turns >= 2

    summed_input = sum(turn.tokens_used.input_tokens for turn in turn_history)
    summed_output = sum(turn.tokens_used.output_tokens for turn in turn_history)

    assert metadata.usage.input_tokens == summed_input
    assert metadata.usage.output_tokens == summed_output
    assert metadata.usage.total_tokens == (
        metadata.usage.input_tokens + metadata.usage.output_tokens
    )

    assert metadata.tools.total_tool_calls == len(metadata.tools.tool_calls)
    assert metadata.tools.total_tool_calls >= 1

    # Turn numbers should be sequential and start at 1.
    turn_numbers = [turn.turn_number for turn in turn_history]
    assert turn_numbers == list(range(1, len(turn_numbers) + 1))

    # Tool details should include execution outcomes and non-negative latency.
    for detail in metadata.tools.tool_calls:
        assert detail.tool_call_id != ""
        assert detail.tool_name != ""
        assert detail.status in {"success", "error", None}
        if detail.execution_time_ms is not None:
            assert detail.execution_time_ms >= 0

    file_path = work_dir / filename
    assert file_path.exists()
    assert token in file_path.read_text(encoding="utf-8")
