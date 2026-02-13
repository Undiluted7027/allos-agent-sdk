from pathlib import Path

import pytest

from allos import Agent, AgentConfig
from allos.providers.base import ProviderChunk
from tests.conftest import get_available_provider_params

providers_to_test = get_available_provider_params()


def _provider_matches(expected: str, actual: str) -> bool:
    if expected == "chat_completions" and actual == "openai":
        return True
    return expected == actual


@pytest.mark.integration
@pytest.mark.parametrize("provider_name, model", providers_to_test)
def test_agent_stream_run_tool_loop_real(
    provider_name: str, model: str, work_dir: Path
):
    """Real integration: Agent.stream_run executes tool loop and yields stream chunks."""
    filename = f"agent_stream_tool_loop_{provider_name}.txt"
    token = "ALLOS_STREAM_TOOL_OK"

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
        f"You must use write_file to create '{filename}' with exact content '{token}'. "
        f"Then use read_file to confirm it. "
        f"After tools complete, answer in one sentence containing '{token}'."
    )

    chunks = list(agent.stream_run(prompt))

    assert len(chunks) > 0
    assert all(isinstance(c, ProviderChunk) for c in chunks)

    error_chunks = [c for c in chunks if c.error]
    assert len(error_chunks) == 0

    content_chunks = [c for c in chunks if c.content]
    assert len(content_chunks) > 0
    full_text = "".join(c.content for c in content_chunks if c.content)
    assert token.lower() in full_text.lower()

    tool_chunks = [
        c for c in chunks if c.tool_call_start or c.tool_call_delta or c.tool_call_done
    ]
    assert len(tool_chunks) > 0

    final_metadata_chunks = [c for c in chunks if c.final_metadata]
    assert len(final_metadata_chunks) >= 1

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
    assert metadata.tools.total_tool_calls >= 1
    assert any(detail.tool_name == "write_file" for detail in metadata.tools.tool_calls)
