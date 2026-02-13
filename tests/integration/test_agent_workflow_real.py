# tests/integration/test_agent_workflow_real.py
import pytest

from allos import Agent, AgentConfig
from tests.conftest import get_available_provider_params

providers_to_test = get_available_provider_params()


def _provider_matches(expected: str, actual: str) -> bool:
    """Handle known provider-name normalization in metadata"""
    if expected == "chat_completions" and actual == "openai":
        return True
    return expected == actual


@pytest.mark.integration
@pytest.mark.parametrize("provider_name, model", providers_to_test)
def test_agent_run_basic_real(provider_name: str, model: str):
    """Agent.run returns content and metadata"""
    agent = Agent(
        AgentConfig(
            provider_name=provider_name, model=model, no_tools=True, max_iterations=2
        )
    )

    result = agent.run("Reply with one short sentence about the sky.")

    assert isinstance(result, str)
    assert result.strip() != ""
    assert agent.last_run_metadata is not None

    metadata = agent.last_run_metadata
    assert metadata is not None
    assert _provider_matches(provider_name, metadata.model.provider)
    assert metadata.model.model_id != ""
    assert metadata.usage.input_tokens >= 0
    assert metadata.usage.output_tokens >= 0
    assert metadata.usage.total_tokens == (
        metadata.usage.input_tokens + metadata.usage.output_tokens
    )
    assert metadata.latency.total_duration_ms >= 0
    assert metadata.turns.total_turns >= 1
    assert metadata.tools.total_tool_calls == 0
    assert len(agent.context.messages) >= 2
