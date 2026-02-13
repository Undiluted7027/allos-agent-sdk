import pytest

from allos import Agent, AgentConfig
from allos.providers import Message, MessageRole, ProviderRegistry
from allos.providers.base import ProviderChunk
from allos.tools.base import BaseTool, ToolParameter
from tests.conftest import get_available_provider_params

providers_to_test = get_available_provider_params()


def _provider_matches(expected: str, actual: str) -> bool:
    """Handle known provider-name normalization in metadata."""
    if expected == "chat_completions" and actual == "openai":
        return True
    return expected == actual


def _assert_basic_metadata(metadata, provider_name: str) -> None:
    assert metadata is not None
    assert _provider_matches(provider_name, metadata.model.provider)
    assert metadata.model.model_id != ""
    assert metadata.usage.input_tokens >= 0
    assert metadata.usage.output_tokens >= 0
    assert metadata.usage.total_tokens == (
        metadata.usage.input_tokens + metadata.usage.output_tokens
    )
    assert metadata.latency.total_duration_ms >= 0


class GetWeatherTool(BaseTool):
    """Tool schema used to exercise provider-side streaming tool calling."""

    name = "get_current_weather"
    description = "Get the current weather in a given location"
    parameters = [
        ToolParameter(
            name="location",
            type="string",
            description="City and state, e.g. Boston, MA",
            required=True,
        )
    ]

    def execute(self, **kwargs):
        # Not used in this test: we only validate provider stream/tool-call behavior.
        return {"ok": True}


@pytest.mark.integration
@pytest.mark.parametrize("provider_name, model", providers_to_test)
def test_stream_run_basic_real(provider_name: str, model: str):
    """Real streaming: stream_run yields content and records metadata."""
    agent = Agent(
        AgentConfig(
            provider_name=provider_name,
            model=model,
            no_tools=True,
            max_iterations=3,
        )
    )

    chunks = list(agent.stream_run("In one sentence, explain recursion."))

    assert len(chunks) > 0
    assert all(isinstance(c, ProviderChunk) for c in chunks)

    error_chunks = [c for c in chunks if c.error]
    assert len(error_chunks) == 0

    content_chunks = [c for c in chunks if c.content]
    assert len(content_chunks) > 0

    final_metadata_chunks = [c for c in chunks if c.final_metadata]
    assert len(final_metadata_chunks) >= 1
    for final_chunk in final_metadata_chunks:
        _assert_basic_metadata(final_chunk.final_metadata, provider_name)

    full_text = "".join(c.content for c in content_chunks if c.content)
    assert full_text.strip() != ""

    assert agent.last_run_metadata is not None
    metadata = agent.last_run_metadata
    _assert_basic_metadata(metadata, provider_name)


@pytest.mark.integration
@pytest.mark.parametrize("provider_name, model", providers_to_test)
def test_stream_with_tool_calling_real(provider_name: str, model: str):
    """Real provider streaming should surface tool-calling chunks."""
    provider = ProviderRegistry.get_provider(provider_name, model=model)
    tools = [GetWeatherTool()]
    messages = [
        Message(
            role=MessageRole.USER,
            content=(
                "Use the get_current_weather tool for Boston, MA. "
                "Do not answer from memory."
            ),
        )
    ]

    chunks = list(provider.stream_chat(messages, tools=tools, temperature=0))

    assert len(chunks) > 0
    assert all(isinstance(c, ProviderChunk) for c in chunks)

    error_chunks = [c for c in chunks if c.error]
    assert len(error_chunks) == 0

    tool_related_chunks = [
        c
        for c in chunks
        if c.tool_call_start or c.tool_call_delta or c.tool_call_done
    ]
    assert len(tool_related_chunks) > 0, (
        f"{provider_name} stream did not emit any tool-calling chunks."
    )

    completed_calls = [c.tool_call_done for c in chunks if c.tool_call_done]
    if completed_calls:
        first_call = completed_calls[0]
        assert first_call is not None
        assert first_call.name == "get_current_weather"
        assert "location" in first_call.arguments
        assert "boston" in str(first_call.arguments["location"]).lower()

    final_metadata_chunks = [c for c in chunks if c.final_metadata]
    assert len(final_metadata_chunks) >= 1
    for final_chunk in final_metadata_chunks:
        _assert_basic_metadata(final_chunk.final_metadata, provider_name)
