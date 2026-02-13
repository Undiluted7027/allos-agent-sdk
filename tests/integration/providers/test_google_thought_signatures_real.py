import sys
from unittest.mock import patch

import pytest

if sys.version_info < (3, 10):
    pytest.skip(
        "Google provider requires Python 3.10+ (google-auth>=2.48.0 dependency). "
        "All Google provider tests skipped on Python 3.9.",
        allow_module_level=True,
    )

from allos.providers import Message, MessageRole, ProviderRegistry
from allos.tools.base import BaseTool, ToolParameter

pytestmark = [pytest.mark.integration, pytest.mark.requires_gemini]


class GetWeatherTool(BaseTool):
    """A tool schema used for real Google tool-calling tests."""

    name = "get_current_weather"
    description = "Get the current weather in a given location"
    parameters = [
        ToolParameter(
            name="location",
            type="string",
            description="The city and state, e.g. Boston, MA",
            required=True,
        )
    ]

    def execute(self, **kwargs):
        # Tool execution is not needed here; provider-only flow test.
        return {"ok": True}


def _extract_forwarded_signatures(contents):
    """Extract thought signatures attached to function_call parts in request contents."""
    forwarded = []
    for content in contents or []:
        for part in getattr(content, "parts", []) or []:
            if getattr(part, "function_call", None) is None:
                continue
            thought_sig = getattr(part, "thought_signature", None)
            if thought_sig:
                forwarded.append(thought_sig)
    return forwarded


def test_google_thought_signatures_roundtrip_real():
    """Validate real thought signatures are emitted then forwarded in next turn."""
    provider = ProviderRegistry.get_provider("google", model="gemini-3-flash-preview")
    tools = [GetWeatherTool()]

    turn1_messages = [
        Message(
            role=MessageRole.USER,
            content=(
                "Use get_current_weather for Boston, MA before answering. "
                "Do not answer from memory."
            ),
        )
    ]

    with patch.object(
        provider.client.models,  # pyright: ignore[reportAttributeAccessIssue]
        "generate_content",
        wraps=provider.client.models.generate_content,  # pyright: ignore[reportAttributeAccessIssue]
    ) as wrapped_generate_content:
        turn1_response = provider.chat(turn1_messages, tools=tools, temperature=0)

        if not turn1_response.tool_calls:
            pytest.skip("Model did not issue a tool call in this real run.")

        if not turn1_response.thought_signatures:
            pytest.skip("Model did not emit thought signatures in this real run.")

        first_tool_call = turn1_response.tool_calls[0]
        if first_tool_call.id not in turn1_response.thought_signatures:
            pytest.skip(
                "Thought signatures present but not mapped to first tool call id."
            )

        turn2_messages = [
            turn1_messages[0],
            Message(
                role=MessageRole.ASSISTANT,
                content=turn1_response.content,
                tool_calls=turn1_response.tool_calls,
                thought_signatures=turn1_response.thought_signatures,
            ),
            Message(
                role=MessageRole.TOOL,
                tool_call_id=first_tool_call.id,
                content='{"temperature":"8C","condition":"cloudy"}',
            ),
            Message(
                role=MessageRole.USER,
                content="Using the tool output, answer in one sentence.",
            ),
        ]

        turn2_response = provider.chat(turn2_messages, tools=tools, temperature=0)

    assert turn2_response.metadata is not None
    assert turn2_response.metadata.model.provider == "google"
    assert turn2_response.metadata.usage.total_tokens >= 0

    assert len(wrapped_generate_content.call_args_list) >= 2
    second_call_contents = wrapped_generate_content.call_args_list[-1].kwargs.get(
        "contents"
    )
    assert second_call_contents is not None

    forwarded_signatures = _extract_forwarded_signatures(second_call_contents)
    assert forwarded_signatures, (
        "Expected at least one thought signature to be forwarded in turn-2 request."
    )

    emitted_signatures = set(turn1_response.thought_signatures.values())
    assert any(sig in emitted_signatures for sig in forwarded_signatures), (
        "No forwarded thought signature matched turn-1 emitted signatures."
    )
