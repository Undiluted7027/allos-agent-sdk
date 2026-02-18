import os
from typing import Optional

import pytest

from allos.cli.utils import validate_api_key
from allos.providers import Message, MessageRole, ProviderRegistry
from allos.tools.base import BaseTool, ToolParameter

pytestmark = pytest.mark.integration


class GetWeatherTool(BaseTool):
    """Tool schema used to exercise alias endpoint tool calling."""

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
        return {"ok": True}


def _provider_matches(expected_alias: str, actual: str) -> bool:
    if actual == expected_alias:
        return True
    # Alias endpoints resolve via the chat_completions implementation.
    if expected_alias in {
        "together",
        "groq",
        "mistral",
        "deepseek",
        "openrouter",
        "cohere_compat",
        "portkey",
        "ollama_compat",
    } and actual in {"chat_completions", "openai"}:
        return True
    return False


def _assert_alias_metadata(response, alias: str) -> None:
    metadata = response.metadata
    assert metadata is not None
    assert _provider_matches(alias, metadata.model.provider)
    assert metadata.model.model_id != ""
    assert metadata.usage.input_tokens >= 0
    assert metadata.usage.output_tokens >= 0
    assert metadata.usage.total_tokens == (
        metadata.usage.input_tokens + metadata.usage.output_tokens
    )
    assert metadata.latency.total_duration_ms >= 0


def _alias_case(
    alias: str,
    *,
    env_var: Optional[str],  # Kept for readability in case declarations.
    default_model: str,
    tool_smoke_enabled: bool = True,
    require_model_env: bool = False,
    extra_marks: Optional[list] = None,
):
    marks = list(extra_marks or [])

    api_key_valid, api_key_message = validate_api_key(alias, None)
    if not api_key_valid:
        marks.append(
            pytest.mark.skipif(
                True,
                reason=f"{alias} not configured: {api_key_message}",
            )
        )

    model_env_var = f"TEST_{alias.upper()}_MODEL"
    if require_model_env:
        marks.append(
            pytest.mark.skipif(
                not os.getenv(model_env_var),
                reason=f"{model_env_var} not set",
            )
        )

    model = os.getenv(model_env_var, default_model)

    return pytest.param(alias, model, tool_smoke_enabled, marks=marks, id=alias)


ALIAS_CASES = [
    _alias_case(
        "together",
        env_var="TOGETHER_API_KEY",
        default_model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    ),
    _alias_case(
        "groq",
        env_var="GROQ_API_KEY",
        default_model="llama-3.1-8b-instant",
    ),
    _alias_case(
        "mistral",
        env_var="MISTRAL_API_KEY",
        default_model="mistral-small-latest",
    ),
    _alias_case(
        "deepseek",
        env_var="DEEPSEEK_API_KEY",
        default_model="deepseek-chat",
    ),
    _alias_case(
        "openrouter",
        env_var="OPENROUTER_API_KEY",
        default_model="openai/gpt-4o-mini",
    ),
    _alias_case(
        "cohere_compat",
        env_var="COHERE_API_KEY",
        default_model="command-a-03-2025",
        tool_smoke_enabled=True,
    ),
    _alias_case(
        "portkey",
        env_var="PORTKEY_API_KEY",
        default_model="@allos/gpt-4o-mini",
        tool_smoke_enabled=False,
        require_model_env=False,
    ),
    _alias_case(
        "ollama_compat",
        env_var=None,
        default_model=os.getenv("TEST_OLLAMA_COMPAT_MODEL", "qwen3:8b"),
        tool_smoke_enabled=True,
        extra_marks=[pytest.mark.requires_ollama],
    ),
]


@pytest.mark.parametrize("provider_name, model, _", ALIAS_CASES)
def test_alias_simple_chat_real(provider_name: str, model: str, _: bool):
    """Real smoke test: each configured alias can complete one basic chat turn."""
    provider = ProviderRegistry.get_provider(provider_name, model=model)

    messages = [
        Message(
            role=MessageRole.USER,
            content="Return exactly this token and nothing else: ALIAS_SMOKE_OK",
        )
    ]

    response = provider.chat(messages, temperature=0)

    assert response.content is not None
    assert response.content.strip() != ""
    assert "alias_smoke_ok" in response.content.lower()
    _assert_alias_metadata(response, provider_name)


@pytest.mark.parametrize("provider_name, model, tool_smoke_enabled", ALIAS_CASES)
def test_alias_tool_calling_smoke_real(
    provider_name: str,
    model: str,
    tool_smoke_enabled: bool,
):
    """Real smoke test for tool-calling on alias endpoints where reliability is acceptable."""
    if not tool_smoke_enabled:
        pytest.skip(f"{provider_name} tool-calling smoke is disabled for reliability.")

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

    response = provider.chat(messages, tools=tools, temperature=0)

    if not response.tool_calls:
        pytest.skip(
            f"{provider_name} did not emit a tool call in this real run (model-dependent)."
        )

    tool_call = response.tool_calls[0]
    assert tool_call.name == "get_current_weather"
    assert "location" in tool_call.arguments
    assert "boston" in str(tool_call.arguments["location"]).lower()
    _assert_alias_metadata(response, provider_name)
