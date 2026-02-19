import sys

import pytest

from allos.providers import Message, MessageRole, ProviderRegistry
from allos.tools.base import BaseTool, ToolParameter
from allos.utils.errors import ProviderError
from tests.conftest import PROVIDER_MODELS


class ImpossibleSchemaTool(BaseTool):
    """Tool with intentionally invalid parameter schema for provider API validation."""

    name = "impossible_schema_tool"
    description = "Tool with intentionally invalid JSON schema type."
    parameters = [
        ToolParameter(
            name="query",
            type="not_a_valid_json_schema_type",
            description="Intentionally invalid schema type for testing.",
            required=True,
        )
    ]

    def execute(self, **kwargs):
        return {"ok": True}


def _assert_normalized_provider_error(
    exc_info: pytest.ExceptionInfo[ProviderError],
    provider: str,
    expected_keywords: list[str],
) -> None:
    """Assert shared error normalization shape across providers."""
    err = exc_info.value
    text = str(err)

    assert isinstance(err, ProviderError)
    assert err.provider == provider
    assert text.startswith(f"ProviderError: [{provider}] ")
    assert len(text) > len(f"ProviderError: [{provider}] ")
    assert any(keyword in text.lower() for keyword in expected_keywords), text


@pytest.mark.integration
@pytest.mark.requires_openai
def test_openai_invalid_model_error_normalized_real():
    provider = ProviderRegistry.get_provider(
        "openai",
        model="__allos_invalid_model__",
    )
    messages = [Message(role=MessageRole.USER, content="hello")]

    with pytest.raises(ProviderError) as exc_info:
        provider.chat(messages, temperature=0)

    _assert_normalized_provider_error(
        exc_info,
        "openai",
        expected_keywords=["model", "notfound", "badrequest", "invalid"],
    )


@pytest.mark.integration
@pytest.mark.requires_anthropic
def test_anthropic_invalid_model_error_normalized_real():
    provider = ProviderRegistry.get_provider(
        "anthropic",
        model="__allos_invalid_model__",
    )
    messages = [Message(role=MessageRole.USER, content="hello")]

    with pytest.raises(ProviderError) as exc_info:
        provider.chat(messages, temperature=0)

    _assert_normalized_provider_error(
        exc_info,
        "anthropic",
        expected_keywords=["model", "bad request", "invalid", "error"],
    )


@pytest.mark.integration
@pytest.mark.requires_openai
def test_chat_completions_invalid_model_error_normalized_real():
    provider = ProviderRegistry.get_provider(
        "chat_completions",
        model="__allos_invalid_model__",
    )
    messages = [Message(role=MessageRole.USER, content="hello")]

    with pytest.raises(ProviderError) as exc_info:
        provider.chat(messages, temperature=0)

    _assert_normalized_provider_error(
        exc_info,
        "chat_completions",
        expected_keywords=["model", "notfound", "badrequest", "invalid"],
    )


@pytest.mark.integration
@pytest.mark.requires_ollama
def test_ollama_invalid_model_error_normalized_real():
    with pytest.raises(ProviderError) as exc_info:
        ProviderRegistry.get_provider("ollama", model="__allos_invalid_model__")

    _assert_normalized_provider_error(
        exc_info,
        "ollama",
        expected_keywords=["model", "not available", "pull"],
    )


@pytest.mark.integration
@pytest.mark.requires_cohere
def test_cohere_invalid_model_error_normalized_real():
    with pytest.raises(ProviderError) as exc_info:
        ProviderRegistry.get_provider("cohere", model="__allos_invalid_model__")

    _assert_normalized_provider_error(
        exc_info,
        "cohere",
        expected_keywords=["model", "not available", "verify"],
    )


@pytest.mark.integration
@pytest.mark.requires_gemini
@pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason="Google provider requires Python 3.10+ (google-auth>=2.48.0 dependency).",
)
def test_google_invalid_model_error_normalized_real():
    with pytest.raises(ProviderError) as exc_info:
        ProviderRegistry.get_provider("google", model="__allos_invalid_model__")

    _assert_normalized_provider_error(
        exc_info,
        "google",
        expected_keywords=["model", "not available", "could not verify"],
    )


@pytest.mark.integration
@pytest.mark.requires_openai
def test_openai_malformed_auth_error_normalized_real():
    provider = ProviderRegistry.get_provider(
        "openai",
        model=PROVIDER_MODELS["openai"],
        api_key="__allos_invalid_api_key__",
    )
    messages = [Message(role=MessageRole.USER, content="hello")]

    with pytest.raises(ProviderError) as exc_info:
        provider.chat(messages, temperature=0)

    _assert_normalized_provider_error(
        exc_info,
        "openai",
        expected_keywords=["auth", "api key", "invalid", "unauthorized"],
    )


@pytest.mark.integration
@pytest.mark.requires_cohere
def test_cohere_malformed_auth_error_normalized_real():
    with pytest.raises(ProviderError) as exc_info:
        ProviderRegistry.get_provider(
            "cohere",
            model=PROVIDER_MODELS["cohere"],
            api_key="__allos_invalid_api_key__",
        )

    _assert_normalized_provider_error(
        exc_info,
        "cohere",
        expected_keywords=["auth", "invalid", "unauthorized", "configuration"],
    )


@pytest.mark.integration
@pytest.mark.requires_gemini
@pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason="Google provider requires Python 3.10+ (google-auth>=2.48.0 dependency).",
)
def test_google_malformed_auth_error_normalized_real():
    with pytest.raises(ProviderError) as exc_info:
        ProviderRegistry.get_provider(
            "google",
            model=PROVIDER_MODELS["google"],
            api_key="__allos_invalid_api_key__",
        )

    _assert_normalized_provider_error(
        exc_info,
        "google",
        expected_keywords=["auth", "api key", "configuration", "credential", "invalid"],
    )


@pytest.mark.integration
@pytest.mark.requires_openai
def test_openai_impossible_tool_schema_error_normalized_real():
    provider = ProviderRegistry.get_provider("openai", model=PROVIDER_MODELS["openai"])
    tools = [ImpossibleSchemaTool()]
    messages = [
        Message(
            role=MessageRole.USER,
            content="Use impossible_schema_tool with query='test'.",
        )
    ]

    with pytest.raises(ProviderError) as exc_info:
        provider.chat(messages, tools=tools, temperature=0)

    _assert_normalized_provider_error(
        exc_info,
        "openai",
        expected_keywords=["badrequest", "invalid", "schema", "tool", "parameter"],
    )
