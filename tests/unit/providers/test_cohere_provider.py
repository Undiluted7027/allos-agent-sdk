from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

pytest.importorskip("cohere")
from cohere.core import ApiError
from cohere.errors import BadRequestError, ServiceUnavailableError, UnauthorizedError

from allos.providers.base import Message, MessageRole, ToolCall
from allos.providers.cohere import CohereProvider
from allos.tools.base import BaseTool, ToolParameter
from allos.utils.errors import ProviderError


class DummyTool(BaseTool):
    name = "get_weather"
    description = "Gets weather for a location"
    parameters = [
        ToolParameter(
            name="location",
            type="string",
            description="City name",
            required=True,
        )
    ]

    def execute(self, **kwargs: Any):
        return {"ok": True}


def _event(event_type: str, **kwargs: Any) -> Any:
    return SimpleNamespace(type=event_type, **kwargs)


def _model(name: str, context_length: int | None = 128000) -> Any:
    return SimpleNamespace(name=name, context_length=context_length)


@pytest.fixture
def provider_and_client():
    with patch("allos.providers.cohere.cohere.ClientV2") as MockClient:
        mock_client = MockClient.return_value
        mock_client.models.list.return_value = SimpleNamespace(
            models=[
                _model("command-r7b-12-2024", context_length=128000),
                _model("command-r-plus-08-2024", context_length=128000),
            ]
        )
        provider = CohereProvider(model="command-r7b-12-2024", api_key="test-key")
        yield provider, mock_client, MockClient


def test_check_env_config(monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    configured, message = CohereProvider.check_env_config()
    assert configured is False
    assert message == "COHERE_API_KEY (Not Set)"

    monkeypatch.setenv("COHERE_API_KEY", "x")
    configured, message = CohereProvider.check_env_config()
    assert configured is True
    assert message == "COHERE_API_KEY (Set)"


@patch("allos.providers.cohere.cohere.ClientV2")
def test_provider_initialization_success_sets_context_window(MockClient):
    mock_client = MockClient.return_value
    mock_client.models.list.return_value = SimpleNamespace(
        models=[_model("command-r7b-12-2024", context_length=200000)]
    )

    provider = CohereProvider(model="command-r7b-12-2024", api_key="test-key")

    MockClient.assert_called_once_with(api_key="test-key")
    assert provider.get_context_window() == 200000


@patch("allos.providers.cohere.cohere.ClientV2")
def test_provider_initialization_model_not_available_has_suggestions(MockClient):
    mock_client = MockClient.return_value
    mock_client.models.list.return_value = SimpleNamespace(
        models=[
            _model("command-r7b-12-2024"),
            _model("command-r-plus-08-2024"),
            _model("command-a-03-2025"),
        ]
    )

    with pytest.raises(ProviderError) as exc_info:
        CohereProvider(model="command-r7b-12-2025", api_key="test-key")

    msg = str(exc_info.value)
    assert "not available" in msg
    assert "Did you mean" in msg
    assert "command-r7b-12-2024" in msg


@patch("allos.providers.cohere.cohere.ClientV2")
def test_provider_initialization_wraps_auth_errors(MockClient):
    MockClient.side_effect = UnauthorizedError(body={"message": "bad key"})

    with pytest.raises(ProviderError) as exc_info:
        CohereProvider(model="command-r7b-12-2024", api_key="bad")

    msg = str(exc_info.value)
    assert "Authentication/configuration error" in msg
    assert "bad key" in msg


@patch("allos.providers.cohere.cohere.ClientV2")
def test_provider_initialization_wraps_non_api_exception(MockClient):
    MockClient.side_effect = RuntimeError("socket error")

    with pytest.raises(ProviderError) as exc_info:
        CohereProvider(model="command-r7b-12-2024", api_key="bad")

    assert "Failed to initialize Cohere client" in str(exc_info.value)
    assert "socket error" in str(exc_info.value)


def test_safe_error_body_variants(provider_and_client):
    provider, _, _ = provider_and_client

    class DictError(Exception):
        body = {"error": "x"}

    class StrError(Exception):
        body = "simple error"

    class NoBodyError(Exception):
        pass

    assert provider._safe_error_body(DictError()) == '{"error": "x"}'
    assert provider._safe_error_body(StrError()) == "simple error"
    assert provider._safe_error_body(NoBodyError()) == ""


def test_safe_error_body_json_dumps_fallback_branch(provider_and_client):
    provider, _, _ = provider_and_client

    class Unserializable:
        def __str__(self):
            return "UNSERIALIZABLE_BODY"

    class BadBodyError(Exception):
        body = Unserializable()

    assert provider._safe_error_body(BadBodyError()) == "UNSERIALIZABLE_BODY"


def test_raise_provider_error_generic_fallback_branch(provider_and_client):
    provider, _, _ = provider_and_client

    with pytest.raises(ProviderError) as exc_info:
        provider._raise_provider_error(RuntimeError("unexpected"), "Prefix")

    assert "Prefix: unexpected" in str(exc_info.value)


def test_find_similar_models_prefix_fallback(provider_and_client):
    provider, _, _ = provider_and_client
    available = [
        _model("command-r-plus-08-2024"),
        _model("command-a-03-2025"),
        _model("embed-v4"),
    ]
    suggestions = provider._find_similar_models("command-z-foo", available)
    assert "command-a-03-2025" in suggestions
    assert "command-r-plus-08-2024" in suggestions


def test_find_similar_models_forced_no_close_matches_branch(provider_and_client):
    provider, _, _ = provider_and_client
    available = [_model("foo-alpha"), _model("foo-beta"), _model("bar-x")]

    with patch("difflib.get_close_matches", return_value=[]):
        suggestions = provider._find_similar_models("foo-unknown", available)

    assert suggestions == ["foo-alpha", "foo-beta"]


def test_convert_messages_formats_all_roles(provider_and_client):
    provider, _, _ = provider_and_client
    messages = [
        Message(role=MessageRole.SYSTEM, content="You are concise."),
        Message(role=MessageRole.USER, content="Weather in Boston?"),
        Message(
            role=MessageRole.ASSISTANT,
            content="I'll check.",
            tool_calls=[ToolCall("call_1", "get_weather", {"location": "Boston"})],
        ),
        Message(role=MessageRole.TOOL, content='{"temp":"72F"}', tool_call_id="call_1"),
    ]

    converted = provider._convert_messages(messages)

    assert [m.role for m in converted] == ["system", "user", "assistant", "tool"]
    assert converted[0].content == "You are concise."
    assert converted[1].content[0].text == "Weather in Boston?"
    assert converted[2].content[0].text == "I'll check."
    assert converted[2].tool_calls[0].id == "call_1"
    assert converted[2].tool_calls[0].function.name == "get_weather"
    assert converted[2].tool_calls[0].function.arguments == '{"location": "Boston"}'
    assert converted[3].tool_call_id == "call_1"
    assert converted[3].content == '{"temp":"72F"}'


def test_convert_tools_builds_schema():
    converted = CohereProvider._convert_tools([DummyTool()]) # pyright: ignore[reportAttributeAccessIssue]
    tool = converted[0]

    assert tool.type == "function"
    assert tool.function.name == "get_weather"
    assert tool.function.parameters["type"] == "object"
    assert tool.function.parameters["required"] == ["location"]
    assert tool.function.parameters["additionalProperties"] is False
    assert tool.function.parameters["properties"]["location"]["type"] == "string"


def test_parse_response_text_and_tool_calls(provider_and_client):
    provider, _, _ = provider_and_client
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="Hello "),
            SimpleNamespace(type="text", text="world"),
        ],
        tool_calls=[
            SimpleNamespace(
                id="call_1",
                function=SimpleNamespace(
                    name="get_weather", arguments='{"location":"Boston"}'
                ),
            )
        ],
    )

    content, tool_calls = provider._parse_response(response)

    assert content == "Hello world"
    assert len(tool_calls) == 1
    assert tool_calls[0].id == "call_1"
    assert tool_calls[0].name == "get_weather"
    assert tool_calls[0].arguments == {"location": "Boston"}


def test_parse_response_raises_for_malformed_tool_json(provider_and_client):
    provider, _, _ = provider_and_client
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="Working...")],
        tool_calls=[
            SimpleNamespace(
                id="call_1",
                function=SimpleNamespace(name="get_weather", arguments='{"location":'),
            )
        ],
    )

    with pytest.raises(ProviderError) as exc_info:
        provider._parse_response(response)

    assert "Failed to decode tool arguments" in str(exc_info.value)


def test_parse_response_no_content_branch(provider_and_client):
    provider, _, _ = provider_and_client
    response = SimpleNamespace(
        content=None,
        tool_calls=[
            SimpleNamespace(
                id="call_1",
                function=SimpleNamespace(name="get_weather", arguments='{"location":"SF"}'),
            )
        ],
    )

    content, tool_calls = provider._parse_response(response)
    assert content is None
    assert tool_calls == []


def test_parse_response_non_dict_tool_args_branch(provider_and_client):
    provider, _, _ = provider_and_client
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="Using tool...")],
        tool_calls=[
            SimpleNamespace(
                id="call_1",
                function=SimpleNamespace(name="get_weather", arguments='["not-a-dict"]'),
            )
        ],
    )

    _, tool_calls = provider._parse_response(response)
    assert len(tool_calls) == 1
    assert tool_calls[0].arguments == {}


@pytest.mark.parametrize(
    "usage, expected",
    [
        (None, (0, 0)),
        (
            SimpleNamespace(
                tokens=SimpleNamespace(input_tokens=11, output_tokens=7),
                billed_units=SimpleNamespace(input_tokens=99, output_tokens=88),
            ),
            (11, 7),
        ),
        (
            SimpleNamespace(
                tokens=SimpleNamespace(input_tokens=0, output_tokens=0),
                billed_units=SimpleNamespace(input_tokens=9, output_tokens=4),
            ),
            (9, 4),
        ),
    ],
)
def test_extract_usage_tokens(usage, expected):
    assert CohereProvider._extract_usage_tokens(usage) == expected # pyright: ignore[reportAttributeAccessIssue]


def test_safe_parse_tool_arguments_handles_non_dict(provider_and_client):
    provider, _, _ = provider_and_client

    parsed, error = provider._safe_parse_tool_arguments('["x"]')
    assert parsed == {}
    assert error is None

    parsed, error = provider._safe_parse_tool_arguments("{bad")
    assert parsed == {}
    assert error is not None


def test_chat_success_with_metadata_and_tools(provider_and_client):
    provider, mock_client, _ = provider_and_client
    mock_client.chat.return_value = SimpleNamespace(
        id="resp_123",
        message=SimpleNamespace(
            content=[SimpleNamespace(type="text", text="I can help.")],
            tool_calls=[
                SimpleNamespace(
                    id="call_1",
                    function=SimpleNamespace(
                        name="get_weather", arguments='{"location":"Boston"}'
                    ),
                )
            ],
        ),
        usage=SimpleNamespace(tokens=SimpleNamespace(input_tokens=12, output_tokens=8)),
    )

    response = provider.chat(
        [Message(role=MessageRole.USER, content="Weather?")],
        tools=[DummyTool()],
        temperature=0.2,
    )

    call_kwargs = mock_client.chat.call_args.kwargs
    assert call_kwargs["model"] == "command-r7b-12-2024"
    assert len(call_kwargs["messages"]) == 1
    assert call_kwargs["temperature"] == 0.2
    assert call_kwargs["tools"][0].function.name == "get_weather"

    assert response.content == "I can help."
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].arguments == {"location": "Boston"}
    assert response.metadata.usage.input_tokens == 12
    assert response.metadata.usage.output_tokens == 8
    assert response.metadata.model.provider == "cohere"
    assert response.metadata.tools.tools_available == ["get_weather"]
    assert response.metadata.tools.total_tool_calls == 1


@pytest.mark.parametrize(
    "error, expected_part",
    [
        (BadRequestError(body={"message": "bad req"}), "BadRequestError"),
        (
            ServiceUnavailableError(body={"message": "down"}),
            "Cohere API server error",
        ),
        (ApiError(body={"message": "oops"}), "Cohere API error"),
    ],
)
def test_chat_error_mapping(provider_and_client, error, expected_part):
    provider, mock_client, _ = provider_and_client
    mock_client.chat.side_effect = error

    with pytest.raises(ProviderError) as exc_info:
        provider.chat([Message(role=MessageRole.USER, content="Hi")])

    assert expected_part in str(exc_info.value)


def test_stream_chat_emits_text_tool_chunks_and_final_metadata(provider_and_client):
    provider, mock_client, _ = provider_and_client
    events = [
        _event("message-start", id="stream_1"),
        _event(
            "content-delta",
            delta=SimpleNamespace(
                message=SimpleNamespace(content=SimpleNamespace(text="Hello "))
            ),
        ),
        _event(
            "tool-call-start",
            index=0,
            delta=SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=SimpleNamespace(
                        id="call_1", function=SimpleNamespace(name="get_weather")
                    )
                )
            ),
        ),
        _event(
            "tool-call-delta",
            index=0,
            delta=SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=SimpleNamespace(
                        function=SimpleNamespace(arguments='{"location":"Bos')
                    )
                )
            ),
        ),
        _event(
            "tool-call-delta",
            index=0,
            delta=SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=SimpleNamespace(
                        function=SimpleNamespace(arguments='ton"}')
                    )
                )
            ),
        ),
        _event("tool-call-end", index=0),
        _event(
            "message-end",
            id="stream_1",
            delta=SimpleNamespace(
                finish_reason="COMPLETE",
                error=None,
                usage=SimpleNamespace(tokens=SimpleNamespace(input_tokens=20, output_tokens=6)),
            ),
        ),
    ]
    mock_client.chat_stream.return_value = iter(events)

    chunks = list(
        provider.stream_chat(
            [Message(role=MessageRole.USER, content="Hi")],
            tools=[DummyTool()],
            temperature=0.3,
        )
    )

    start_chunk = next(c for c in chunks if c.tool_call_start)
    delta_chunks = [c for c in chunks if c.tool_call_delta]
    done_chunk = next(c for c in chunks if c.tool_call_done is not None)
    final_chunk = chunks[-1]

    assert any(c.content == "Hello " for c in chunks)
    assert start_chunk.tool_call_start == {
        "id": "call_1",
        "name": "get_weather",
        "index": 0,
    }
    assert "".join(c.tool_call_delta for c in delta_chunks if c.tool_call_delta) == (
        '{"location":"Boston"}'
    )
    assert done_chunk.tool_call_done is not None
    assert done_chunk.tool_call_done.arguments == {"location": "Boston"}

    call_kwargs = mock_client.chat_stream.call_args.kwargs
    assert call_kwargs["temperature"] == 0.3
    assert call_kwargs["tools"][0].function.name == "get_weather"

    assert final_chunk.final_metadata is not None
    assert final_chunk.final_metadata.request_id == "stream_1"
    assert final_chunk.final_metadata.usage.input_tokens == 20
    assert final_chunk.final_metadata.usage.output_tokens == 6
    assert final_chunk.final_metadata.tools.total_tool_calls == 1


@patch("allos.providers.cohere.cohere.ClientV2")
def test_provider_initialization_wraps_api_error_branch(MockClient):
    MockClient.side_effect = ApiError(body={"message": "init api error"})

    with pytest.raises(ProviderError) as exc_info:
        CohereProvider(model="command-r7b-12-2024", api_key="test-key")

    assert "Cohere API error" in str(exc_info.value)
    assert "init api error" in str(exc_info.value)


@patch("allos.providers.cohere.cohere.ClientV2")
def test_verify_model_available_no_suggestions_else_branch(MockClient):
    mock_client = MockClient.return_value
    mock_client.models.list.return_value = SimpleNamespace(
        models=[_model("alpha-model"), _model("beta-model"), _model("gamma-model")]
    )

    with patch.object(CohereProvider, "_find_similar_models", return_value=[]):
        with pytest.raises(ProviderError) as exc_info:
            CohereProvider(model="zzzz-model", api_key="test-key")

    msg = str(exc_info.value)
    assert "Model 'zzzz-model' not available." in msg
    assert "Available models:" in msg
    assert "Did you mean" not in msg


def test_verify_model_available_api_error_branch(provider_and_client):
    provider, mock_client, _ = provider_and_client
    mock_client.models.list.side_effect = ApiError(body={"message": "list failed"})

    with pytest.raises(ProviderError) as exc_info:
        provider._verify_model_available()

    assert "Could not verify model" in str(exc_info.value)
    assert "list failed" in str(exc_info.value)


def test_stream_chat_message_end_error_yields_error_chunk(provider_and_client):
    provider, mock_client, _ = provider_and_client
    mock_client.chat_stream.return_value = iter(
        [
            _event("message-start", id="stream_err"),
            _event(
                "message-end",
                id="stream_err",
                delta=SimpleNamespace(
                    finish_reason="ERROR",
                    error="provider-side failure",
                    usage=SimpleNamespace(tokens=SimpleNamespace(input_tokens=3, output_tokens=0)),
                ),
            ),
        ]
    )

    chunks = list(provider.stream_chat([Message(role=MessageRole.USER, content="Hi")]))

    assert any(
        c.error and "Cohere stream error: provider-side failure" in c.error
        for c in chunks
    )
    assert chunks[-1].final_metadata is not None


def test_on_content_delta_no_text_branch(provider_and_client):
    provider, _, _ = provider_and_client
    chunks = provider._on_content_delta(
        _event("content-delta", delta=SimpleNamespace(message=SimpleNamespace(content=None))),
        {},
    )
    assert chunks == []


def test_on_tool_call_start_guard_branches(provider_and_client):
    provider, _, _ = provider_and_client
    state = provider._initialize_stream_state()

    assert provider._on_tool_call_start(_event("tool-call-start"), state) == []
    assert (
        provider._on_tool_call_start(_event("tool-call-start", index=0), state) == []
    )
    assert (
        provider._on_tool_call_start(
            _event(
                "tool-call-start",
                index=0,
                delta=SimpleNamespace(
                    message=SimpleNamespace(
                        tool_calls=SimpleNamespace(
                            id="call_1",
                            function=SimpleNamespace(name=None),
                        )
                    )
                ),
            ),
            state,
        )
        == []
    )


def test_on_tool_call_delta_guard_branches(provider_and_client):
    provider, _, _ = provider_and_client
    state = provider._initialize_stream_state()
    state["in_progress_tool_calls"][0] = {"id": "call_1", "name": "tool_x", "arguments": ""}

    assert provider._on_tool_call_delta(_event("tool-call-delta"), state) == []
    assert provider._on_tool_call_delta(_event("tool-call-delta", index=2), state) == []

    chunks = provider._on_tool_call_delta(
        _event(
            "tool-call-delta",
            index=0,
            delta=SimpleNamespace(message=SimpleNamespace(tool_calls=SimpleNamespace(function=SimpleNamespace(arguments=None)))),
        ),
        state,
    )
    assert chunks == []


def test_on_tool_call_end_guard_branches(provider_and_client):
    provider, _, _ = provider_and_client
    state = provider._initialize_stream_state()
    assert provider._on_tool_call_end(_event("tool-call-end"), state) == []
    assert provider._on_tool_call_end(_event("tool-call-end", index=0), state) == []


def test_dispatch_stream_event_no_handler_branch(provider_and_client):
    provider, _, _ = provider_and_client
    state = provider._initialize_stream_state()
    chunks = provider._dispatch_stream_event(_event("unknown-event"), state)
    assert chunks == []


def test_stream_chat_flushes_incomplete_tool_calls(provider_and_client):
    provider, mock_client, _ = provider_and_client
    mock_client.chat_stream.return_value = iter(
        [
            _event("message-start", id="stream_partial"),
            _event(
                "tool-call-start",
                index=0,
                delta=SimpleNamespace(
                    message=SimpleNamespace(
                        tool_calls=SimpleNamespace(
                            id="call_99", function=SimpleNamespace(name="get_weather")
                        )
                    )
                ),
            ),
            _event(
                "tool-call-delta",
                index=0,
                delta=SimpleNamespace(
                    message=SimpleNamespace(
                        tool_calls=SimpleNamespace(
                            function=SimpleNamespace(arguments='{"location":"NYC"}')
                        )
                    )
                ),
            ),
            _event(
                "message-end",
                id="stream_partial",
                delta=SimpleNamespace(
                    finish_reason="COMPLETE",
                    error=None,
                    usage=SimpleNamespace(tokens=SimpleNamespace(input_tokens=5, output_tokens=3)),
                ),
            ),
        ]
    )

    chunks = list(provider.stream_chat([Message(role=MessageRole.USER, content="Hi")]))
    final = chunks[-1].final_metadata

    assert final is not None
    assert final.tools.total_tool_calls == 1
    assert final.tools.tool_calls[0].tool_call_id == "call_99"
    assert final.tools.tool_calls[0].arguments == {"location": "NYC"}


def test_stream_chat_tool_parse_error_yields_error_chunk(provider_and_client):
    provider, mock_client, _ = provider_and_client
    mock_client.chat_stream.return_value = iter(
        [
            _event("message-start", id="stream_bad_tool"),
            _event(
                "tool-call-start",
                index=0,
                delta=SimpleNamespace(
                    message=SimpleNamespace(
                        tool_calls=SimpleNamespace(
                            id="call_bad", function=SimpleNamespace(name="get_weather")
                        )
                    )
                ),
            ),
            _event(
                "tool-call-delta",
                index=0,
                delta=SimpleNamespace(
                    message=SimpleNamespace(
                        tool_calls=SimpleNamespace(function=SimpleNamespace(arguments='{"x":'))
                    )
                ),
            ),
            _event("tool-call-end", index=0),
            _event(
                "message-end",
                id="stream_bad_tool",
                delta=SimpleNamespace(
                    finish_reason="COMPLETE",
                    error=None,
                    usage=SimpleNamespace(tokens=SimpleNamespace(input_tokens=1, output_tokens=1)),
                ),
            ),
        ]
    )

    chunks = list(provider.stream_chat([Message(role=MessageRole.USER, content="Hi")]))

    assert any(c.error and "Failed to parse tool arguments" in c.error for c in chunks)
    done_chunk = next(c for c in chunks if c.tool_call_done is not None)
    assert done_chunk.tool_call_done is not None
    assert done_chunk.tool_call_done.arguments == {}


@pytest.mark.parametrize(
    "error, expected_part",
    [
        (BadRequestError(body={"message": "bad req"}), "BadRequestError"),
        (
            ServiceUnavailableError(body={"message": "down"}),
            "Cohere API streaming server error",
        ),
        (ApiError(body={"message": "stream oops"}), "Cohere API streaming error"),
    ],
)
def test_stream_chat_error_mapping(provider_and_client, error, expected_part):
    provider, mock_client, _ = provider_and_client
    mock_client.chat_stream.side_effect = error

    with pytest.raises(ProviderError) as exc_info:
        list(provider.stream_chat([Message(role=MessageRole.USER, content="Hi")]))

    assert expected_part in str(exc_info.value)


@patch("allos.providers.cohere.cohere.ClientV2")
def test_get_context_window_fallback(MockClient):
    mock_client = MockClient.return_value
    mock_client.models.list.return_value = SimpleNamespace(
        models=[_model("command-r7b-12-2024", context_length=None)]
    )

    provider = CohereProvider(model="command-r7b-12-2024", api_key="test-key")
    assert provider.get_context_window() == 4096
