import json
import os
from pathlib import Path

import pytest
from click.testing import CliRunner

import allos.cli.main as cli_main_module
from allos.cli.main import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


PROVIDER_CASES = [
    pytest.param(
        "openai",
        "gpt-4o",
        marks=[pytest.mark.requires_openai],
        id="openai",
    ),
    pytest.param(
        "ollama",
        os.getenv("TEST_OLLAMA_MODEL", "qwen3:8b"),
        marks=[pytest.mark.requires_ollama],
        id="ollama",
    ),
]


def _load_session_messages(session_path: Path):
    with session_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data["context"]["messages"]


@pytest.mark.integration
@pytest.mark.parametrize("provider_name, model", PROVIDER_CASES)
def test_cli_stream_mode_real_session_roundtrip(
    runner: CliRunner,
    provider_name: str,
    model: str,
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Real CLI smoke: stream mode succeeds and session save/load continues context."""
    stream_chunks = []
    original_print_stream_chunk = cli_main_module._print_stream_chunk

    def _capture_and_print(chunk):
        stream_chunks.append(chunk)
        original_print_stream_chunk(chunk)

    monkeypatch.setattr(cli_main_module, "_print_stream_chunk", _capture_and_print)

    session_path = work_dir / f"stream_session_{provider_name}.json"
    prompt_one = "Write three short sentences about stream testing. Prefix with STREAM_ONE."
    prompt_two = "Write three short sentences about session loading. Prefix with STREAM_TWO."

    result_one = runner.invoke(
        main,
        [
            "--stream",
            "--provider",
            provider_name,
            "--model",
            model,
            "--no-tools",
            "--session",
            str(session_path),
            prompt_one,
        ],
        catch_exceptions=False,
    )

    assert result_one.exit_code == 0
    assert "Streaming Response" in result_one.output
    assert "Stream Error:" not in result_one.output
    assert session_path.exists()

    # Incremental streaming should produce multiple chunk print events.
    assert len(stream_chunks) > 1
    assert any(chunk.content for chunk in stream_chunks)

    messages_after_first_run = _load_session_messages(session_path)
    user_prompts_after_first_run = [
        msg.get("content")
        for msg in messages_after_first_run
        if msg.get("role") == "user"
    ]
    assert prompt_one in user_prompts_after_first_run

    stream_chunks.clear()

    result_two = runner.invoke(
        main,
        [
            "--stream",
            "--provider",
            provider_name,
            "--model",
            model,
            "--no-tools",
            "--session",
            str(session_path),
            prompt_two,
        ],
        catch_exceptions=False,
    )

    assert result_two.exit_code == 0
    assert "Loading session from" in result_two.output
    assert "Streaming Response" in result_two.output
    assert "Stream Error:" not in result_two.output
    assert len(stream_chunks) > 1
    assert any(chunk.content for chunk in stream_chunks)

    messages_after_second_run = _load_session_messages(session_path)
    assert len(messages_after_second_run) > len(messages_after_first_run)

    user_prompts_after_second_run = [
        msg.get("content")
        for msg in messages_after_second_run
        if msg.get("role") == "user"
    ]
    assert prompt_one in user_prompts_after_second_run
    assert prompt_two in user_prompts_after_second_run
