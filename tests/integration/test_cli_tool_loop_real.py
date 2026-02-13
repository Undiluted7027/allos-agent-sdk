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


CLI_PROVIDER_CASES = [
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
@pytest.mark.parametrize("provider_name, model", CLI_PROVIDER_CASES)
def test_cli_run_tool_loop_session_roundtrip_real(
    runner: CliRunner,
    provider_name: str,
    model: str,
    work_dir: Path,
):
    """Real CLI integration: run mode executes tools and persists session continuity."""
    filename = f"cli_tool_loop_{provider_name}.txt"
    token = "ALLOS_CLI_TOOL_OK"
    session_path = work_dir / f"cli_tool_loop_{provider_name}.json"

    prompt_one = (
        f"Use write_file to create '{filename}' with exact content '{token}'. "
        f"Then use read_file for '{filename}'. "
        f"Return one sentence containing '{token}'."
    )
    prompt_two = (
        f"Now read_file '{filename}' again and return one sentence containing '{token}'."
    )

    result_one = runner.invoke(
        main,
        [
            "--provider",
            provider_name,
            "--model",
            model,
            "--tool",
            "write_file",
            "--tool",
            "read_file",
            "--auto-approve",
            "--session",
            str(session_path),
            prompt_one,
        ],
        catch_exceptions=False,
    )

    assert result_one.exit_code == 0
    assert session_path.exists()

    file_path = work_dir / filename
    assert file_path.exists()
    assert token in file_path.read_text(encoding="utf-8")

    messages_after_first_run = _load_session_messages(session_path)
    assert prompt_one in [
        msg.get("content")
        for msg in messages_after_first_run
        if msg.get("role") == "user"
    ]

    result_two = runner.invoke(
        main,
        [
            "--provider",
            provider_name,
            "--model",
            model,
            "--tool",
            "write_file",
            "--tool",
            "read_file",
            "--auto-approve",
            "--session",
            str(session_path),
            prompt_two,
        ],
        catch_exceptions=False,
    )

    assert result_two.exit_code == 0
    assert "Loading session from" in result_two.output

    messages_after_second_run = _load_session_messages(session_path)
    assert len(messages_after_second_run) > len(messages_after_first_run)
    user_messages = [
        msg.get("content")
        for msg in messages_after_second_run
        if msg.get("role") == "user"
    ]
    assert prompt_one in user_messages
    assert prompt_two in user_messages


@pytest.mark.integration
@pytest.mark.parametrize("provider_name, model", CLI_PROVIDER_CASES)
def test_cli_stream_tool_loop_session_roundtrip_real(
    runner: CliRunner,
    provider_name: str,
    model: str,
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Real CLI integration: stream mode executes tools and persists session continuity."""
    stream_chunks = []
    original_print_stream_chunk = cli_main_module._print_stream_chunk

    def _capture_and_print(chunk):
        stream_chunks.append(chunk)
        original_print_stream_chunk(chunk)

    monkeypatch.setattr(cli_main_module, "_print_stream_chunk", _capture_and_print)

    filename = f"cli_stream_tool_loop_{provider_name}.txt"
    token = "ALLOS_CLI_STREAM_TOOL_OK"
    session_path = work_dir / f"cli_stream_tool_loop_{provider_name}.json"

    prompt_one = (
        f"Use write_file to create '{filename}' with exact content '{token}'. "
        f"Then use read_file for '{filename}'. "
        f"Return one sentence containing '{token}'."
    )
    prompt_two = (
        f"Now read_file '{filename}' again and return one sentence containing '{token}'."
    )

    result_one = runner.invoke(
        main,
        [
            "--stream",
            "--provider",
            provider_name,
            "--model",
            model,
            "--tool",
            "write_file",
            "--tool",
            "read_file",
            "--auto-approve",
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
    assert len(stream_chunks) > 1
    assert any(chunk.content for chunk in stream_chunks)

    file_path = work_dir / filename
    assert file_path.exists()
    assert token in file_path.read_text(encoding="utf-8")

    messages_after_first_run = _load_session_messages(session_path)
    assert prompt_one in [
        msg.get("content")
        for msg in messages_after_first_run
        if msg.get("role") == "user"
    ]

    stream_chunks.clear()

    result_two = runner.invoke(
        main,
        [
            "--stream",
            "--provider",
            provider_name,
            "--model",
            model,
            "--tool",
            "write_file",
            "--tool",
            "read_file",
            "--auto-approve",
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
    user_messages = [
        msg.get("content")
        for msg in messages_after_second_run
        if msg.get("role") == "user"
    ]
    assert prompt_one in user_messages
    assert prompt_two in user_messages
