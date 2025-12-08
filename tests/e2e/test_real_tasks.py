# tests/e2e/test_real_tasks.py

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from allos.cli.main import main
from allos.providers.base import Message, ProviderResponse, ToolCall
from allos.utils.token_counter import count_tokens

pytestmark = pytest.mark.e2e


# --- Fixtures ---


@pytest.fixture
def runner() -> CliRunner:
    """Provides a CliRunner instance for invoking the CLI."""
    return CliRunner()


# --- E2E Tests with Mocked LLM ---


@patch("allos.providers.openai.OpenAIProvider.chat")
def test_task_create_hello_world_script(
    mock_chat, runner: CliRunner, work_dir: Path, mock_metadata_factory
):
    """
    E2E Test: "Create a Python script that prints Hello World"
    - Mocks the LLM but uses the real Agent, Tools, and Filesystem.
    """
    metadata_history = []
    turn_intents = [
        {
            "tool_calls": [
                ToolCall(
                    id="call_1",
                    name="write_file",
                    arguments={
                        "path": "hello.py",
                        "content": "print('Hello, World!')",
                    },
                )
            ]
        },
        {"content": "The file 'hello.py' has been created."},
    ]

    def dynamic_chat_side_effect(messages: list[Message], **kwargs) -> ProviderResponse:
        call_index = mock_chat.call_count - 1
        intent = turn_intents[call_index]
        input_tokens = count_tokens(" ".join([m.content or "" for m in messages]))
        metadata = mock_metadata_factory(usage={"input_tokens": input_tokens})
        metadata_history.append(metadata)
        return ProviderResponse(
            content=intent.get("content"),
            tool_calls=intent.get("tool_calls", []),
            metadata=metadata,
        )

    mock_chat.side_effect = dynamic_chat_side_effect

    result = runner.invoke(
        main,
        ["--auto-approve", "Create a Python script that prints Hello World"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert "The file 'hello.py' has been created." in result.output
    assert len(metadata_history) == 2
    assert (
        metadata_history[1].usage.input_tokens > metadata_history[0].usage.input_tokens
    )

    created_file = work_dir / "hello.py"
    assert created_file.exists()
    assert created_file.read_text() == "print('Hello, World!')"


@patch("allos.providers.openai.OpenAIProvider.chat")
def test_task_read_and_count_lines(
    mock_chat, runner: CliRunner, work_dir: Path, mock_metadata_factory
):
    """
    E2E Test: "Read main.py and count the lines"
    """
    (work_dir / "main.py").write_text("line 1\nline 2\nline 3")
    metadata_history = []
    turn_intents = [
        {"tool_calls": [ToolCall("call_1", "read_file", {"path": "main.py"})]},
        {"content": "The file 'main.py' contains 3 lines."},
    ]

    def dynamic_chat_side_effect(messages: list[Message], **kwargs) -> ProviderResponse:
        call_index = mock_chat.call_count - 1
        intent = turn_intents[call_index]
        input_tokens = count_tokens(" ".join([m.content or "" for m in messages]))
        metadata = mock_metadata_factory(usage={"input_tokens": input_tokens})
        metadata_history.append(metadata)
        return ProviderResponse(
            content=intent.get("content"),
            tool_calls=intent.get("tool_calls", []),
            metadata=metadata,
        )

    mock_chat.side_effect = dynamic_chat_side_effect

    result = runner.invoke(main, ["Read main.py and count the lines"])

    assert result.exit_code == 0
    assert "The file 'main.py' contains 3 lines." in result.output
    assert mock_chat.call_count == 2
    assert (
        metadata_history[1].usage.input_tokens > metadata_history[0].usage.input_tokens
    )


@patch("allos.providers.openai.OpenAIProvider.chat")
def test_task_fix_syntax_error(
    mock_chat, runner: CliRunner, work_dir: Path, mock_metadata_factory
):
    """
    E2E Test: "Fix syntax error in test.py"
    This tests tool chaining: read -> edit
    """
    buggy_code = "def my_function()\n    print('hello')"
    (work_dir / "test.py").write_text(buggy_code)
    metadata_history = []
    turn_intents = [
        {"tool_calls": [ToolCall("call_1", "read_file", {"path": "test.py"})]},
        {
            "tool_calls": [
                ToolCall(
                    "call_2",
                    "edit_file",
                    {
                        "path": "test.py",
                        "find_string": "def my_function()",
                        "replace_with": "def my_function():",
                    },
                )
            ]
        },
        {"content": "I have fixed the syntax error."},
    ]

    def dynamic_chat_side_effect(messages: list[Message], **kwargs) -> ProviderResponse:
        call_index = mock_chat.call_count - 1
        intent = turn_intents[call_index]
        input_tokens = count_tokens(" ".join([m.content or "" for m in messages]))
        metadata = mock_metadata_factory(usage={"input_tokens": input_tokens})
        metadata_history.append(metadata)
        return ProviderResponse(
            content=intent.get("content"),
            tool_calls=intent.get("tool_calls", []),
            metadata=metadata,
        )

    mock_chat.side_effect = dynamic_chat_side_effect

    result = runner.invoke(
        main,
        ["--auto-approve", "Fix the syntax error in test.py"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert "fixed the syntax error" in result.output
    assert (
        work_dir / "test.py"
    ).read_text() == "def my_function():\n    print('hello')"
    assert len(metadata_history) == 3
    assert (
        metadata_history[1].usage.input_tokens > metadata_history[0].usage.input_tokens
    )
    assert (
        metadata_history[2].usage.input_tokens > metadata_history[1].usage.input_tokens
    )


@patch("allos.providers.openai.OpenAIProvider.chat")
def test_e2e_multi_turn_tool_chaining_workflow(
    mock_chat, runner: CliRunner, work_dir: Path, mock_metadata_factory
):
    """
    Tests a complete agent workflow via the CLI with dynamic metadata.
    """
    metadata_history = []
    turn_intents = [
        {"tool_calls": [ToolCall("1", "list_directory", {})]},
        {"tool_calls": [ToolCall("2", "read_file", {"path": "instructions.txt"})]},
        {
            "tool_calls": [
                ToolCall(
                    "3",
                    "write_file",
                    {"path": "app.py", "content": "print('Hello from chained tool!')"},
                )
            ]
        },
        {"tool_calls": [ToolCall("4", "shell_exec", {"command": "python app.py"})]},
        {
            "content": "I have created and executed the script. The output was 'Hello from chained tool!'."
        },
    ]

    def dynamic_chat_side_effect(messages: list[Message], **kwargs) -> ProviderResponse:
        call_index = mock_chat.call_count - 1
        intent = turn_intents[call_index]
        input_tokens = count_tokens(" ".join([m.content or "" for m in messages]))
        metadata = mock_metadata_factory(usage={"input_tokens": input_tokens})
        metadata_history.append(metadata)
        return ProviderResponse(
            content=intent.get("content"),
            tool_calls=intent.get("tool_calls", []),
            metadata=metadata,
        )

    mock_chat.side_effect = dynamic_chat_side_effect

    (work_dir / "instructions.txt").write_text(
        "Create a python script named app.py that prints 'Hello from chained tool!'"
    )

    result = runner.invoke(
        main,
        ["--auto-approve", "Follow the instructions in the text file."],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert "Hello from chained tool!" in result.output
    assert mock_chat.call_count == 5

    assert len(metadata_history) == 5
    assert (
        metadata_history[1].usage.input_tokens > metadata_history[0].usage.input_tokens
    )
    assert (
        metadata_history[4].usage.input_tokens > metadata_history[3].usage.input_tokens
    )

    final_script = work_dir / "app.py"
    assert final_script.exists()
    assert "Hello from chained tool!" in final_script.read_text()
