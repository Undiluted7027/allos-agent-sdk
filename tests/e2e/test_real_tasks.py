# tests/e2e/test_real_tasks.py

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from allos.cli.main import main
from allos.providers.base import ProviderResponse, ToolCall
from allos.providers.metadata import Metadata

pytestmark = pytest.mark.e2e


# --- Fixtures ---


@pytest.fixture
def runner() -> CliRunner:
    """Provides a CliRunner instance for invoking the CLI."""
    return CliRunner()


# --- E2E Tests with Mocked LLM ---


@patch("allos.providers.openai.OpenAIProvider.chat")
def test_task_create_hello_world_script(
    mock_chat, runner: CliRunner, work_dir: Path, mock_metadata: Metadata
):
    """
    E2E Test: "Create a Python script that prints Hello World"
    - Mocks the LLM but uses the real Agent, Tools, and Filesystem.
    """
    # Define the sequence of LLM responses
    # Turn 1: LLM decides to write the file.
    response1 = ProviderResponse(
        tool_calls=[
            ToolCall(
                id="call_1",
                name="write_file",
                arguments={"path": "hello.py", "content": "print('Hello, World!')"},
            )
        ],
        metadata=mock_metadata,
    )
    # Turn 2: After writing the file, LLM gives a final answer.
    response2 = ProviderResponse(
        content="The file 'hello.py' has been created.", metadata=mock_metadata
    )
    mock_chat.side_effect = [response1, response2]

    # Invoke the CLI, auto-approving the tool use
    result = runner.invoke(
        main,
        ["--auto-approve", "Create a Python script that prints Hello World"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    # Check the final output from the agent
    assert "The file 'hello.py' has been created." in result.output

    # Verify the side effect: the file should actually exist with the correct content
    created_file = work_dir / "hello.py"
    assert created_file.exists()
    assert created_file.read_text() == "print('Hello, World!')"


@patch("allos.providers.openai.OpenAIProvider.chat")
def test_task_read_and_count_lines(
    mock_chat, runner: CliRunner, work_dir: Path, mock_metadata: Metadata
):
    """
    E2E Test: "Read main.py and count the lines"
    """
    # Setup: Create a file for the agent to read
    (work_dir / "main.py").write_text("line 1\nline 2\nline 3")

    # Define LLM responses
    # Turn 1: LLM decides to read the file.
    response1 = ProviderResponse(
        tool_calls=[
            ToolCall(id="call_1", name="read_file", arguments={"path": "main.py"})
        ],
        metadata=mock_metadata,
    )
    # Turn 2: After reading, LLM provides the final answer.
    response2 = ProviderResponse(
        content="The file 'main.py' contains 3 lines.", metadata=mock_metadata
    )
    mock_chat.side_effect = [response1, response2]

    result = runner.invoke(main, ["Read main.py and count the lines"])

    assert result.exit_code == 0
    assert "The file 'main.py' contains 3 lines." in result.output
    assert mock_chat.call_count == 2


@patch("allos.providers.openai.OpenAIProvider.chat")
def test_task_fix_syntax_error(
    mock_chat, runner: CliRunner, work_dir: Path, mock_metadata: Metadata
):
    """
    E2E Test: "Fix syntax error in test.py"
    This tests tool chaining: read -> edit
    """
    # Setup: Create a file with a syntax error
    buggy_code = "def my_function()\n    print('hello')"
    (work_dir / "test.py").write_text(buggy_code)

    # Define LLM responses
    # Turn 1: LLM decides to read the file to see the error.
    response1 = ProviderResponse(
        tool_calls=[
            ToolCall(id="call_1", name="read_file", arguments={"path": "test.py"})
        ],
        metadata=mock_metadata,
    )
    # Turn 2: After reading, LLM identifies the missing colon and decides to edit.
    response2 = ProviderResponse(
        tool_calls=[
            ToolCall(
                id="call_2",
                name="edit_file",
                arguments={
                    "path": "test.py",
                    "find_string": "def my_function()",
                    "replace_with": "def my_function():",
                },
            )
        ],
        metadata=mock_metadata,
    )
    # Turn 3: After editing, LLM confirms the fix.
    response3 = ProviderResponse(
        content="I have fixed the syntax error by adding a colon to the function definition.",
        metadata=mock_metadata,
    )
    mock_chat.side_effect = [response1, response2, response3]

    # Invoke the CLI, auto-approving the tool use
    result = runner.invoke(
        main,
        ["--auto-approve", "Fix the syntax error in test.py"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert "fixed the syntax error" in result.output

    # Verify the file was actually fixed
    fixed_code = (work_dir / "test.py").read_text()
    assert fixed_code == "def my_function():\n    print('hello')"


# --- This is the "final boss" test from test_agent_workflow.py, now as a CLI E2E test ---
@patch("allos.providers.openai.OpenAIProvider.chat")
def test_e2e_multi_turn_tool_chaining_workflow(
    mock_chat, runner: CliRunner, work_dir: Path, mock_metadata: Metadata
):
    """
    Tests a complete agent workflow via the CLI, involving multiple turns and tool chaining.
    """
    # --- Mock Provider Setup ---
    # Turn 1: LLM decides to list the directory
    response1 = ProviderResponse(
        tool_calls=[ToolCall("1", "list_directory", {})], metadata=mock_metadata
    )
    # Turn 2: After seeing the files, LLM decides to read 'instructions.txt'
    response2 = ProviderResponse(
        tool_calls=[ToolCall("2", "read_file", {"path": "instructions.txt"})],
        metadata=mock_metadata,
    )
    # Turn 3: After reading the instructions, LLM decides to write 'app.py'
    response3 = ProviderResponse(
        tool_calls=[
            ToolCall(
                "3",
                "write_file",
                {"path": "app.py", "content": "print('Hello from chained tool!')"},
            )
        ],
        metadata=mock_metadata,
    )
    # Turn 4: After writing the file, LLM decides to execute it
    response4 = ProviderResponse(
        tool_calls=[ToolCall("4", "shell_exec", {"command": "python app.py"})],
        metadata=mock_metadata,
    )
    # Turn 5: After executing, LLM gives the final answer
    response5 = ProviderResponse(
        content="I have created and executed the script. The output was 'Hello from chained tool!'.",
        metadata=mock_metadata,
    )
    mock_chat.side_effect = [response1, response2, response3, response4, response5]

    # --- Test Setup ---
    (work_dir / "instructions.txt").write_text(
        "Create a python script named app.py that prints 'Hello from chained tool!'"
    )

    # --- Run the Agent via CLI ---
    result = runner.invoke(
        main,
        ["--auto-approve", "Follow the instructions in the text file."],
        catch_exceptions=False,
    )

    # --- Assertions ---
    assert result.exit_code == 0
    assert "Hello from chained tool!" in result.output
    assert mock_chat.call_count == 5

    # Verify the final state of the filesystem
    final_script = work_dir / "app.py"
    assert final_script.exists()
    assert "Hello from chained tool!" in final_script.read_text()
