# tests/integration/test_tool_execution.py

from pathlib import Path
from unittest.mock import patch

from allos.agent import Agent, AgentConfig
from allos.providers.base import ProviderResponse, ToolCall
from allos.providers.metadata import Metadata

# --- File Tool Integration Tests ---


def test_file_tools_integration_workflow(work_dir: Path):
    """Tests a sequence of file tool executions on the real filesystem."""
    # We don't need a real agent, just the tools
    from allos.tools import ToolRegistry

    write_tool = ToolRegistry.get_tool("write_file")
    read_tool = ToolRegistry.get_tool("read_file")
    edit_tool = ToolRegistry.get_tool("edit_file")
    list_tool = ToolRegistry.get_tool("list_directory")

    # 1. Write a file
    write_result = write_tool.execute(path="test.txt", content="Hello\nWorld")
    assert write_result["status"] == "success"
    assert (work_dir / "test.txt").read_text() == "Hello\nWorld"

    # 2. Read the file
    read_result = read_tool.execute(path="test.txt")
    assert read_result["status"] == "success"
    assert read_result["content"] == "Hello\nWorld"

    # 3. Edit the file
    edit_result = edit_tool.execute(
        path="test.txt", find_string="World", replace_with="Allos"
    )
    assert edit_result["status"] == "success"
    assert (work_dir / "test.txt").read_text() == "Hello\nAllos"

    # 4. List the directory to confirm file presence
    list_result = list_tool.execute()
    assert list_result["status"] == "success"
    assert "test.txt" in list_result["contents"]


# --- Shell Tool and Permission System Integration Test ---


@patch("rich.console.Console.input", return_value="y")
def test_shell_exec_with_permission_granted(
    mock_input, work_dir: Path, mock_metadata: Metadata
):
    """Tests running shell_exec when the user grants permission."""
    config = AgentConfig(
        provider_name="openai", model="test", tool_names=["shell_exec"]
    )
    agent = Agent(config)

    with patch.object(agent.provider, "chat") as mock_chat:
        # --- FIX: Use side_effect to provide a sequence of responses ---
        # Turn 1: LLM requests the tool.
        response1 = ProviderResponse(
            tool_calls=[ToolCall("1", "shell_exec", {"command": "echo 'Success'"})],
            metadata=mock_metadata,
        )
        # Turn 2: After the tool succeeds, LLM gives a final answer.
        response2 = ProviderResponse(
            content="The command was successful.", metadata=mock_metadata
        )
        mock_chat.side_effect = [response1, response2]

        agent.run("test")

    # Assert that the prompt was shown only once.
    mock_input.assert_called_once_with(
        "[bold yellow]❓ Allow tool 'shell_exec' to run? (y/n): [/]"
    )

    # Check the final context to see the successful tool result.
    # Context should be [user, assistant_tool_call, user_tool_result, final_assistant]
    assert len(agent.context) == 4
    tool_result_msg = agent.context.messages[2]  # The tool result is the 3rd message
    assert tool_result_msg.content is not None
    assert tool_result_msg.role == "tool"
    assert '"status": "success"' in tool_result_msg.content
    assert '"stdout": "Success\\n"' in tool_result_msg.content


@patch("rich.console.Console.input", return_value="n")  # Deny permission
def test_shell_exec_with_permission_denied(
    mock_input, work_dir: Path, mock_metadata: Metadata
):
    """Tests that the agent recovers when the user denies permission."""
    config = AgentConfig(
        provider_name="openai", model="test", tool_names=["shell_exec"]
    )
    agent = Agent(config)

    with patch.object(agent.provider, "chat") as mock_chat:
        # --- FIX: Use side_effect for a smarter mock ---
        # Turn 1: LLM requests the tool.
        response1 = ProviderResponse(
            tool_calls=[ToolCall("1", "shell_exec", {"command": "echo 'Failure'"})],
            metadata=mock_metadata,
        )
        # Turn 2: After permission is denied, LLM sees the error and gives a text response.
        response2 = ProviderResponse(
            content="I cannot proceed without permission.", metadata=mock_metadata
        )
        mock_chat.side_effect = [response1, response2]

        agent.run("test")

    # Assert that the permission prompt was shown only once.
    mock_input.assert_called_once()

    # Check the context to ensure the error was recorded and the agent recovered.
    # Context: [user, assistant_tool_call, user_tool_error, final_assistant]
    assert len(agent.context) == 4
    tool_result_msg = agent.context.messages[2]
    assert tool_result_msg.content is not None
    assert '"status": "error"' in tool_result_msg.content
    assert "Permission denied by user" in tool_result_msg.content
