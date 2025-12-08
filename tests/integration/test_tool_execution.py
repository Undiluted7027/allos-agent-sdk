# tests/integration/test_tool_execution.py

from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

from allos.agent import Agent, AgentConfig
from allos.providers.base import Message, ProviderResponse, ToolCall
from allos.tools import ToolRegistry
from allos.utils.errors import ToolError
from allos.utils.token_counter import count_tokens

# --- File Tool Integration Tests ---


def test_file_tools_integration_workflow(work_dir: Path):
    """Tests a sequence of file tool executions on the real filesystem."""
    # We don't need a real agent, just the tools

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
    mock_input, work_dir: Path, mock_metadata_factory
):
    """Tests running shell_exec when the user grants permission."""
    config = AgentConfig(
        provider_name="openai", model="test", tool_names=["shell_exec"]
    )
    agent = Agent(config)

    with patch.object(agent.provider, "chat") as mock_chat:
        turn_intents: List[Dict[str, Any]] = [
            {
                "tool_calls": [
                    ToolCall("1", "shell_exec", {"command": "echo 'Success'"})
                ]
            },
            {"content": "The command was successful."},
        ]

        def dynamic_chat_side_effect(
            messages: list[Message], **kwargs
        ) -> ProviderResponse:
            call_index = mock_chat.call_count - 1
            intent: Dict[str, Any] = turn_intents[call_index]
            input_tokens = count_tokens(" ".join([m.content or "" for m in messages]))
            metadata = mock_metadata_factory(usage={"input_tokens": input_tokens})
            return ProviderResponse(
                content=intent.get("content"),
                tool_calls=intent.get("tool_calls", []),
                metadata=metadata,
            )

        mock_chat.side_effect = dynamic_chat_side_effect
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
    mock_input, work_dir: Path, mock_metadata_factory
):
    """Tests that the agent recovers when the user denies permission."""
    config = AgentConfig(
        provider_name="openai", model="test", tool_names=["shell_exec"]
    )
    agent = Agent(config)

    with patch.object(agent.provider, "chat") as mock_chat:
        turn_intents: List[Dict[str, Any]] = [
            {
                "tool_calls": [
                    ToolCall("1", "shell_exec", {"command": "echo 'Failure'"})
                ]
            },
            {"content": "I cannot proceed without permission."},
        ]

        def dynamic_chat_side_effect(
            messages: list[Message], **kwargs
        ) -> ProviderResponse:
            call_index = mock_chat.call_count - 1
            intent: Dict[str, Any] = turn_intents[call_index]
            input_tokens = count_tokens(" ".join([m.content or "" for m in messages]))
            metadata = mock_metadata_factory(usage={"input_tokens": input_tokens})
            return ProviderResponse(
                content=intent.get("content"),
                tool_calls=intent.get("tool_calls", []),
                metadata=metadata,
            )

        mock_chat.side_effect = dynamic_chat_side_effect
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


def test_agent_handles_tool_validation_error(
    work_dir: Path, mock_metadata_factory, mocker
):
    """
    Tests that the agent correctly catches a ToolError during validation,
    skips execution, and records the error in its context.
    """
    # 1. Setup the agent and mock its provider
    config = AgentConfig(
        provider_name="openai",
        model="test",
        tool_names=["write_file"],
        auto_approve=True,
    )
    agent = Agent(config)

    # 2. Configure the mock LLM's responses
    with patch.object(agent.provider, "chat") as mock_chat:
        turn_intents: List[Dict[str, Any]] = [
            {
                "tool_calls": [
                    ToolCall(
                        "call123", "write_file", {"path": "test.txt", "content": 12345}
                    )
                ]
            },
            {
                "content": "I apologize, I provided an invalid type for the 'content' argument."
            },
        ]

        def dynamic_chat_side_effect(
            messages: list[Message], **kwargs
        ) -> ProviderResponse:
            call_index = mock_chat.call_count - 1
            intent: Dict[str, Any] = turn_intents[call_index]
            input_tokens = count_tokens(" ".join([m.content or "" for m in messages]))
            metadata = mock_metadata_factory(usage={"input_tokens": input_tokens})
            return ProviderResponse(
                content=intent.get("content"),
                tool_calls=intent.get("tool_calls", []),
                metadata=metadata,
            )

        mock_chat.side_effect = dynamic_chat_side_effect

        # 3. Patch the real tool's validation method to raise the expected error
        # This simulates the behavior of our proposed enhancement to BaseTool.
        write_tool_instance = next(
            (t for t in agent.tools if t.name == "write_file"), None
        )
        assert write_tool_instance is not None

        validation_error = ToolError(
            "Invalid type for argument 'content'. Expected string, but got int."
        )
        mocker.patch.object(
            write_tool_instance, "validate_arguments", side_effect=validation_error
        )
        # Also spy on the `execute` method to ensure it's not called
        execute_spy = mocker.spy(write_tool_instance, "execute")

        # 4. Run the agent
        final_answer = agent.run("Write a file with invalid content.")

        # 5. Assert the outcomes
        # The agent should have recovered and given the LLM's final answer
        assert "I apologize" in final_answer
        assert mock_chat.call_count == 2

        # Crucially, the tool's execute method should NEVER have been called
        execute_spy.assert_not_called()

        # The file should not exist on the filesystem
        assert not (work_dir / "test.txt").exists()

        # The context should contain the recorded validation error
        assert len(agent.context) == 4
        tool_result_message = agent.context.messages[2]
        assert tool_result_message.role == "tool"
        assert (
            tool_result_message.content
            and '"status": "error"' in tool_result_message.content
        )
        assert "Invalid type for argument 'content'" in tool_result_message.content
