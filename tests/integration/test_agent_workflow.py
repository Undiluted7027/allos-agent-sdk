# tests/integration/test_agent_workflow.py

from pathlib import Path
from unittest.mock import patch

from allos.agent import Agent, AgentConfig
from allos.providers.base import ProviderResponse, ToolCall


@patch("rich.console.Console.input", return_value="y")  # Auto-approve all tool calls
@patch("allos.agent.agent.ProviderRegistry.get_provider")
def test_multi_turn_tool_chaining_workflow(
    mock_get_provider, mock_input, work_dir: Path
):
    """
    Tests a complete agent workflow involving multiple turns and tool chaining.
    The agent must:
    1. List files to understand its environment.
    2. Read a file to get context.
    3. Write a new file based on the context.
    4. Execute the new file.
    """
    # --- Mock Provider Setup ---
    mock_provider = mock_get_provider.return_value

    mock_provider.get_context_window.return_value = 32000

    # Define the sequence of LLM responses
    # Turn 1: LLM decides to list the directory
    response1 = ProviderResponse(tool_calls=[ToolCall("1", "list_directory", {})])

    # Turn 2: After seeing the files, LLM decides to read 'instructions.txt'
    response2 = ProviderResponse(
        tool_calls=[ToolCall("2", "read_file", {"path": "instructions.txt"})]
    )

    # Turn 3: After reading the instructions, LLM decides to write 'app.py'
    response3 = ProviderResponse(
        tool_calls=[
            ToolCall(
                "3",
                "write_file",
                {"path": "app.py", "content": "print('Hello from chained tool!')"},
            )
        ]
    )

    # Turn 4: After writing the file, LLM decides to execute it
    response4 = ProviderResponse(
        tool_calls=[ToolCall("4", "shell_exec", {"command": "python app.py"})]
    )

    # Turn 5: After executing, LLM gives the final answer
    response5 = ProviderResponse(
        content="I have created and executed the script as instructed. The output was 'Hello from chained tool!'."
    )

    mock_provider.chat.side_effect = [
        response1,
        response2,
        response3,
        response4,
        response5,
    ]

    # --- Test Setup ---
    # Create an initial file for the agent to find
    (work_dir / "instructions.txt").write_text(
        "Create a python script named app.py that prints 'Hello from chained tool!'"
    )

    config = AgentConfig(
        provider_name="test",
        model="test",
        tool_names=["list_directory", "read_file", "write_file", "shell_exec"],
    )
    agent = Agent(config)

    # --- Run the Agent ---
    final_answer = agent.run("Follow the instructions in the text file.")

    # --- Assertions ---
    assert "Hello from chained tool!" in final_answer
    assert mock_provider.chat.call_count == 5  # Verify all turns were taken

    # Verify the final state of the filesystem
    final_script = work_dir / "app.py"
    assert final_script.exists()
    assert "Hello from chained tool!" in final_script.read_text()

    # Verify the agent's final context
    assert len(agent.context) == 10  # 1 user, 4 assistant, 4 tool, 1 final assistant
