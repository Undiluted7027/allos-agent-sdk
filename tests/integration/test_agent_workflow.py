# tests/integration/test_agent_workflow.py

from pathlib import Path
from unittest.mock import patch

from allos.agent import Agent, AgentConfig
from allos.providers.base import Message, ProviderResponse, ToolCall
from allos.utils.token_counter import count_tokens


@patch("rich.console.Console.input", return_value="y")  # Auto-approve all tool calls
@patch("allos.agent.agent.ProviderRegistry.get_provider")
def test_multi_turn_tool_chaining_workflow(
    mock_get_provider, mock_input, work_dir: Path, mock_metadata_factory
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

    # Define the sequence of LLM intents for each turn
    turn_intents = [
        # LLM decides to list the directory
        {"tool_calls": [ToolCall("1", "list_directory", {})]},
        # After seeing the files, LLM decides to read 'instructions.txt'
        {"tool_calls": [ToolCall("2", "read_file", {"path": "instructions.txt"})]},
        # After reading the instructions, LLM decides to write 'app.py'
        {
            "tool_calls": [
                ToolCall(
                    "3",
                    "write_file",
                    {"path": "app.py", "content": "print('Hello from chained tool!')"},
                )
            ]
        },
        # After writing the file, LLM decides to execute it
        {"tool_calls": [ToolCall("4", "shell_exec", {"command": "python app.py"})]},
        # After executing, LLM gives the final answer
        {
            "content": "I have created and executed the script as instructed. The output was 'Hello from chained tool!'."
        },
    ]

    # Create a dynamic side_effect function for the mock chat
    def dynamic_chat_side_effect(messages: list[Message], **kwargs) -> ProviderResponse:
        call_index = mock_provider.chat.call_count - 1
        intent = turn_intents[call_index]

        # Calculate input tokens based on the current context
        input_text = " ".join([msg.content or "" for msg in messages])
        input_tokens = count_tokens(input_text)

        # Use the metadata factory to create a response with correct token counts
        dynamic_metadata = mock_metadata_factory(usage={"input_tokens": input_tokens})

        return ProviderResponse(
            content=intent.get("content"),
            tool_calls=intent.get("tool_calls", []),
            metadata=dynamic_metadata,
        )

    mock_provider.chat.side_effect = dynamic_chat_side_effect

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

    # Verify that input token counts are increasing each turn
    assert agent.last_run_metadata is not None
    turn_history = agent.last_run_metadata.turns.turn_history
    assert len(turn_history) == 5

    tokens_turn1 = turn_history[0].tokens_used.input_tokens
    tokens_turn2 = turn_history[1].tokens_used.input_tokens
    tokens_turn3 = turn_history[2].tokens_used.input_tokens

    assert tokens_turn1 > 0
    # The context grows after the first tool result is added
    assert tokens_turn2 > tokens_turn1
    # The context grows again after the second tool result
    assert tokens_turn3 > tokens_turn2

    # Verify the final state of the filesystem
    final_script = work_dir / "app.py"
    assert final_script.exists()
    assert "Hello from chained tool!" in final_script.read_text()

    # Verify the agent's final context
    assert len(agent.context) == 10  # 1 user, 4 assistant, 4 tool, 1 final assistant
