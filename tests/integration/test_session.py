# tests/integration/test_session.py

from pathlib import Path
from unittest.mock import patch

import pytest

from allos.agent import Agent, AgentConfig
from allos.providers.base import Message, ProviderResponse, ToolCall
from allos.utils.token_counter import count_tokens


# We can keep the real provider/tool registries, but mock the provider's .chat method
@patch("allos.providers.openai.OpenAIProvider.chat")
@patch("allos.providers.anthropic.AnthropicProvider.chat")
@patch("allos.agent.agent.Agent._check_tool_permission", return_value=True)
@pytest.mark.parametrize("provider_name", ["openai", "anthropic"])
def test_session_save_and_load_with_filesystem(
    mock_check_permission,
    mock_anthropic_chat,
    mock_openai_chat,
    provider_name,
    work_dir: Path,
    mock_metadata_factory,
):
    """
    Tests the full end-to-end workflow of saving and loading a session to/from the filesystem.
    """
    mock_provider_chat = (
        mock_openai_chat if provider_name == "openai" else mock_anthropic_chat
    )

    # --- 1. SETUP and INITIAL RUN ---
    # Define the sequence of LLM intents for each turn
    turn_intents = [
        {
            "tool_calls": [
                ToolCall("1", "write_file", {"path": "test.txt", "content": "hello"})
            ]
        },
        {"content": "File created."},
    ]

    # Create a dynamic side_effect function for the mock chat
    def dynamic_chat_side_effect(messages: list[Message], **kwargs) -> ProviderResponse:
        call_index = mock_provider_chat.call_count - 1
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

    mock_provider_chat.side_effect = dynamic_chat_side_effect

    config = AgentConfig(
        provider_name=provider_name, model="test-model", tool_names=["write_file"]
    )
    agent = Agent(config)

    # Run the agent part-way
    agent.run("Create a test file.")

    # Context should be: user_prompt, assistant_tool_call, user_tool_result, final_assistant
    assert len(agent.context) == 4

    # Verify that the dynamic metadata was aggregated correctly
    assert agent.last_run_metadata is not None
    assert agent.last_run_metadata.usage.total_tokens > 0
    assert len(agent.last_run_metadata.turns.turn_history) == 2

    # --- 2. SAVE SESSION ---
    session_filepath = work_dir / "test_session.json"
    agent.save_session(session_filepath)
    assert session_filepath.exists()

    # --- 3. LOAD SESSION ---
    # We don't need to mock anything here, as load_session is a classmethod
    loaded_agent = Agent.load_session(session_filepath)

    # --- 4. VERIFY LOADED STATE ---
    assert isinstance(loaded_agent, Agent)
    assert loaded_agent.config.provider_name == provider_name
    assert len(loaded_agent.context) == 4
    # Verify the content of the loaded context
    assert loaded_agent.context.messages[0].content == "Create a test file."
    assert loaded_agent.context.messages[2].role == "tool"
