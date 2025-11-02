# tests/integration/test_session.py

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from allos.agent import Agent, AgentConfig
from allos.providers.base import ProviderResponse, ToolCall
from allos.tools.base import ToolPermission


@pytest.mark.integration
@pytest.mark.parametrize("provider_name", ["openai", "anthropic"])
@patch("allos.agent.agent.Agent._check_tool_permission", return_value=True)
def test_session_save_and_load_end_to_end(
    mock_check_permission, provider_name, work_dir: Path
):
    """
    Tests the full end-to-end workflow:
    1. Run an agent part-way through a task.
    2. Save the agent's session.
    3. Load the session into a new agent instance.
    4. Verify the new agent's state is correct.
    5. Continue the task with the loaded agent and verify it uses the loaded context.
    """
    # --- 1. SETUP and INITIAL RUN ---

    # Mock the provider and tool registries for this test
    with (
        patch("allos.agent.agent.ProviderRegistry") as MockProviderRegistry,
        patch("allos.agent.agent.ToolRegistry") as MockToolRegistry,
    ):
        # Setup mocks
        mock_provider = MockProviderRegistry.get_provider.return_value
        mock_tool = MockToolRegistry.get_tool.return_value
        mock_tool.name = "search"
        mock_tool.permission = ToolPermission.ALWAYS_ALLOW  # Simplify test
        mock_tool.execute.return_value = {"status": "success", "results": "found"}

        # Define the sequence of LLM responses for the first run
        tool_call_response = ProviderResponse(
            tool_calls=[ToolCall("1", "search", {"query": "allos"})]
        )
        mock_provider.chat.side_effect = [
            tool_call_response
        ]  # Stop after one tool call

        config = AgentConfig(
            provider_name=provider_name, model="test-model", tool_names=["search"]
        )
        agent = Agent(config)

        # Run the agent until it hits the first stopping point (after the tool call)
        # We expect it to raise an error because we only provided one mock response
        with pytest.raises(Exception):  # noqa: B017
            agent.run("Search for allos")

        # At this point, context should be: user_prompt, assistant_tool_call, user_tool_result
        assert len(agent.context) == 3

        # --- 2. SAVE SESSION ---
        session_filepath = work_dir / "test_session.json"
        agent.save_session(session_filepath)

        assert session_filepath.exists()

        # Verify file content
        with open(session_filepath, "r") as f:
            data = json.load(f)
        assert data["config"]["provider_name"] == provider_name
        assert len(data["context"]["messages"]) == 3

    # --- 3. LOAD SESSION ---
    # Use a new set of mocks for the loaded agent
    with (
        patch("allos.agent.agent.ProviderRegistry") as MockProviderRegistry2,
        patch("allos.agent.agent.ToolRegistry"),
    ):
        # Load the agent from the file
        loaded_agent = Agent.load_session(session_filepath)

        # --- 4. VERIFY LOADED STATE ---
        assert isinstance(loaded_agent, Agent)
        assert loaded_agent.config.provider_name == provider_name
        assert len(loaded_agent.context) == 3
        assert loaded_agent.context.messages[0].role == "user"
        assert loaded_agent.context.messages[2].role == "tool"

        # --- 5. CONTINUE CONVERSATION ---
        # Setup a new mock provider for the loaded agent
        mock_provider2 = MockProviderRegistry2.get_provider.return_value
        final_answer_response = ProviderResponse(content="The answer is Allos.")
        mock_provider2.chat.return_value = final_answer_response

        # Re-assign the provider on the loaded agent to our new mock
        loaded_agent.provider = mock_provider2

        # Run with a new prompt to continue the conversation
        result = loaded_agent.run("Now summarize the results")

        # Assert that the final answer is correct
        assert result == "The answer is Allos."

        # CRITICAL ASSERTION: Check that the provider was called with the FULL loaded context
        mock_provider2.chat.assert_called_once()
        call_kwargs = mock_provider2.chat.call_args.kwargs
        sent_messages = call_kwargs.get("messages")

        assert sent_messages is not None

        # Context should be: [original_user, original_assistant, original_tool_result, new_user]
        assert len(sent_messages) == 4
        assert sent_messages[0].content == "Search for allos"
        assert sent_messages[3].content == "Now summarize the results"
