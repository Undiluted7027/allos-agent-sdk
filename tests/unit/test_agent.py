# tests/unit/test_agent.py

from unittest.mock import call, patch

import pytest

from allos.agent import Agent, AgentConfig
from allos.providers.base import ProviderResponse, ToolCall
from allos.tools.base import ToolPermission
from allos.utils.errors import AllosError, ContextWindowExceededError


@pytest.fixture
def mock_get_tool(mocker):
    """Fixture to mock ToolRegistry.get_tool."""
    return mocker.patch("allos.agent.agent.ToolRegistry.get_tool")


class TestAgent:
    def test_initialization(self, mock_get_provider, mock_get_tool):
        """Test that the agent initializes its provider and tools correctly."""
        config = AgentConfig(
            provider_name="openai",
            model="gpt-4o",
            tool_names=["read_file", "write_file"],
        )
        agent = Agent(config)

        mock_get_provider.assert_called_once_with("openai", model="gpt-4o")
        mock_get_tool.assert_has_calls(
            [call("read_file"), call("write_file")], any_order=True
        )
        assert agent.provider == mock_get_provider.return_value
        assert len(agent.tools) == 2

    def test_run_simple_chat(self, mock_get_provider):
        """Test a simple run with no tool calls."""
        mock_provider = mock_get_provider.return_value
        mock_provider.chat.return_value = ProviderResponse(content="Final Answer")

        config = AgentConfig(provider_name="test", model="test")
        agent = Agent(config)

        result = agent.run("What is Allos?")

        assert result == "Final Answer"
        mock_provider.chat.assert_called_once()
        assert len(agent.context) == 2

    @patch("allos.agent.agent.Agent._check_tool_permission", return_value=True)
    def test_run_with_one_tool_call(
        self, mock_check_permission, mock_get_provider, mock_get_tool
    ):
        """Test a run that involves a single tool call and then a final answer."""
        mock_provider = mock_get_provider.return_value
        mock_tool = mock_get_tool.return_value
        mock_tool.name = "search"
        mock_tool.execute.return_value = {"status": "success", "results": "found"}

        tool_call_response = ProviderResponse(
            tool_calls=[ToolCall("1", "search", {"query": "allos"})]
        )
        final_answer_response = ProviderResponse(content="The answer is Allos.")
        mock_provider.chat.side_effect = [tool_call_response, final_answer_response]

        config = AgentConfig(provider_name="test", model="test", tool_names=["search"])
        agent = Agent(config)

        result = agent.run("Search for allos")

        assert result == "The answer is Allos."
        assert mock_provider.chat.call_count == 2
        mock_tool.execute.assert_called_once_with(query="allos")
        # Context: user, assistant (tool_call), user (tool_result), assistant (final)
        assert len(agent.context) == 4

    def test_max_iterations_reached(self, mock_get_provider, mock_get_tool):
        """Test that the agent stops and raises an error if max_iterations is reached."""
        mock_provider = mock_get_provider.return_value
        mock_provider.chat.return_value = ProviderResponse(
            tool_calls=[ToolCall("1", "fake_tool", {})]
        )

        config = AgentConfig(provider_name="test", model="test", max_iterations=3)
        agent = Agent(config)

        with pytest.raises(AllosError) as excinfo:
            agent.run("Start loop")

        assert "reached maximum iterations" in str(excinfo.value)
        assert mock_provider.chat.call_count == 3

    @patch("rich.console.Console.input", return_value="y")
    def test_permission_check_ask_user_yes(
        self, mock_input, mock_get_provider, mock_get_tool
    ):
        """Test that ASK_USER permission with 'y' input returns True."""
        mock_tool = mock_get_tool.return_value
        mock_tool.permission = ToolPermission.ASK_USER

        config = AgentConfig(
            provider_name="test", model="test", tool_names=["ask_tool"]
        )
        agent = Agent(config)

        permission_granted = agent._check_tool_permission(mock_tool)
        assert permission_granted is True
        mock_input.assert_called_once()

    @patch("rich.console.Console.input", return_value="n")
    def test_permission_check_ask_user_no(
        self, mock_input, mock_get_provider, mock_get_tool
    ):
        """Test that ASK_USER permission with 'n' input returns False."""
        mock_tool = mock_get_tool.return_value
        mock_tool.permission = ToolPermission.ASK_USER

        config = AgentConfig(
            provider_name="test", model="test", tool_names=["ask_tool"]
        )
        agent = Agent(config)

        permission_granted = agent._check_tool_permission(mock_tool)
        assert permission_granted is False

    @patch("allos.agent.agent.Agent._check_tool_permission", return_value=False)
    def test_tool_execution_fails_on_permission_denied(
        self, mock_check_permission, mock_get_provider, mock_get_tool
    ):
        """Test that a tool call is skipped if permission is denied."""
        mock_provider = mock_get_provider.return_value
        mock_tool = mock_get_tool.return_value
        mock_tool.name = "denied_tool"

        # The LLM asks to use the tool
        llm_response = ProviderResponse(tool_calls=[ToolCall("1", "denied_tool", {})])
        # The LLM provides a final answer in the next turn
        final_answer_response = ProviderResponse(
            content="Okay, I will not use the tool."
        )
        mock_provider.chat.side_effect = [llm_response, final_answer_response]

        config = AgentConfig(
            provider_name="test", model="test", tool_names=["denied_tool"]
        )
        agent = Agent(config)
        agent.run("Use the denied tool")

        # Assert that the tool's execute method was NEVER called
        mock_tool.execute.assert_not_called()

        # Check the context to ensure an error result was added for the tool
        tool_result_message = agent.context.messages[2]
        assert tool_result_message.role == "tool"
        assert (
            tool_result_message.content
            and "Permission denied by user" in tool_result_message.content
        )

    def test_permission_check_always_allow(self, mock_get_provider, mock_get_tool):
        """Test that ALWAYS_ALLOW permission returns True without asking."""
        mock_tool = mock_get_tool.return_value
        mock_tool.permission = ToolPermission.ALWAYS_ALLOW

        config = AgentConfig(
            provider_name="test", model="test", tool_names=["allow_tool"]
        )
        agent = Agent(config)

        # We don't need to mock console input, as it should not be called
        permission_granted = agent._check_tool_permission(mock_tool)
        assert permission_granted is True

    def test_permission_check_always_deny(self, mock_get_provider, mock_get_tool):
        """Test that ALWAYS_DENY permission returns False without asking."""
        mock_tool = mock_get_tool.return_value
        mock_tool.permission = ToolPermission.ALWAYS_DENY

        config = AgentConfig(
            provider_name="test", model="test", tool_names=["deny_tool"]
        )
        agent = Agent(config)

        permission_granted = agent._check_tool_permission(mock_tool)
        assert permission_granted is False

    @patch("rich.console.Console.input", side_effect=KeyboardInterrupt)
    def test_permission_check_ask_user_interrupted(
        self, mock_input, mock_get_provider, mock_get_tool
    ):
        """Test that ASK_USER permission is denied if the user interrupts the prompt."""
        mock_tool = mock_get_tool.return_value
        mock_tool.permission = ToolPermission.ASK_USER

        config = AgentConfig(
            provider_name="test", model="test", tool_names=["ask_tool"]
        )
        agent = Agent(config)

        permission_granted = agent._check_tool_permission(mock_tool)
        assert permission_granted is False
        mock_input.assert_called_once()

    def test_permission_check_unknown_permission_returns_false(
        self, mock_get_provider, mock_get_tool
    ):
        """
        Test that an unknown or invalid tool permission defaults to False.
        """
        mock_tool = mock_get_tool.return_value
        # Assign an invalid value that is not one of the ToolPermission enums
        mock_tool.permission = 99  # An integer or any other invalid value

        config = AgentConfig(
            provider_name="test", model="test", tool_names=["unknown_perm_tool"]
        )
        agent = Agent(config)

        # The method should not ask for input and should default to denying permission
        permission_granted = agent._check_tool_permission(mock_tool)
        assert permission_granted is False

    def test_permission_check_auto_approve(self, mock_get_provider, mock_get_tool):
        """Test that auto_approve=True bypasses all other permission checks."""
        mock_tool = mock_get_tool.return_value
        # Give the tool a restrictive permission to prove auto-approve overrides it
        mock_tool.permission = ToolPermission.ALWAYS_DENY

        config = AgentConfig(
            provider_name="test",
            model="test",
            tool_names=["any_tool"],
            auto_approve=True,  # The key setting for this test
        )
        agent = Agent(config)

        # Even though the tool is ALWAYS_DENY, auto_approve should grant permission
        permission_granted = agent._check_tool_permission(mock_tool)
        assert permission_granted is True

    def test_context_window_exceeded_raises_error(self, mock_get_provider):
        """Test that an error is raised if the context is too large."""
        mock_provider = mock_get_provider.return_value
        # Configure the mock provider to report a small context window
        mock_provider.get_context_window.return_value = 100

        config = AgentConfig(provider_name="test", model="test")
        agent = Agent(config)

        # Add a very long message to the context to exceed the limit
        long_content = "word " * 200  # This will be > 100 tokens
        agent.context.add_user_message(long_content)

        with pytest.raises(ContextWindowExceededError) as excinfo:
            agent.run("Another prompt")  # The run method will call _get_llm_response

        assert "Conversation context has grown too large" in str(excinfo.value)
        # Ensure the provider's chat method was never called
        mock_provider.chat.assert_not_called()

    def test_run_passes_max_tokens_to_provider(self, mock_get_provider):
        """Test that max_tokens is passed to provider.chat."""
        mock_provider = mock_get_provider.return_value
        mock_provider.chat.return_value = ProviderResponse(content="Answer")

        config = AgentConfig(provider_name="test", model="test", max_tokens=50)
        agent = Agent(config)

        agent.run("prompt")

        # Verify kwargs
        call_kwargs = mock_provider.chat.call_args.kwargs
        assert call_kwargs["max_tokens"] == 50

    @patch("allos.agent.agent.ToolRegistry.get_tool")
    def test_no_tools_config_prevents_tool_loading(
        self, mock_get_tool, mock_get_provider
    ):
        """Test that no_tools=True results in empty tool list even if names provided."""
        config = AgentConfig(
            provider_name="test", model="test", tool_names=["read_file"], no_tools=True
        )
        agent = Agent(config)

        assert len(agent.tools) == 0
        mock_get_tool.assert_not_called()

    def test_initialization_passes_base_url_and_api_key(
        self, mock_get_provider, mock_get_tool
    ):
        """Test that Agent passes base_url and api_key from config to provider."""
        config = AgentConfig(
            provider_name="chat_completions",
            model="test-model",
            base_url="http://custom.url",
            api_key="secret-key",
        )
        Agent(config)

        mock_get_provider.assert_called_once()
        call_kwargs = mock_get_provider.call_args.kwargs

        assert call_kwargs["base_url"] == "http://custom.url"
        assert call_kwargs["api_key"] == "secret-key"
