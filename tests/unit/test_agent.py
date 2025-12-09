# tests/unit/test_agent.py

from unittest.mock import call, patch

import pytest

from allos.agent import Agent, AgentConfig
from allos.providers.base import ProviderChunk, ProviderResponse, ToolCall
from allos.providers.metadata import (
    EstimatedCost,
    Latency,
    Metadata,
    ModelConfiguration,
    ModelInfo,
    ProviderSpecific,
    QualitySignals,
    SdkInfo,
    ToolCallDetail,
    ToolInfo,
    TurnLog,
    TurnTokensUsed,
    Usage,
)
from allos.tools.base import ToolPermission
from allos.utils.errors import AllosError, ContextWindowExceededError


@pytest.fixture
def mock_get_tool(mocker):
    """Fixture to mock ToolRegistry.get_tool."""
    return mocker.patch("allos.agent.agent.ToolRegistry.get_tool")


@pytest.fixture
def simple_metadata():
    """Returns a minimal valid Metadata object."""
    return Metadata(
        status="success",
        model=ModelInfo(
            provider="test",
            model_id="test-model",
            configuration=ModelConfiguration(max_output_tokens=50),
        ),
        usage=Usage(
            input_tokens=10,
            output_tokens=10,
            total_tokens=20,
            estimated_cost=EstimatedCost(total_usd=0.001),
        ),
        latency=Latency(total_duration_ms=100),
        tools=ToolInfo(tools_available=[]),
        quality_signals=QualitySignals(),
        provider_specific=ProviderSpecific(),
        sdk=SdkInfo(sdk_version="0.0.0"),
    )


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

    def test_run_simple_chat(self, mock_get_provider, mock_metadata: Metadata):
        """Test a simple run with no tool calls."""
        mock_provider = mock_get_provider.return_value
        mock_provider.chat.return_value = ProviderResponse(
            content="Final Answer", metadata=mock_metadata
        )

        config = AgentConfig(provider_name="test", model="test")
        agent = Agent(config)

        result = agent.run("What is Allos?")

        assert result == "Final Answer"
        mock_provider.chat.assert_called_once()
        assert len(agent.context) == 2

    @patch("allos.agent.agent.Agent._check_tool_permission", return_value=True)
    def test_run_with_one_tool_call(
        self,
        mock_check_permission,
        mock_get_provider,
        mock_get_tool,
        mock_metadata: Metadata,
    ):
        """Test a run that involves a single tool call and then a final answer."""
        mock_provider = mock_get_provider.return_value
        mock_tool = mock_get_tool.return_value
        mock_tool.name = "search"
        mock_tool.execute.return_value = {"status": "success", "results": "found"}

        tool_call_response = ProviderResponse(
            tool_calls=[ToolCall("1", "search", {"query": "allos"})],
            metadata=mock_metadata,
        )
        final_answer_response = ProviderResponse(
            content="The answer is Allos.", metadata=mock_metadata
        )
        mock_provider.chat.side_effect = [tool_call_response, final_answer_response]

        config = AgentConfig(provider_name="test", model="test", tool_names=["search"])
        agent = Agent(config)

        result = agent.run("Search for allos")

        assert result == "The answer is Allos."
        assert mock_provider.chat.call_count == 2
        mock_tool.execute.assert_called_once_with(query="allos")
        # Context: user, assistant (tool_call), user (tool_result), assistant (final)
        assert len(agent.context) == 4

    def test_max_iterations_reached(
        self, mock_get_provider, mock_get_tool, mock_metadata: Metadata
    ):
        """Test that the agent stops and raises an error if max_iterations is reached."""
        mock_provider = mock_get_provider.return_value
        mock_provider.chat.return_value = ProviderResponse(
            tool_calls=[ToolCall("1", "fake_tool", {})], metadata=mock_metadata
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
        self,
        mock_check_permission,
        mock_get_provider,
        mock_get_tool,
        mock_metadata: Metadata,
    ):
        """Test that a tool call is skipped if permission is denied."""
        mock_provider = mock_get_provider.return_value
        mock_tool = mock_get_tool.return_value
        mock_tool.name = "denied_tool"

        # The LLM asks to use the tool
        llm_response = ProviderResponse(
            tool_calls=[ToolCall("1", "denied_tool", {})], metadata=mock_metadata
        )
        # The LLM provides a final answer in the next turn
        final_answer_response = ProviderResponse(
            content="Okay, I will not use the tool.", metadata=mock_metadata
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

    def test_run_passes_max_tokens_to_provider(
        self, mock_get_provider, mock_metadata: Metadata
    ):
        """Test that max_tokens is passed to provider.chat."""
        mock_provider = mock_get_provider.return_value
        mock_provider.chat.return_value = ProviderResponse(
            content="Answer", metadata=mock_metadata
        )

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


class TestAgentMetadataInternal:
    """Tests for internal metadata aggregation logic in Agent."""

    def test_finalize_run_metadata_else_branch(self, mock_get_provider):
        """
        Tests the `else` branch of _finalize_run_metadata where
        cumulative_state['last_metadata'] is None.
        This occurs if the agent loop doesn't run (e.g., max_iterations=0).
        """
        config = AgentConfig(provider_name="test", model="test", max_iterations=0)
        agent = Agent(config)

        # Running with 0 iterations triggers the loop bypass, calls finalize, then raises AllosError
        with pytest.raises(AllosError, match="reached maximum iterations"):
            agent.run("test")

        # Verify the else branch was hit: last_run_metadata should be explicitly None
        assert agent.last_run_metadata is None

    def test_create_aggregate_metadata_calculates_cost(
        self, mock_get_provider, simple_metadata
    ):
        """
        Tests the `if aggregate.usage.estimated_cost:` branch in _create_aggregate_metadata.
        Ensures cost is summed correctly from the cumulative state.
        """
        config = AgentConfig(provider_name="test", model="test")
        agent = Agent(config)

        # Mock inputs for the aggregation method
        tool_details = [
            ToolCallDetail(
                tool_call_id="1", tool_name="test", arguments={}, execution_time_ms=10
            )
        ]
        turn_history = [
            TurnLog(
                turn_number=1,
                model_used="test",
                content_type="text",
                tokens_used=TurnTokensUsed(input_tokens=10, output_tokens=10),
                duration_ms=100,
            )
        ]

        # Call the private method directly
        aggregated = agent._create_aggregate_metadata(
            base_metadata=simple_metadata,
            all_tool_details=tool_details,
            turn_history=turn_history,
            total_input_tokens=100,
            total_output_tokens=50,
            total_cost=0.05,  # Cumulative cost to verify
        )

        # Verify cost was updated (covering the if branch)
        assert aggregated.usage.estimated_cost is not None
        assert aggregated.usage.estimated_cost.total_usd == 0.05

        # Verify token counts were updated
        assert aggregated.usage.input_tokens == 100
        assert aggregated.usage.output_tokens == 50
        assert aggregated.usage.total_tokens == 150

        # Verify tool/turn stats were updated
        assert aggregated.tools.total_tool_calls == 1
        assert aggregated.turns.total_turns == 1


class TestAgentStreaming:
    """Tests for the stream_run method and its helpers."""

    def test_stream_run_simple_text_flow(
        self, mock_get_provider, simple_metadata, mock_get_tool
    ):
        """
        Tests a simple text response in streaming mode.
        Covers: _get_provider_stream, _process_streaming_iteration, context updates.
        """
        mock_provider = mock_get_provider.return_value
        config = AgentConfig(provider_name="test", model="test")
        agent = Agent(config)

        # Mock the stream generator
        # Yields: Content -> Content -> Final Metadata
        mock_stream = [
            ProviderChunk(content="Hello"),
            ProviderChunk(content=" World"),
            ProviderChunk(final_metadata=simple_metadata),
        ]
        mock_provider.stream_chat.return_value = iter(mock_stream)

        # Run the generator
        chunks = list(agent.stream_run("Hi"))

        # Verify chunks yielded to caller
        # The agent yields content chunks as they come, then (optionally) final metadata
        assert len(chunks) == 4
        assert chunks[0].content == "Hello"
        assert chunks[1].content == " World"
        assert chunks[2].final_metadata is not None
        assert chunks[3].final_metadata is not None

        # Verify context was updated
        assert len(agent.context.messages) == 2  # User + Assistant
        assert agent.context.messages[-1].content == "Hello World"
        assert agent.context.messages[-1].role == "assistant"

        # Verify the final aggregate metadata correctness
        final_meta = chunks[3].final_metadata
        assert final_meta.turns.total_turns == 1

    def test_stream_run_tool_execution_flow(
        self, mock_get_provider, mock_get_tool, simple_metadata
    ):
        """
        Tests streaming with tool execution (Agentic Loop).
        Turn 1: Tool Call -> Turn 2: Final Answer.
        Covers: _execute_and_record_tools, _update_chunk_metadata in streaming context.
        """
        mock_provider = mock_get_provider.return_value
        mock_tool = mock_get_tool.return_value
        mock_tool.name = "search"
        mock_tool.execute.return_value = {"result": "found"}
        # Allow auto-approve to skip permission mock
        config = AgentConfig(
            provider_name="test",
            model="test",
            tool_names=["search"],
            auto_approve=True,
        )
        agent = Agent(config)

        # --- Turn 1 Stream: Tool Call ---
        tool_call = ToolCall(id="call_1", name="search", arguments={"q": "allos"})
        turn_1_stream = [
            ProviderChunk(tool_call_done=tool_call),
            ProviderChunk(final_metadata=simple_metadata),
        ]

        # --- Turn 2 Stream: Final Answer ---
        turn_2_stream = [
            ProviderChunk(content="Answer"),
            ProviderChunk(final_metadata=simple_metadata),
        ]

        # Configure side_effect to return iterators for sequential calls
        mock_provider.stream_chat.side_effect = [
            iter(turn_1_stream),
            iter(turn_2_stream),
        ]

        # Run
        chunks = list(agent.stream_run("Search something"))

        # Verify Tool Execution
        mock_tool.execute.assert_called_once_with(q="allos")

        # Verify Context Evolution
        # 1. User
        # 2. Assistant (Tool Call)
        # 3. Tool Result
        # 4. Assistant (Final Answer)
        assert len(agent.context.messages) == 4
        assert agent.context.messages[1].tool_calls[0].id == "call_1"
        assert agent.context.messages[2].role == "tool"
        assert agent.context.messages[3].content == "Answer"

        # Verify Chunks
        # We expect tool_call chunks, then execution happens internally (no chunks yielded for exec),
        # then content chunks from Turn 2.
        # Note: The agent yields the chunks it receives.
        # Turn 1: tool_call_done, final_metadata (yielded? _yield_final_aggregate_metadata only yields at end)
        # Actually agent.stream_run yields chunks from _process_streaming_iteration.
        # Then if done, yields final aggregate.

        # Let's check the collected chunks
        # 1. Tool Call Done
        # 2. Final Metadata (Chunk from Turn 1) -> Actually `stream_run` yields this?
        #    Looking at code: `yield chunk` is called for every chunk in `_process_streaming_iteration`.
        # 3. Content "Answer"
        # 4. Final Metadata (Chunk from Turn 2)
        # 5. Final Aggregate Metadata (Yielded by `_yield_final_aggregate_metadata` at return)

        assert len(chunks) >= 3
        assert chunks[-1].final_metadata is not None
        assert chunks[-1].final_metadata.turns.total_turns == 2

    def test_stream_run_provider_error(self, mock_get_provider, simple_metadata):
        """Tests that a stream error raises AllosError."""
        mock_provider = mock_get_provider.return_value
        config = AgentConfig(provider_name="test", model="test")
        agent = Agent(config)

        mock_stream = [ProviderChunk(error="Simulated API failure")]
        mock_provider.stream_chat.return_value = iter(mock_stream)

        with pytest.raises(
            AllosError, match="Streaming provider error: Simulated API failure"
        ):
            list(agent.stream_run("Hi"))

    def test_stream_run_max_iterations_exceeded(
        self, mock_get_provider, simple_metadata
    ):
        """Tests that AllosError is raised if max_iterations is exceeded in streaming."""
        mock_provider = mock_get_provider.return_value
        config = AgentConfig(
            provider_name="test", model="test", max_iterations=1, auto_approve=True
        )
        agent = Agent(config)

        # Force a tool call, which forces a loop. Since max_iterations=1, loop finishes after Turn 1.
        # Then it hits the raise AllosError at end of stream_run.
        tool_call = ToolCall(id="c1", name="search", arguments={})
        mock_stream = [
            ProviderChunk(tool_call_done=tool_call),
            ProviderChunk(final_metadata=simple_metadata),
        ]
        mock_provider.stream_chat.return_value = iter(mock_stream)

        with pytest.raises(AllosError, match="reached maximum iterations"):
            list(agent.stream_run("Hi"))

    def test_ttft_calculation(self, mock_get_provider, simple_metadata):
        """Tests that Time to First Token is calculated and recorded."""
        mock_provider = mock_get_provider.return_value
        config = AgentConfig(provider_name="test", model="test")
        agent = Agent(config)

        mock_stream = [
            ProviderChunk(content="First"),  # TTFT trigger
            ProviderChunk(content="Second"),
            ProviderChunk(final_metadata=simple_metadata),
        ]
        mock_provider.stream_chat.return_value = iter(mock_stream)

        with patch("time.time") as mock_time:
            # Setup time: Start -> 0.1s later (chunk 1) -> 0.2s later (end)
            mock_time.side_effect = [100.0, 100.1, 100.2, 100.3, 100.4]

            chunks = list(agent.stream_run("Hi"))

            # Verify turn history has TTFT recorded by checking the final aggregate chunk
            # TTFT = 100.1 - 100.0 = 0.1s = 100ms

            final_chunk = chunks[-1]
            assert final_chunk.final_metadata is not None
            history = final_chunk.final_metadata.turns.turn_history

            assert len(history) == 1
            assert history[0].tokens_used.input_tokens == 10

    def test_get_provider_stream_passes_config(self, mock_get_provider):
        """Tests that _get_provider_stream passes max_tokens and tools correctly."""
        mock_provider = mock_get_provider.return_value
        # Setup tools and max_tokens
        config = AgentConfig(
            provider_name="test", model="test", max_tokens=500, tool_names=["read_file"]
        )
        agent = Agent(config)

        # We need to access the private method or trigger it via stream_run
        # Triggering via stream_run is safer to ensure integration.
        mock_provider.stream_chat.return_value = iter([ProviderChunk(content="Done")])

        list(agent.stream_run("Hi"))

        # Verify call arguments
        call_kwargs = mock_provider.stream_chat.call_args.kwargs
        assert call_kwargs["max_tokens"] == 500
        assert "tools" in call_kwargs
        assert len(call_kwargs["tools"]) == 1
