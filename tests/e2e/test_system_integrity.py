# tests/integration/test_system_integrity.py

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from allos.agent import Agent, AgentConfig
from allos.context import ConversationContext
from allos.providers.base import (
    ProviderChunk,
    ProviderResponse,
    ToolCall,
)
from allos.tools import BaseTool, ToolParameter, tool
from allos.tools.base import ToolPermission
from allos.tools.registry import _tool_registry
from allos.utils.errors import ContextWindowExceededError, FileOperationError

# We use the integration marker, but these specific tests mock the network layer
# to verify logic/schema handling without incurring API costs or requiring keys.
# pytestmark = pytest.mark.integration


class TestSystemIntegrity:
    """
    Critical logic tests for system boundaries, error recovery, and state persistence.
    These tests verify that the 'Reason-Act' loop holds together under stress.
    """

    @patch("allos.providers.anthropic.anthropic.Anthropic")
    def test_frankenstein_session_openai_to_anthropic(
        self, MockAnthropic, work_dir: Path, mock_metadata_factory
    ):
        """
        Scenario 1: Cross-Provider Hydration ("The Frankenstein Test")

        Goal: Verify that a session started with OpenAI (which generates specific tool call IDs
        and formats) can be saved, loaded, and successfully continued by Anthropic,
        which has a strictly different message format (content blocks).
        """
        # 1. Simulate an OpenAI session state
        # Context: User -> Asst (Call Tool) -> Tool (Result)
        context = ConversationContext()
        context.add_user_message("List files")

        # OpenAI style tool call ID
        openai_call_id = "call_openai_12345"

        context.add_assistant_message(
            content=None,
            tool_calls=[
                ToolCall(id=openai_call_id, name="list_directory", arguments={})
            ],
        )
        context.add_tool_result_message(openai_call_id, '{"files": ["test.txt"]}')

        # 2. Save this "OpenAI-generated" state to disk
        session_file = work_dir / "frankenstein.json"
        config_data = {
            "provider_name": "openai",
            "model": "gpt-4o",
            "tool_names": ["list_directory"],
        }
        with open(session_file, "w") as f:
            json.dump({"config": config_data, "context": context.to_dict()}, f)

        # 3. Load Session but FORCE provider to Anthropic
        # We manually load to simulate the CLI override behavior or manual config change
        agent = Agent.load_session(session_file)
        agent.config.provider_name = "anthropic"
        agent.config.model = "claude-haiku-4-5"

        # Re-initialize provider (this happens in CLI, we do it manually here)
        # This will trigger AnthropicProvider.__init__ which uses our MockAnthropic
        from allos.providers import ProviderRegistry

        agent.provider = ProviderRegistry.get_provider(
            "anthropic", model="claude-haiku-4-5"
        )

        # Mock the Anthropic client to capture what gets sent
        mock_client = MockAnthropic.return_value
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(type="text", text="I see the files.")],
            stop_reason="end_turn",
            usage=MagicMock(input_tokens=10, output_tokens=5),
        )

        # 4. Execution: The Agent tries to send the OpenAI-flavored history to Anthropic
        agent.run("What files are there?")

        # 5. Verification: Did the adapter layer correctly translate the history?
        call_kwargs = mock_client.messages.create.call_args.kwargs
        sent_messages = call_kwargs["messages"]

        # Check the Assistant Message (Index 1) conversion
        # OpenAI had `tool_calls` list -> Anthropic needs `content` list with `tool_use` block
        asst_msg = sent_messages[1]
        assert asst_msg["role"] == "assistant"
        assert isinstance(asst_msg["content"], list)
        assert asst_msg["content"][0]["type"] == "tool_use"
        assert asst_msg["content"][0]["id"] == openai_call_id  # The ID must persist!

        # Check the Tool Result Message (Index 2) conversion
        # OpenAI had `role="tool"` -> Anthropic needs `role="user"` with `tool_result` block
        tool_msg = sent_messages[2]
        assert tool_msg["role"] == "user"
        assert isinstance(tool_msg["content"], list)
        assert tool_msg["content"][0]["type"] == "tool_result"
        assert tool_msg["content"][0]["tool_use_id"] == openai_call_id

    def test_streaming_vs_sync_metadata_parity(
        self, mock_get_provider, mock_metadata_factory
    ):
        """
        Scenario 2: Streaming vs. Synchronous Parity

        Goal: Verify that the complex streaming accumulation logic produces the exact
        same token counts and cost estimates as the simple synchronous implementation.
        """
        # Setup: A deterministic response sequence
        content_str = "A" * 10  # 10 chars
        # Simple metadata to be returned by both mocks
        fixed_usage = {"input_tokens": 10, "output_tokens": 10}

        # --- Run 1: Synchronous ---
        config = AgentConfig(provider_name="test", model="test")
        agent_sync = Agent(config)

        mock_sync_provider = mock_get_provider.return_value
        mock_sync_provider.chat.return_value = ProviderResponse(
            content=content_str, metadata=mock_metadata_factory(usage=fixed_usage)
        )

        agent_sync.run("Prompt")
        sync_meta = agent_sync.last_run_metadata

        # --- Run 2: Streaming ---
        agent_stream = Agent(config)
        mock_stream_provider = mock_get_provider.return_value

        # Simulate chunks: Content -> Final Metadata
        chunks = [
            ProviderChunk(content=content_str),
            ProviderChunk(final_metadata=mock_metadata_factory(usage=fixed_usage)),
        ]
        mock_stream_provider.stream_chat.return_value = iter(chunks)

        # Consume stream
        list(agent_stream.stream_run("Prompt"))
        stream_meta = agent_stream.last_run_metadata

        # --- Assertion ---
        # The totals should match exactly
        assert sync_meta and stream_meta
        assert sync_meta.usage.input_tokens == stream_meta.usage.input_tokens
        assert sync_meta.usage.output_tokens == stream_meta.usage.output_tokens
        assert sync_meta.usage.total_tokens == stream_meta.usage.total_tokens

        # Even cost should match (assuming static pricing dict is stable)
        if sync_meta.usage.estimated_cost and stream_meta.usage.estimated_cost:
            assert (
                sync_meta.usage.estimated_cost.total_usd
                == stream_meta.usage.estimated_cost.total_usd
            )

    def test_agent_recovery_from_tool_failure(
        self, mock_get_provider, mock_metadata_factory
    ):
        """
        Scenario 3: Agentic Error Recovery (The "Stumble" Test)

        Goal: Verify that when a tool raises an Exception (simulating File Not Found),
        the Agent does NOT crash. Instead, it must catch the error, add it to the context,
        and let the LLM see the error in the next turn.
        """
        config = AgentConfig(
            provider_name="test",
            model="test",
            tool_names=["read_file"],
            auto_approve=True,
        )
        agent = Agent(config)

        # 1. Setup Tool to Fail
        # Prepare a mock tool that behaves exactly how we want
        mock_tool_instance = MagicMock(spec=BaseTool)
        mock_tool_instance.name = "read_file"
        mock_tool_instance.description = "Reads a file"
        mock_tool_instance.permission = ToolPermission.ALWAYS_ALLOW
        mock_tool_instance.execute.side_effect = FileOperationError(
            "Simulated Disk Error"
        )

        # Patch ToolRegistry to return our specific mock instance
        with patch(
            "allos.agent.agent.ToolRegistry.get_tool", return_value=mock_tool_instance
        ):
            # 2. Setup Provider Turns
            # Turn 1: LLM calls read_file
            # Turn 2: LLM sees error and apologizes
            mock_provider = mock_get_provider.return_value

            turn_1_resp = ProviderResponse(
                tool_calls=[
                    ToolCall(
                        id="call_1", name="read_file", arguments={"path": "bad.txt"}
                    )
                ],
                metadata=mock_metadata_factory(),
            )
            turn_2_resp = ProviderResponse(
                content="I apologize, I could not read the file due to a disk error.",
                metadata=mock_metadata_factory(),
            )
            mock_provider.chat.side_effect = [turn_1_resp, turn_2_resp]

            # 3. Execution
            response = agent.run("Read bad.txt")

            # 4. Assertions
            assert "I apologize" in response

            # Check Context for the Error Trace
            # Context: User -> Asst(Call) -> Tool(Error) -> Asst(Response)
            assert len(agent.context) == 4
            error_msg = agent.context.messages[2]
            assert error_msg.role == "tool"
            # Standardized error JSON format check
            assert error_msg.content and '"status": "error"' in error_msg.content
            assert error_msg.content and "Simulated Disk Error" in error_msg.content

    def test_context_window_safety_limit(
        self, mock_get_provider, mock_metadata_factory
    ):
        """
        Scenario 4: Context Window Limit (The "Data Dump" Test)

        Goal: Ensure the SDK proactively stops a request *before* sending it to the provider
        if the context history has grown too large, preventing expensive API errors.
        """
        config = AgentConfig(provider_name="test", model="test")
        agent = Agent(config)

        # Mock Provider reporting a tiny window
        mock_provider = mock_get_provider.return_value
        # Use a limit larger than buffer (2048) so the check passes initially
        mock_provider.get_context_window.return_value = 4096

        # 1. Fill context to be safe (e.g. 50 tokens)
        safe_text = "word " * 50
        agent.context.add_user_message(safe_text)

        # 2. Run - Should pass (mock returns generic response)
        mock_provider.chat.return_value = ProviderResponse(
            content="ok", metadata=mock_metadata_factory()
        )
        agent.run("Go")

        # 3. Fill context to overflow (buffer is 2048, so limit is ~2048)
        # Add 3000 words (~3000 tokens)
        overflow_text = "word " * 3000
        # We manually add to context to simulate a large previous turn result
        agent.context.add_assistant_message(overflow_text)

        # 4. Run - Should Fail PROACTIVELY
        # We reset mock to ensure it's not called this time
        mock_provider.chat.reset_mock()

        with pytest.raises(ContextWindowExceededError) as exc:
            agent.run("Next")

        assert "Conversation context has grown too large" in str(exc.value)
        # CRITICAL: The provider should NOT have been called.
        # The SDK saved us money/latency.
        mock_provider.chat.assert_not_called()

    @patch("allos.agent.agent.Agent._check_tool_permission", return_value=True)
    def test_tool_argument_self_correction(
        self, mock_perm, mock_get_provider, mock_metadata_factory
    ):
        """
        Scenario 5: Tool Argument Hallucination Handling (Self-Correction)

        Goal: Verify that if the LLM hallucinates an invalid argument type (e.g. string instead of int),
        the system feeds the `ToolError` back to the LLM, allowing it to retry with correct types
        in the same `run()` loop.
        """
        # Register the tool temporarily for this test
        tool_name = "add_numbers"

        @tool
        class StrictMathTool(BaseTool):
            name = tool_name
            parameters = [
                ToolParameter("x", "integer", "First number"),
                ToolParameter("y", "integer", "Second number"),
            ]

            def execute(self, **kwargs):
                return {"sum": kwargs["x"] + kwargs["y"]}

        try:
            config = AgentConfig(
                provider_name="test", model="test", tool_names=[tool_name]
            )
            agent = Agent(config)

            # 2. Setup Provider Turns (The "Retry" Loop)
            mock_provider = mock_get_provider.return_value

            # Turn 1: LLM hallucinates "five" (string) instead of 5 (int)
            # Note: 'read_file' mock factory isn't used here, we use manual responses
            turn_1_bad = ProviderResponse(
                tool_calls=[
                    ToolCall(
                        id="c1", name="add_numbers", arguments={"x": "five", "y": 10}
                    )
                ],
                metadata=mock_metadata_factory(),
            )

            # Turn 2: LLM sees the error and retries with 5 (int)
            turn_2_good = ProviderResponse(
                tool_calls=[
                    ToolCall(id="c2", name="add_numbers", arguments={"x": 5, "y": 10})
                ],
                metadata=mock_metadata_factory(),
            )

            # Turn 3: Final Answer
            turn_3_final = ProviderResponse(
                content="The sum is 15", metadata=mock_metadata_factory()
            )

            mock_provider.chat.side_effect = [turn_1_bad, turn_2_good, turn_3_final]

            # 3. Execution
            result = agent.run("Add five and ten")

            # 4. Assertions
            assert result == "The sum is 15"

            # Verify the error capture
            error_msg = agent.context.messages[2]
            assert error_msg.role == "tool"
            assert error_msg.content and '"status": "error"' in error_msg.content
            assert error_msg.content and "Invalid type" in error_msg.content

        finally:
            # Cleanup registry
            if tool_name in _tool_registry:
                del _tool_registry[tool_name]

    @patch("allos.agent.agent.Agent._check_tool_permission", return_value=True)
    def test_parallel_tool_calling(
        self, mock_perm, mock_get_provider, mock_metadata_factory
    ):
        """
        Scenario 6: Parallel Tool Calling

        Goal: Verify that the agent correctly handles multiple tool calls returned
        in a single LLM turn. It must execute all of them and maintain the correct
        tool_call_id mapping in the history.
        """
        config = AgentConfig(
            provider_name="test", model="test", tool_names=["read_file"]
        )
        agent = Agent(config)

        # 1. Setup a Mock Tool that we can track
        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = "get_weather"
        # Return different values based on input to verify mapping
        mock_tool.execute.side_effect = lambda location: {
            "temp": 80 if location == "TX" else 50
        }

        with patch("allos.agent.agent.ToolRegistry.get_tool", return_value=mock_tool):
            mock_provider = mock_get_provider.return_value

            # Turn 1: LLM requests TWO tools at once
            turn_1_parallel = ProviderResponse(
                tool_calls=[
                    ToolCall(
                        id="call_A", name="get_weather", arguments={"location": "TX"}
                    ),
                    ToolCall(
                        id="call_B", name="get_weather", arguments={"location": "NY"}
                    ),
                ],
                metadata=mock_metadata_factory(),
            )

            # Turn 2: Final Summary
            turn_2_final = ProviderResponse(
                content="TX is 80, NY is 50", metadata=mock_metadata_factory()
            )

            mock_provider.chat.side_effect = [turn_1_parallel, turn_2_final]

            # 2. Execution
            result = agent.run("Compare weather in TX and NY")

            # 3. Assertions
            assert "TX is 80" in result

            # Verify Execute was called twice
            assert mock_tool.execute.call_count == 2

            # Verify Context Integrity
            # 0: User
            # 1: Asst (Parallel Calls - should store BOTH in one message)
            # 2: Tool Result A
            # 3: Tool Result B
            # 4: Asst (Final)
            assert len(agent.context) == 5

            asst_msg = agent.context.messages[1]
            assert len(asst_msg.tool_calls) == 2
            assert asst_msg.tool_calls[0].id == "call_A"
            assert asst_msg.tool_calls[1].id == "call_B"

            # Check Result mapping
            res_A = agent.context.messages[2]
            res_B = agent.context.messages[3]

            assert res_A.role == "tool" and res_A.tool_call_id == "call_A"
            assert res_B.role == "tool" and res_B.tool_call_id == "call_B"

    @patch(
        "rich.console.Console.input", return_value="y"
    )  # User says 'Yes' to the prompt
    def test_security_blocklist_enforcement(
        self, mock_input, mock_get_provider, mock_metadata_factory
    ):
        """
        Scenario 7: The "Malicious Compliance" Security Test

        Goal: Verify that even if the User APPROVES a dangerous command (or auto-approve is on),
        the `ShellExecuteTool` internal blocklist acts as a final fail-safe and prevents execution.
        """
        # Note: We must NOT mock the ShellExecuteTool here. We need the REAL logic
        # to ensure the 'DANGEROUS_COMMANDS' check triggers.
        config = AgentConfig(
            provider_name="test", model="test", tool_names=["shell_exec"]
        )
        agent = Agent(config)

        mock_provider = mock_get_provider.return_value

        # Turn 1: LLM tries to run a dangerous command
        turn_1_dangerous = ProviderResponse(
            tool_calls=[
                ToolCall(id="c1", name="shell_exec", arguments={"command": "rm -rf /"})
            ],
            metadata=mock_metadata_factory(),
        )

        # Turn 2: LLM sees the blocklist error
        turn_2_response = ProviderResponse(
            content="I cannot execute that command due to safety protocols.",
            metadata=mock_metadata_factory(),
        )

        mock_provider.chat.side_effect = [turn_1_dangerous, turn_2_response]

        # We need to spy on subprocess.run to ensure it was NEVER called
        with patch("subprocess.run") as mock_subprocess:
            result = agent.run("Destroy the system")

            # Assertions
            assert "cannot execute" in result

            # Critical: Subprocess must NOT have been called
            mock_subprocess.assert_not_called()

            # Context Verification
            # The tool should have returned an error status locally
            error_msg = agent.context.messages[2]
            assert error_msg.role == "tool"
            assert error_msg.content and '"status": "error"' in error_msg.content
            assert error_msg.content and "blocklist" in error_msg.content
