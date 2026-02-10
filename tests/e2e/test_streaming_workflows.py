"""End-to-end tests for streaming workflows across all providers.

Tests comprehensive streaming functionality including:
- Streaming across all major providers
- Streaming with function calling (interleaved tool calls and text)
- Stream cancellation and resource cleanup

Uses MetadataBuilder for realistic metadata construction.
"""

import sys
from unittest.mock import MagicMock

import pytest

from tests.e2e.conftest import (
    create_test_metadata_with_builder,
    mock_provider_environment,
)

# Skip entire module on Python 3.9
if sys.version_info < (3, 10):
    pytest.skip("Google provider tests require Python 3.10+", allow_module_level=True)


from allos import Agent, AgentConfig
from allos.providers.base import ProviderChunk, ProviderResponse, ToolCall
from tests.conftest import PROVIDER_MODELS

pytestmark = pytest.mark.e2e


@pytest.mark.e2e
class TestStreamingWorkflows:
    """E2E tests for streaming functionality across providers.

    Tests streaming behavior including:
    1. Basic streaming across all providers
    2. Streaming with tool calls (interleaved chunks)
    3. Stream cancellation and cleanup

    Uses realistic metadata via MetadataBuilder.
    """

    def test_stream_all_providers(self, work_dir, available_providers):
        """
        Test basic streaming works for each available provider.

        Verifies:
        1. Provider yields ProviderChunk objects
        2. Content accumulates correctly
        3. Final metadata chunk is present
        4. Token counts are accurate
        5. TTFT (time to first token) is recorded
        6. No errors in stream
        """
        for provider_name in available_providers:
            # Skip Google on Python 3.9
            if provider_name == "google" and sys.version_info < (3, 10):
                continue

            with mock_provider_environment(provider_name):
                agent = Agent(
                    AgentConfig(
                        provider_name=provider_name,
                        model=PROVIDER_MODELS.get(provider_name, "default"),
                        tool_names=[],  # No tools for simple streaming test
                        auto_approve=True,
                    )
                )

                # Mock streaming response
                def mock_stream_response(
                    messages, provider_name=provider_name, **kwargs
                ):
                    """Simulate streaming text response."""
                    # Content chunks (simulate gradual token generation)
                    yield ProviderChunk(content="The ")
                    yield ProviderChunk(content="answer ")
                    yield ProviderChunk(content="is ")
                    yield ProviderChunk(content="42")
                    yield ProviderChunk(content=".")

                    # Final metadata chunk
                    provider_specific = None
                    if provider_name == "openai":
                        provider_specific = {
                            "openai": {
                                "system_fingerprint": f"fp_stream_{provider_name}"
                            }
                        }
                    elif provider_name == "ollama":
                        provider_specific = {
                            "ollama": {
                                "warm_up": False,
                                "warm_up_duration_seconds": None,
                            }
                        }
                    elif provider_name == "google":
                        provider_specific = {
                            "google": {
                                "used_thought_signatures": False,
                                "vertexai": False,
                            }
                        }

                    metadata = create_test_metadata_with_builder(
                        provider_name=provider_name,
                        model_id=PROVIDER_MODELS.get(provider_name, "default"),
                        input_tokens=20,
                        output_tokens=5,
                        provider_specific_fields=provider_specific,
                    )

                    yield ProviderChunk(final_metadata=metadata)

                agent.provider.stream_chat = MagicMock(side_effect=mock_stream_response)

                # Consume stream
                chunks = list(agent.stream_run("What is the answer?"))

                # Verify chunks received
                assert len(chunks) > 0, f"No chunks received for {provider_name}"

                # Verify content chunks
                content_chunks = [c for c in chunks if c.content]
                assert len(content_chunks) == 5, (
                    f"Expected 5 content chunks for {provider_name}, got {len(content_chunks)}"
                )

                # Verify content accumulation
                full_content = "".join(
                    c.content for c in content_chunks if c.content is not None
                )
                assert full_content == "The answer is 42.", (
                    f"Content mismatch for {provider_name}: {full_content}"
                )

                # Verify final metadata chunk
                metadata_chunks = [c for c in chunks if c.final_metadata]
                assert len(metadata_chunks) >= 1, (
                    f"No final metadata for {provider_name}"
                )

                # Verify agent metadata was updated
                assert agent.last_run_metadata is not None, (
                    f"No last_run_metadata for {provider_name}"
                )
                assert agent.last_run_metadata.usage.total_tokens == 25  # 20 + 5
                assert agent.last_run_metadata.usage.input_tokens == 20
                assert agent.last_run_metadata.usage.output_tokens == 5

                # Verify TTFT (time to first token) is recorded
                assert (
                    agent.last_run_metadata.latency.time_to_first_token_ms is not None
                ), f"TTFT not recorded for {provider_name}"
                assert agent.last_run_metadata.latency.time_to_first_token_ms >= 0, (
                    f"Invalid TTFT for {provider_name}"
                )

                # Verify no error chunks
                error_chunks = [c for c in chunks if c.error]
                assert len(error_chunks) == 0, (
                    f"Error chunks found for {provider_name}: {error_chunks}"
                )

    def test_stream_with_function_calling(self, work_dir, available_providers):
        """
        Test streaming with interleaved tool calls.

        Workflow:
        1. Agent streams thinking text
        2. Agent streams tool call (start, deltas, done)
        3. Tool executes
        4. Agent streams final response

        Verifies:
        1. Tool call chunks received properly
        2. Text and tool chunks are interleaved
        3. Tool execution happens between turns
        4. Final state is correct
        5. Metadata tracks tool calls
        """
        if len(available_providers) < 1:
            pytest.skip("Need at least 1 provider for streaming tool test")

        provider_name = available_providers[0]

        # Skip Google on Python 3.9
        if provider_name == "google" and sys.version_info < (3, 10):
            if len(available_providers) > 1:
                provider_name = available_providers[1]
            else:
                pytest.skip("No suitable provider available")

        with mock_provider_environment(provider_name):
            agent = Agent(
                AgentConfig(
                    provider_name=provider_name,
                    model=PROVIDER_MODELS.get(provider_name, "default"),
                    tool_names=["write_file"],
                    auto_approve=True,
                )
            )

            # Track which turn we're on
            call_count = [0]

            def mock_stream_with_tools(messages, **kwargs):
                """Simulate streaming with tool call."""
                call_count[0] += 1

                if call_count[0] == 1:
                    # Turn 1: Stream thinking + tool call
                    yield ProviderChunk(content="I'll ")
                    yield ProviderChunk(content="write ")
                    yield ProviderChunk(content="a file. ")

                    # Tool call chunks
                    yield ProviderChunk(
                        tool_call_start={
                            "id": "call_write_1",
                            "name": "write_file",
                            "index": 0,
                        }
                    )
                    yield ProviderChunk(tool_call_delta='{"path": "test')
                    yield ProviderChunk(tool_call_delta='.txt", "content": "')
                    yield ProviderChunk(tool_call_delta='Hello streaming"}')
                    yield ProviderChunk(
                        tool_call_done=ToolCall(
                            id="call_write_1",
                            name="write_file",
                            arguments={
                                "path": "test.txt",
                                "content": "Hello streaming",
                            },
                        )
                    )

                    # Final metadata for turn 1
                    metadata = create_test_metadata_with_builder(
                        provider_name=provider_name,
                        model_id=PROVIDER_MODELS.get(provider_name, "default"),
                        input_tokens=50,
                        output_tokens=30,
                    )
                    yield ProviderChunk(final_metadata=metadata)

                else:
                    # Turn 2: Stream final response after tool execution
                    yield ProviderChunk(content="File ")
                    yield ProviderChunk(content="created ")
                    yield ProviderChunk(content="successfully!")

                    # Final metadata for turn 2
                    metadata = create_test_metadata_with_builder(
                        provider_name=provider_name,
                        model_id=PROVIDER_MODELS.get(provider_name, "default"),
                        input_tokens=70,  # Includes tool result in context
                        output_tokens=20,
                    )
                    yield ProviderChunk(final_metadata=metadata)

            agent.provider.stream_chat = MagicMock(side_effect=mock_stream_with_tools)

            # Consume stream
            chunks = list(agent.stream_run("Create a test file"))

            # Verify chunks received
            assert len(chunks) > 0

            # Verify content chunks from both turns
            content_chunks = [c for c in chunks if c.content]
            assert len(content_chunks) >= 6  # 3 from turn 1, 3 from turn 2

            # Verify tool call chunks
            tool_start_chunks = [c for c in chunks if c.tool_call_start]
            assert len(tool_start_chunks) == 1

            tool_delta_chunks = [c for c in chunks if c.tool_call_delta]
            assert len(tool_delta_chunks) == 3

            tool_done_chunks = [c for c in chunks if c.tool_call_done]
            assert len(tool_done_chunks) == 1
            assert tool_done_chunks[0].tool_call_done
            assert tool_done_chunks[0].tool_call_done.name == "write_file"
            assert tool_done_chunks[0].tool_call_done.arguments["path"] == "test.txt"

            # Verify tool was executed (file should exist)
            test_file = work_dir / "test.txt"
            assert test_file.exists()
            assert test_file.read_text() == "Hello streaming"

            # Verify final metadata tracks tool calls
            assert agent.last_run_metadata is not None
            assert agent.last_run_metadata.tools.total_tool_calls == 1
            assert agent.last_run_metadata.turns.total_turns == 2

            # Verify aggregated token counts (from both turns)
            assert agent.last_run_metadata.usage.input_tokens == 120  # 50 + 70
            assert agent.last_run_metadata.usage.output_tokens == 50  # 30 + 20
            assert agent.last_run_metadata.usage.total_tokens == 170

    def test_stream_cancellation_early(self, work_dir, available_providers):
        """
        Test early stream cancellation (after first chunk).

        Verifies:
        1. Stream terminates cleanly when broken early
        2. No exceptions raised
        3. Partial chunks received
        4. Context state remains valid
        """
        if len(available_providers) < 1:
            pytest.skip("Need at least 1 provider for cancellation test")

        provider_name = available_providers[0]
        if provider_name == "google" and sys.version_info < (3, 10):
            if len(available_providers) > 1:
                provider_name = available_providers[1]
            else:
                pytest.skip("No suitable provider available")

        with mock_provider_environment(provider_name):
            agent = Agent(
                AgentConfig(
                    provider_name=provider_name,
                    model=PROVIDER_MODELS.get(provider_name, "default"),
                    tool_names=[],
                    auto_approve=True,
                )
            )

            def mock_long_stream(messages, **kwargs):
                """Simulate a long streaming response."""
                for i in range(100):
                    yield ProviderChunk(content=f"word{i} ")

                metadata = create_test_metadata_with_builder(
                    provider_name=provider_name,
                    model_id=PROVIDER_MODELS.get(provider_name, "default"),
                    input_tokens=10,
                    output_tokens=100,
                )
                yield ProviderChunk(final_metadata=metadata)

            agent.provider.stream_chat = MagicMock(side_effect=mock_long_stream)

            # Start streaming but cancel early
            stream = agent.stream_run("Generate a long text")
            chunks_received = []

            try:
                for i, chunk in enumerate(stream):
                    chunks_received.append(chunk)
                    if i >= 2:  # Cancel after 3 chunks
                        break
            except Exception as e:
                pytest.fail(f"Exception during early cancellation: {e}")

            # Verify partial chunks received
            assert len(chunks_received) == 3
            assert all(c.content for c in chunks_received)

            # Verify no exceptions and context is valid
            assert len(agent.context.messages) > 0
            # User message should be in context
            assert any(
                m.content == "Generate a long text" for m in agent.context.messages
            )

    def test_stream_cancellation_mid_stream(self, work_dir, available_providers):
        """
        Test mid-stream cancellation (during content generation).

        Verifies:
        1. Stream can be stopped mid-way
        2. Cleanup is proper (no orphaned resources)
        3. Partial state is accessible
        4. Agent can continue with new prompts
        """
        if len(available_providers) < 1:
            pytest.skip("Need at least 1 provider for mid-stream cancellation test")

        provider_name = available_providers[0]
        if provider_name == "google" and sys.version_info < (3, 10):
            if len(available_providers) > 1:
                provider_name = available_providers[1]
            else:
                pytest.skip("No suitable provider available")

        with mock_provider_environment(provider_name):
            agent = Agent(
                AgentConfig(
                    provider_name=provider_name,
                    model=PROVIDER_MODELS.get(provider_name, "default"),
                    tool_names=["write_file"],
                    auto_approve=True,
                )
            )

            call_count = [0]

            def mock_stream_for_cancellation(messages, **kwargs):
                """Simulate streaming that will be cancelled mid-way."""
                call_count[0] += 1

                if call_count[0] == 1:
                    # First call: will be cancelled
                    for i in range(50):
                        yield ProviderChunk(content=f"chunk{i} ")

                    metadata = create_test_metadata_with_builder(
                        provider_name=provider_name,
                        model_id=PROVIDER_MODELS.get(provider_name, "default"),
                        input_tokens=20,
                        output_tokens=50,
                    )
                    yield ProviderChunk(final_metadata=metadata)
                else:
                    # Second call: agent continues with new prompt
                    yield ProviderChunk(content="New response")

                    metadata = create_test_metadata_with_builder(
                        provider_name=provider_name,
                        model_id=PROVIDER_MODELS.get(provider_name, "default"),
                        input_tokens=15,
                        output_tokens=10,
                    )
                    yield ProviderChunk(final_metadata=metadata)

            agent.provider.stream_chat = MagicMock(
                side_effect=mock_stream_for_cancellation
            )

            # First stream: cancel mid-way
            stream1 = agent.stream_run("First prompt")
            chunks1 = []

            try:
                for i, chunk in enumerate(stream1):
                    chunks1.append(chunk)
                    if i >= 9:  # Cancel after 10 chunks (mid-stream)
                        break
            except Exception as e:
                pytest.fail(f"Exception during mid-stream cancellation: {e}")

            # Verify partial chunks received
            assert len(chunks1) == 10

            # Verify agent can continue with new prompt
            # Note: Context might be in partial state, but agent should handle it
            chunks2 = list(agent.stream_run("Second prompt"))

            # Verify second stream works
            content2 = "".join(c.content for c in chunks2 if c.content)
            assert "New response" in content2

    def test_stream_consistency_vs_sync(self, work_dir, available_providers):
        """
        Verify streaming produces same results as synchronous execution.

        This is a regression test to ensure streaming accumulation logic
        matches the simpler synchronous implementation.

        Verifies:
        1. Token counts match (input, output, total)
        2. Cost estimates match
        3. Tool calls tracked identically
        4. Final content is the same
        """
        if len(available_providers) < 1:
            pytest.skip("Need at least 1 provider for consistency test")

        provider_name = available_providers[0]
        if provider_name == "google" and sys.version_info < (3, 10):
            if len(available_providers) > 1:
                provider_name = available_providers[1]
            else:
                pytest.skip("No suitable provider available")

        with mock_provider_environment(provider_name):
            # Agent 1: Synchronous
            agent_sync = Agent(
                AgentConfig(
                    provider_name=provider_name,
                    model=PROVIDER_MODELS.get(provider_name, "default"),
                    tool_names=[],
                    auto_approve=True,
                )
            )

            # Agent 2: Streaming
            agent_stream = Agent(
                AgentConfig(
                    provider_name=provider_name,
                    model=PROVIDER_MODELS.get(provider_name, "default"),
                    tool_names=[],
                    auto_approve=True,
                )
            )

            # Deterministic response content
            test_content = "The answer is 42."

            # Mock synchronous response
            metadata_sync = create_test_metadata_with_builder(
                provider_name=provider_name,
                model_id=PROVIDER_MODELS.get(provider_name, "default"),
                input_tokens=25,
                output_tokens=15,
            )

            agent_sync.provider.chat = MagicMock(
                return_value=ProviderResponse(
                    content=test_content,
                    tool_calls=[],
                    metadata=metadata_sync,
                )
            )

            # Mock streaming response (same content, but chunked)
            def mock_stream_equivalent(messages, **kwargs):
                # Chunk the same content
                words = test_content.split()
                for word in words:
                    yield ProviderChunk(content=word + " ")

                metadata_stream = create_test_metadata_with_builder(
                    provider_name=provider_name,
                    model_id=PROVIDER_MODELS.get(provider_name, "default"),
                    input_tokens=25,
                    output_tokens=15,
                )
                yield ProviderChunk(final_metadata=metadata_stream)

            agent_stream.provider.stream_chat = MagicMock(
                side_effect=mock_stream_equivalent
            )

            # Run both
            response_sync = agent_sync.run("What is the answer?")
            chunks_stream = list(agent_stream.stream_run("What is the answer?"))

            # Verify content matches
            content_stream = "".join(c.content for c in chunks_stream if c.content)
            assert response_sync.strip() == content_stream.strip()

            # Verify metadata matches
            meta_sync = agent_sync.last_run_metadata
            meta_stream = agent_stream.last_run_metadata

            assert meta_sync is not None and meta_stream is not None

            # Token counts should match exactly
            assert meta_sync.usage.input_tokens == meta_stream.usage.input_tokens
            assert meta_sync.usage.output_tokens == meta_stream.usage.output_tokens
            assert meta_sync.usage.total_tokens == meta_stream.usage.total_tokens

            # Costs should match
            assert meta_sync.usage.estimated_cost and meta_stream.usage.estimated_cost
            assert (
                meta_sync.usage.estimated_cost.total_usd
                == meta_stream.usage.estimated_cost.total_usd
            )
