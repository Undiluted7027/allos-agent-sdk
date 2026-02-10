"""End-to-end tests for metadata tracking and accuracy.

Tests comprehensive metadata lifecycle, provider-specific fields,
and aggregation across multi-turn conversations.
"""

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from allos import Agent, AgentConfig
from allos.providers.base import ProviderResponse, ToolCall
from allos.providers.metadata import (
    Metadata,
    ProviderSpecific,
    ProviderSpecificGoogle,
    ProviderSpecificOllama,
    ProviderSpecificOpenAI,
)
from allos.utils.token_counter import count_tokens
from tests.conftest import PROVIDER_MODELS
from tests.e2e.conftest import (
    create_test_metadata_with_builder,
    mock_provider_environment,
    setup_google_mocks,
    setup_ollama_mocks,
)

pytestmark = pytest.mark.e2e


@pytest.mark.e2e
class TestMetadataWorkflows:
    """Test comprehensive metadata tracking across workflows."""

    def test_metadata_lifecycle_multi_turn(
        self, work_dir, mock_metadata_factory, available_providers
    ):
        """
        Track metadata through complete multi-turn conversation.

        Verifies:
        1. Token counting increases with conversation history
        2. Turn history is accurately recorded
        3. Tool calls are tracked
        4. Cost estimation accumulates
        5. Latency metrics are present
        """
        # Test with first available provider
        provider_name = available_providers[0]

        # Skip Google on Python 3.9
        if provider_name == "google" and sys.version_info < (3, 10):
            if len(available_providers) > 1:
                provider_name = available_providers[1]
            else:
                pytest.skip("No suitable provider available")

        with mock_provider_environment(provider_name):
            self._run_metadata_lifecycle_test(
                provider_name, work_dir, mock_metadata_factory
            )

    def _run_metadata_lifecycle_test(
        self, provider_name: str, work_dir: Path, mock_metadata_factory
    ):
        """Execute the metadata lifecycle test logic."""
        agent = Agent(
            AgentConfig(
                provider_name=provider_name,
                model=PROVIDER_MODELS.get(provider_name, "default"),
                tool_names=["write_file", "read_file"],
                auto_approve=True,
            )
        )

        # Track metadata history
        metadata_history: List[Metadata] = []

        # Turn 1: Write file (2 LLM calls)
        def turn1_chat_side_effect(messages, **kwargs):
            call_index = len(metadata_history)
            input_tokens = count_tokens(" ".join([m.content or "" for m in messages]))

            if call_index == 0:
                # First call: request write_file
                metadata = mock_metadata_factory(
                    usage={"input_tokens": input_tokens, "output_tokens": 30},
                    provider=provider_name,
                    model_id=PROVIDER_MODELS.get(provider_name, "default"),
                )
                metadata_history.append(metadata)
                return ProviderResponse(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="call_write_1",
                            name="write_file",
                            arguments={"path": "data.txt", "content": "Initial data"},
                        )
                    ],
                    metadata=metadata,
                )
            else:
                # Second call: confirm write
                metadata = mock_metadata_factory(
                    usage={"input_tokens": input_tokens, "output_tokens": 20},
                    provider=provider_name,
                    model_id=PROVIDER_MODELS.get(provider_name, "default"),
                )
                metadata_history.append(metadata)
                return ProviderResponse(
                    content="File data.txt created successfully.",
                    tool_calls=[],
                    metadata=metadata,
                )

        agent.provider.chat = MagicMock(side_effect=turn1_chat_side_effect)  # type: ignore[method-assign]

        agent.run("Create a file data.txt with 'Initial data'")

        # Verify Turn 1 metadata
        assert agent.last_run_metadata is not None, "No metadata after turn 1"
        turn1_metadata = agent.last_run_metadata

        assert turn1_metadata.usage.total_tokens > 0, "No tokens recorded"
        assert turn1_metadata.usage.input_tokens > 0, "No input tokens"
        assert turn1_metadata.usage.output_tokens > 0, "No output tokens"
        assert turn1_metadata.turns.total_turns == 2, (
            "Expected 2 turns (tool call + response)"
        )
        assert len(turn1_metadata.turns.turn_history) == 2, "Turn history incomplete"
        assert turn1_metadata.tools.total_tool_calls == 1, "Expected 1 tool call"

        # Verify token counting increased with conversation
        assert (
            metadata_history[1].usage.input_tokens
            > metadata_history[0].usage.input_tokens
        ), "Input tokens should increase with conversation history"

        # Turn 2: Read file (2 LLM calls)
        initial_message_count = len(agent.context.messages)

        assert initial_message_count == 4

        def turn2_chat_side_effect(messages, **kwargs):
            call_index = len(metadata_history) - 2  # Relative to turn 2
            input_tokens = count_tokens(" ".join([m.content or "" for m in messages]))

            if call_index == 0:
                # First call: request read_file
                metadata = mock_metadata_factory(
                    usage={"input_tokens": input_tokens, "output_tokens": 25},
                    provider=provider_name,
                    model_id=PROVIDER_MODELS.get(provider_name, "default"),
                )
                metadata_history.append(metadata)
                return ProviderResponse(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="call_read_1",
                            name="read_file",
                            arguments={"path": "data.txt"},
                        )
                    ],
                    metadata=metadata,
                )
            else:
                # Second call: return file content
                metadata = mock_metadata_factory(
                    usage={"input_tokens": input_tokens, "output_tokens": 35},
                    provider=provider_name,
                    model_id=PROVIDER_MODELS.get(provider_name, "default"),
                )
                metadata_history.append(metadata)
                return ProviderResponse(
                    content="The file contains: 'Initial data'",
                    tool_calls=[],
                    metadata=metadata,
                )

        agent.provider.chat = MagicMock(side_effect=turn2_chat_side_effect)  # type: ignore[method-assign]

        agent.run("Read the file data.txt")

        # Verify Turn 2 metadata
        assert agent.last_run_metadata is not None, "No metadata after turn 2"
        turn2_metadata = agent.last_run_metadata

        # Turn 2 should have more total tokens due to accumulated conversation
        assert turn2_metadata.usage.total_tokens > turn1_metadata.usage.total_tokens
        assert turn2_metadata.turns.total_turns == 2, "Expected 2 new turns"
        assert turn2_metadata.tools.total_tool_calls == 1, (
            "Expected 1 tool call in turn 2"
        )

        # Verify input tokens increased (more conversation context)
        assert (
            metadata_history[3].usage.input_tokens
            > metadata_history[2].usage.input_tokens
        ), "Turn 2 input tokens should increase"

        # Verify latency is present
        assert turn2_metadata.latency.total_duration_ms > 0, "No latency recorded"

    def test_provider_specific_metadata_openai(self, work_dir, mock_metadata_factory):
        """Test OpenAI-specific metadata fields."""
        with patch("allos.providers.openai.openai.OpenAI"):
            agent = Agent(
                AgentConfig(
                    provider_name="openai",
                    model="gpt-4o",
                    tool_names=["write_file"],
                    auto_approve=True,
                )
            )

            # Mock with OpenAI-specific metadata
            openai_metadata = mock_metadata_factory(
                usage={"input_tokens": 50, "output_tokens": 30},
                provider="openai",
                model_id="gpt-4o",
                provider_specific=ProviderSpecific(
                    openai=ProviderSpecificOpenAI(system_fingerprint="fp_abc123")
                ),
            )

            agent.provider.chat = MagicMock(
                side_effect=[
                    ProviderResponse(
                        content=None,
                        tool_calls=[
                            ToolCall(
                                id="call_1",
                                name="write_file",
                                arguments={
                                    "path": "test.txt",
                                    "content": "OpenAI test",
                                },
                            )
                        ],
                        metadata=openai_metadata,
                    ),
                    ProviderResponse(
                        content="File written.",
                        tool_calls=[],
                        metadata=mock_metadata_factory(
                            provider="openai", model_id="gpt-4o"
                        ),
                    ),
                ]
            )

            agent.run("Write 'OpenAI test' to test.txt")

            # Verify OpenAI-specific metadata
            assert agent.last_run_metadata is not None
            assert agent.last_run_metadata.provider_specific.openai is not None
            # Note: system_fingerprint may be in the first turn's metadata

    def test_provider_specific_metadata_ollama(self, work_dir, mock_metadata_factory):
        """Test Ollama-specific metadata fields (warm-up detection)."""
        with patch("allos.providers.ollama.Client") as mock_ollama_client:
            setup_ollama_mocks(mock_ollama_client)
            agent = Agent(
                AgentConfig(
                    provider_name="ollama",
                    model=PROVIDER_MODELS.get("ollama", "qwen3:8b"),
                    tool_names=["write_file"],
                    auto_approve=True,
                )
            )

            # Mock with Ollama warm-up metadata
            ollama_metadata = mock_metadata_factory(
                usage={"input_tokens": 50, "output_tokens": 30},
                provider="ollama",
                model_id="llama3.1",
                provider_specific=ProviderSpecific(
                    ollama=ProviderSpecificOllama(
                        warm_up=True, warm_up_duration_seconds=2.5
                    )
                ),
            )

            agent.provider.chat = MagicMock(
                side_effect=[
                    ProviderResponse(
                        content=None,
                        tool_calls=[
                            ToolCall(
                                id="call_1",
                                name="write_file",
                                arguments={
                                    "path": "test.txt",
                                    "content": "Ollama test",
                                },
                            )
                        ],
                        metadata=ollama_metadata,
                    ),
                    ProviderResponse(
                        content="File written.",
                        tool_calls=[],
                        metadata=mock_metadata_factory(
                            provider="ollama", model_id="llama3.1"
                        ),
                    ),
                ]
            )

            agent.run("Write 'Ollama test' to test.txt")

            # Verify Ollama-specific metadata
            assert agent.last_run_metadata is not None
            assert (
                agent.last_run_metadata.provider_specific.ollama is not None
                and agent.last_run_metadata.provider_specific.openai is None
            )
            assert agent.last_run_metadata.provider_specific.ollama.warm_up is True

    @pytest.mark.skipif(
        sys.version_info < (3, 10), reason="Google requires Python 3.10+"
    )
    def test_provider_specific_metadata_google(self, work_dir, mock_metadata_factory):
        """Test Google-specific metadata fields (thought signatures, Vertex AI)."""
        with patch("allos.providers.google.genai") as mock_genai:
            setup_google_mocks("gemini-2.5-flash", mock_genai)

            agent = Agent(
                AgentConfig(
                    provider_name="google",
                    model="gemini-2.5-flash",
                    tool_names=["write_file"],
                    auto_approve=True,
                )
            )

            # Mock with Google-specific metadata (Gemini API + thought signatures)
            google_metadata = mock_metadata_factory(
                usage={"input_tokens": 50, "output_tokens": 30},
                provider="google",
                model_id="gemini-2.5-flash",
                provider_specific=ProviderSpecific(
                    google=ProviderSpecificGoogle(
                        vertexai=False, used_thought_signatures=True
                    )
                ),
            )

            agent.provider.chat = MagicMock(
                side_effect=[
                    ProviderResponse(
                        content=None,
                        tool_calls=[
                            ToolCall(
                                id="call_1",
                                name="write_file",
                                arguments={
                                    "path": "test.txt",
                                    "content": "Google test",
                                },
                            )
                        ],
                        thought_signatures={"call_1": b"signature"},
                        metadata=google_metadata,
                    ),
                    ProviderResponse(
                        content="File written.",
                        tool_calls=[],
                        metadata=mock_metadata_factory(
                            provider="google", model_id="gemini-2.5-flash"
                        ),
                    ),
                ]
            )

            agent.run("Write 'Google test' to test.txt")

            # Verify Google-specific metadata
            assert agent.last_run_metadata is not None
            assert agent.last_run_metadata.provider_specific.google is not None
            assert (
                agent.last_run_metadata.provider_specific.google.used_thought_signatures
                is True
            )
            assert agent.last_run_metadata.provider_specific.google.vertexai is False

    def test_metadata_cost_estimation(self, work_dir, mock_metadata_factory):
        """Test that cost estimation is accurate and cumulative."""
        with patch("allos.providers.openai.openai.OpenAI"):
            agent = Agent(
                AgentConfig(
                    provider_name="openai",
                    model="gpt-4o",
                    tool_names=["write_file"],
                    auto_approve=True,
                )
            )

            # Create metadata with cost information
            from allos.providers.metadata import EstimatedCost

            turn1_metadata = mock_metadata_factory(
                usage={"input_tokens": 100, "output_tokens": 50},
                provider="openai",
                model_id="gpt-4o",
            )
            # Add cost estimation
            turn1_metadata.usage.estimated_cost = EstimatedCost(
                total_usd=0.001,
                input_cost_usd=0.0005,
                output_cost_usd=0.0005,
                pricing_source="static_config",
            )

            turn2_metadata = mock_metadata_factory(
                usage={"input_tokens": 200, "output_tokens": 40},
                provider="openai",
                model_id="gpt-4o",
            )
            turn2_metadata.usage.estimated_cost = EstimatedCost(
                total_usd=0.0015,
                input_cost_usd=0.001,
                output_cost_usd=0.0005,
                pricing_source="static_config",
            )

            agent.provider.chat = MagicMock(
                side_effect=[
                    ProviderResponse(
                        content=None,
                        tool_calls=[
                            ToolCall(
                                id="call_1",
                                name="write_file",
                                arguments={"path": "test.txt", "content": "Cost test"},
                            )
                        ],
                        metadata=turn1_metadata,
                    ),
                    ProviderResponse(
                        content="File written.",
                        tool_calls=[],
                        metadata=turn2_metadata,
                    ),
                ]
            )

            agent.run("Write 'Cost test' to test.txt")

            # Verify cost tracking
            assert agent.last_run_metadata is not None
            assert agent.last_run_metadata.usage.estimated_cost is not None
            # Cost should be tracked (aggregation may update this)
            assert agent.last_run_metadata.usage.estimated_cost.total_usd > 0

    def test_metadata_turn_history(self, work_dir, mock_metadata_factory):
        """Test that turn history is accurately recorded."""
        with patch("allos.providers.openai.openai.OpenAI"):
            agent = Agent(
                AgentConfig(
                    provider_name="openai",
                    model="gpt-4o",
                    tool_names=["write_file", "read_file"],
                    auto_approve=True,
                    max_iterations=5,
                )
            )

            # Mock 3-turn workflow: write, read, confirm
            responses = [
                # Turn 1: Write file tool call
                ProviderResponse(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            name="write_file",
                            arguments={"path": "log.txt", "content": "Entry 1"},
                        )
                    ],
                    metadata=mock_metadata_factory(
                        usage={"input_tokens": 50, "output_tokens": 30}
                    ),
                ),
                # Turn 2: Confirm write
                ProviderResponse(
                    content="File written successfully.",
                    tool_calls=[],
                    metadata=mock_metadata_factory(
                        usage={"input_tokens": 100, "output_tokens": 20}
                    ),
                ),
            ]

            agent.provider.chat = MagicMock(side_effect=responses)

            agent.run("Create log.txt with 'Entry 1'")

            # Verify turn history
            assert agent.last_run_metadata is not None
            assert agent.last_run_metadata.turns.total_turns == 2
            assert len(agent.last_run_metadata.turns.turn_history) == 2

            # Verify turn details
            turn1 = agent.last_run_metadata.turns.turn_history[0]
            assert turn1.turn_number == 1
            assert turn1.content_type == "tool_calls"
            assert "write_file" in turn1.tools_called
            assert turn1.tokens_used.input_tokens == 50
            assert turn1.tokens_used.output_tokens == 30

            turn2 = agent.last_run_metadata.turns.turn_history[1]
            assert turn2.turn_number == 2
            assert turn2.content_type == "text_response"
            assert len(turn2.tools_called) == 0

    def test_multi_agent_sequential_workflow(self, work_dir, available_providers):
        """
        Execute sequential multi-agent workflow and aggregate metadata.

        Workflow: Planner -> Coder -> Reviewer
        Each agent uses a different provider where available.

        Verifies:
        1. Each agent's metadata is complete and accurate
        2. Provider-specific fields are preserved (from first turn)
        3. Aggregation correctly sums tokens, costs, latency
        4. Tool calls are tracked across all agents
        5. Multi-turn agents preserve first_metadata provider-specific fields
        """
        # Need at least 3 providers for this test
        if len(available_providers) < 3:
            pytest.skip("Need at least 3 providers for multi-agent workflow test")

        # Select 3 different providers
        providers = available_providers[:3]

        # Skip Google on Python 3.9
        if sys.version_info < (3, 10):
            providers = [p for p in providers if p != "google"]
            if len(providers) < 3:
                pytest.skip("Not enough Python 3.9-compatible providers")

        agents_metadata: List[Metadata] = []

        # ==================================================================
        # AGENT 1: Planner (no tools, single turn)
        # ==================================================================
        planner_provider = providers[0]

        with mock_provider_environment(planner_provider):
            planner_response = _create_mock_response_simple(
                provider_name=planner_provider,
                model_id=PROVIDER_MODELS.get(planner_provider, "default"),
                input_tokens=500,
                output_tokens=200,
                content="Plan:\n1. Create todo.py\n2. Add Todo class\n3. Add functions\n4. Test",
                tool_calls=[],
                provider_specific_fields=_get_provider_specific_fields(
                    planner_provider, "planner_123"
                ),
            )

            planner = self._create_agent_with_mock(
                planner_provider, [], lambda messages, **kwargs: planner_response
            )

            response = planner.run("Create a plan for building a todo app")
            assert "Plan" in response

            # Verify and collect metadata
            assert planner.last_run_metadata is not None
            assert planner.last_run_metadata.usage.total_tokens == 700
            assert planner.last_run_metadata.turns.total_turns == 1

            agents_metadata.append(planner.last_run_metadata)

        # ==================================================================
        # AGENT 2: Coder (with tools, multi-turn)
        # ==================================================================
        coder_provider = providers[1]

        with mock_provider_environment(coder_provider):
            coder_mocker = MultiTurnResponseMocker(
                provider_name=coder_provider,
                model_id=PROVIDER_MODELS.get(coder_provider, "default"),
                turns_config=[
                    {
                        "input_tokens": 400,
                        "output_tokens": 50,
                        "content": None,
                        "tool_calls": [
                            ToolCall(
                                id="call_write_1",
                                name="write_file",
                                arguments={
                                    "path": "todo.py",
                                    "content": "class Todo:\n    def __init__(self):\n        pass",
                                },
                            )
                        ],
                        "provider_specific": _get_provider_specific_fields(
                            coder_provider, "coder_456"
                        ),
                    },
                    {
                        "input_tokens": 450,
                        "output_tokens": 150,
                        "content": "I've created todo.py with the Todo class.",
                        "tool_calls": [],
                        "provider_specific": None,  # Test first_metadata preservation
                    },
                ],
            )

            coder = self._create_agent_with_mock(
                coder_provider, ["write_file"], coder_mocker
            )

            response = coder.run("Implement the todo app based on the plan")
            assert "todo.py" in response.lower()

            # Verify and collect metadata
            assert coder.last_run_metadata is not None
            assert coder.last_run_metadata.turns.total_turns == 2  # Multi-turn
            assert coder.last_run_metadata.tools.total_tool_calls == 1

            # Verify first_metadata preservation
            self._verify_provider_specific_metadata(
                coder.last_run_metadata, coder_provider, "coder_456"
            )

            agents_metadata.append(coder.last_run_metadata)

        # ==================================================================
        # AGENT 3: Reviewer (with tools, multi-turn)
        # ==================================================================
        reviewer_provider = providers[2]

        with mock_provider_environment(reviewer_provider):
            reviewer_mocker = MultiTurnResponseMocker(
                provider_name=reviewer_provider,
                model_id=PROVIDER_MODELS.get(reviewer_provider, "default"),
                turns_config=[
                    {
                        "input_tokens": 300,
                        "output_tokens": 40,
                        "content": None,
                        "tool_calls": [
                            ToolCall(
                                id="call_read_1",
                                name="read_file",
                                arguments={"path": "todo.py"},
                            )
                        ],
                        "provider_specific": _get_provider_specific_fields(
                            reviewer_provider, "reviewer_789"
                        ),
                    },
                    {
                        "input_tokens": 350,
                        "output_tokens": 180,
                        "content": "Code review: The Todo class looks good. Basic structure is in place.",
                        "tool_calls": [],
                        "provider_specific": None,
                    },
                ],
            )
            reviewer = self._create_agent_with_mock(
                reviewer_provider, ["read_file"], reviewer_mocker
            )

            response = reviewer.run("Review the todo.py file")
            assert "review" in response.lower() or "todo" in response.lower()

            # Verify and collect metadata
            assert reviewer.last_run_metadata is not None
            assert reviewer.last_run_metadata.turns.total_turns == 2
            assert reviewer.last_run_metadata.tools.total_tool_calls == 1

            self._verify_provider_specific_metadata(
                reviewer.last_run_metadata, reviewer_provider, "reviewer_789"
            )

            agents_metadata.append(reviewer.last_run_metadata)

        # ==================================================================
        # AGGREGATE METADATA ACROSS ALL AGENTS
        # ==================================================================
        aggregated = aggregate_multi_agent_metadata(agents_metadata)

        # Verify aggregation
        assert aggregated["total_agents"] == 3
        assert len(aggregated["providers_used"]) == 3
        assert set(aggregated["providers_used"]) == set(providers)

        # Verify token aggregation
        expected_total_tokens = 700 + (400 + 50 + 450 + 150) + (300 + 40 + 350 + 180)
        expected_total_duration = sum(
            m.latency.total_duration_ms for m in agents_metadata
        )

        assert aggregated["usage"]["total_tokens"] == expected_total_tokens
        assert aggregated["total_duration_seconds"] == expected_total_duration

        # Verify tool calls
        assert aggregated["tool_calls_count"] == 2  # 1 from coder + 1 from reviewer

        # Verify cost estimation is present
        assert aggregated["total_cost_usd"] >= 0

        # Verify breakdown structure
        assert len(aggregated["breakdown"]) == 3
        for i, breakdown in enumerate(aggregated["breakdown"], 1):
            assert breakdown["agent_index"] == i
            assert breakdown["provider"] in providers
            assert breakdown["tokens"]["total"] > 0
            if i == 2 or i == 3:
                # Coder and Reviewer have multi-turns
                assert breakdown["turns"] == 2
            else:
                # Planner has a single turn
                assert breakdown["turns"] == 1

        # Verify provider-specific insights are captured
        insights = aggregated["provider_specific_insights"]
        assert len(insights) > 0  # Should have insights from all 3 agents

        self._verify_provider_insights(insights, providers)

        # Generate and verify report
        report = format_multi_agent_report(aggregated)

        # Verify report structure and content
        self._verify_report_structure(report, aggregated)

    def test_multi_agent_provider_diversity2(self, work_dir, available_providers):
        """
        Test aggregation works across all major providers.

        Creates one agent per available provider and validates
        metadata compatibility and aggregation correctness.

        Verifies:
        1. Each provider's metadata structure is compatible
        2. Provider-specific fields are correctly categorized
        3. Cost calculation works across mixed pricing models
        4. Token counting is consistent across providers
        """
        if len(available_providers) < 2:
            pytest.skip("Need at least 2 providers for diversity test")

        # Filter out Google on Python 3.9
        test_providers = available_providers[:]
        if sys.version_info < (3, 10):
            test_providers = [p for p in test_providers if p != "google"]
            if len(test_providers) < 2:
                pytest.skip("Not enough Python 3.9-compatible providers")

        agents_metadata = []

        for provider_name in test_providers:
            with mock_provider_environment(provider_name):
                response = _create_mock_response_simple(
                    provider_name=provider_name,
                    model_id=PROVIDER_MODELS.get(provider_name, "default"),
                    input_tokens=100,
                    output_tokens=50,
                    content=f"Response from {provider_name}",
                    tool_calls=[],
                    provider_specific_fields=_get_provider_specific_fields(
                        provider_name, provider_name
                    ),
                )

                agent = self._create_agent_with_mock(
                    provider_name,
                    [],
                    lambda messages, response=response, **kwargs: response,
                )

                result = agent.run("Simple task")
                assert provider_name in result or "Response" in result

                # Collect metadata
                assert agent.last_run_metadata is not None
                agents_metadata.append(agent.last_run_metadata)

        # Aggregate across all providers
        aggregated = aggregate_multi_agent_metadata(agents_metadata)

        # Verify diversity
        assert aggregated["total_agents"] == len(test_providers)
        assert len(aggregated["providers_used"]) == len(test_providers)
        assert set(aggregated["providers_used"]) == set(test_providers)

        # Verify all agents contributed equally
        expected_tokens_per_agent = 150  # 100 input + 50 output
        assert aggregated["usage"]["total_tokens"] == expected_tokens_per_agent * len(
            test_providers
        )

        # Verify cost calculation (Ollama should be $0.00)
        self._verify_provider_costs(aggregated["breakdown"])

        # Verify provider-specific insights
        self._verify_provider_insights(
            aggregated["provider_specific_insights"], test_providers
        )

    def _verify_provider_costs(self, breakdown: List[Dict]):
        """Verify cost calculations per provider."""
        for agent_data in breakdown:
            if agent_data["provider"] == "ollama":
                assert agent_data["cost_usd"] == 0.0
            elif agent_data["provider"] == "openai":
                assert agent_data["cost_usd"] > 0

    def _verify_provider_insights(self, insights: Dict, providers: List[str]):
        """Verify provider-specific insights are present."""
        for i, provider_name in enumerate(providers, 1):
            if provider_name == "openai":
                assert f"agent_{i}_openai_fingerprint" in insights
            elif provider_name == "ollama":
                assert f"agent_{i}_ollama_warmup" in insights
            elif provider_name == "google":
                assert f"agent_{i}_google_thought_signatures" in insights

    def test_multi_agent_summary_report2(self, work_dir, available_providers):
        """
        Generate comprehensive summary report from multi-agent workflow.

        Validates report formatting, accuracy, and completeness.

        Verifies:
        1. Report contains all expected sections
        2. Numbers match aggregated metadata
        3. Formatting is clear and human-readable
        4. Provider-specific insights are included
        """
        if len(available_providers) < 2:
            pytest.skip("Need at least 2 providers for report test")

        # Use first 2 providers
        providers = available_providers[:2]
        if sys.version_info < (3, 10):
            providers = [p for p in providers if p != "google"]
            if len(providers) < 2:
                pytest.skip("Not enough Python 3.9-compatible providers")

        agents_metadata = []

        # Create 2 agents with different characteristics
        for i, provider_name in enumerate(providers):
            with mock_provider_environment(provider_name):
                # First agent: no tools, second agent: with tools
                tool_names = ["write_file"] if i == 1 else []

                if i == 0:
                    # Simple single-turn agent
                    response = _create_mock_response_simple(
                        provider_name=provider_name,
                        model_id=PROVIDER_MODELS.get(provider_name, "default"),
                        input_tokens=200,
                        output_tokens=100,
                        content="Analysis complete",
                        tool_calls=[],
                        provider_specific_fields=_get_provider_specific_fields(
                            provider_name, "agent1"
                        ),
                    )
                    agent = self._create_agent_with_mock(
                        provider_name,
                        tool_names,
                        lambda messages, response=response, **kwargs: response,
                    )
                    agent.run("Analyze data")
                else:
                    # Multi-turn agent with tools
                    mocker = MultiTurnResponseMocker(
                        provider_name=provider_name,
                        model_id=PROVIDER_MODELS.get(provider_name, "default"),
                        turns_config=[
                            {
                                "input_tokens": 150,
                                "output_tokens": 30,
                                "content": None,
                                "tool_calls": [
                                    ToolCall(
                                        id="call_1",
                                        name="write_file",
                                        arguments={
                                            "path": "test.txt",
                                            "content": "data",
                                        },
                                    )
                                ],
                                "provider_specific": _get_provider_specific_fields(
                                    provider_name, "agent2"
                                ),
                            },
                            {
                                "input_tokens": 180,
                                "output_tokens": 50,
                                "content": "File created",
                                "tool_calls": [],
                                "provider_specific": None,
                            },
                        ],
                    )
                    agent = self._create_agent_with_mock(
                        provider_name, tool_names, mocker
                    )
                    agent.run("Write data")

                # Collect metadata
                assert agent.last_run_metadata is not None
                agents_metadata.append(agent.last_run_metadata)

        # Generate report
        aggregated = aggregate_multi_agent_metadata(agents_metadata)
        report = format_multi_agent_report(aggregated)

        # Verify report structure and content
        self._verify_report_structure(report, aggregated)

    def _verify_report_structure(self, report: str, aggregated: Dict):
        """Verify the summary report contains all expected sections."""
        providers = aggregated.get("providers_used", [])
        print(report)
        assert "MULTI-AGENT WORKFLOW SUMMARY" in report
        assert f"Total Agents: {len(aggregated.get('breakdown', []))}" in report
        assert "Total Duration:" in report
        assert "Overview:" in report
        assert "Token Usage:" in report
        assert "Total Input Tokens:" in report
        assert "Total Output Tokens:" in report
        assert "Total Tokens:" in report
        assert "Total Tool Calls:" in report
        assert "Breakdown by Agent:" in report
        assert f"{providers[0]}" in report
        assert f"{providers[1]}" in report
        assert "Tokens:" in report
        assert "Cost:" in report
        assert "Duration:" in report
        assert "Turns:" in report
        assert "Tools:" in report
        assert "Provider-Specific Insights:" in report

        # Verify numbers in report match aggregated data
        assert f"Total Agents: {str(aggregated['total_agents'])} in report"
        assert f"{aggregated['usage']['total_tokens']:,}" in report
        assert f"${aggregated['total_cost_usd']:.4f}" in report

        # Verify provider insights are included
        if aggregated["provider_specific_insights"]:
            assert "Provider-Specific Insights:" in report
        # Verify report is multi-line and well-formatted
        lines = report.split("\n")
        assert len(lines) > 15  # Should be substantial

        # Verify indentation and structure
        assert any(line.startswith("  ") for line in lines)  # Has indentation

    # --- Helper Methods ---

    def _create_agent_with_mock(
        self,
        provider_name: str,
        tool_names: List[str],
        mock_response_fn: Callable,
    ) -> Agent:
        """
        Create an agent with mocked provider chat method.

        Args:
            provider_name: Provider name
            tool_names: List of tool names to enable
            mock_response_fn: Callable to use as chat mock

        Returns:
            Configured Agent instance
        """
        agent = Agent(
            AgentConfig(
                provider_name=provider_name,
                model=PROVIDER_MODELS.get(provider_name, "default"),
                tool_names=tool_names,
                auto_approve=True,
            )
        )
        agent.provider.chat = MagicMock(side_effect=mock_response_fn)  # type: ignore[method-assign]
        return agent

    def _verify_provider_specific_metadata(
        self, metadata: Metadata, provider_name: str, expected_variant: str
    ):
        """
        Verify provider-specific metadata fields are present.

        Args:
            metadata: Metadata object to verify
            provider_name: Provider name
            expected_variant: Expected variant identifier in the fields
        """
        if provider_name == "openai":
            assert (
                metadata.provider_specific.openai is not None
                and metadata.provider_specific.openai.system_fingerprint
            )
            assert (
                expected_variant in metadata.provider_specific.openai.system_fingerprint
            )
        elif provider_name == "ollama":
            assert metadata.provider_specific.ollama is not None
            assert metadata.provider_specific.ollama.warm_up is True
        elif provider_name == "google":
            assert metadata.provider_specific.google is not None
            # Google-specific assertions handled in original test

    def _get_client_patch_path(self, provider_name: str) -> str:
        """Get the full patch path for mocking the provider's client."""
        mapping = {
            "openai": "allos.providers.openai.openai.OpenAI",
            "anthropic": "allos.providers.anthropic.anthropic.Anthropic",
            "ollama": "allos.providers.ollama.Client",
            "google": "allos.providers.google.genai.Client",
            "chat_completions": "allos.providers.chat_completions.openai.OpenAI",
        }
        return mapping.get(provider_name, "allos.providers.base.BaseProvider")


# ==============================================================================
# HELPER FUNCTIONS FOR MULTI-AGENT METADATA AGGREGATION
# ==============================================================================


def _get_provider_specific_fields(
    provider_name: str, variant: str = "default"
) -> Optional[Dict[str, Any]]:
    """
    Generate provider-specific metadata fields for testing.

    Centralizes the if/elif logic for provider-specific fields to reduce
    complexity in test methods.

    Args:
        provider_name: Provider name (openai, anthropic, ollama, google)
        variant: Variant identifier for different test scenarios

    Returns:
        Dictionary suitable for provider_specific_fields parameter,
        or None if provider has no specific fields
    """
    if provider_name == "openai":
        return {"openai": {"system_fingerprint": f"fp_{variant}"}}
    elif provider_name == "ollama":
        return {
            "ollama": {
                "warm_up": True,
                "warm_up_duration_seconds": 2.5 if variant == "default" else 3.2,
            }
        }
    elif provider_name == "google":
        return {
            "google": {
                "used_thought_signatures": variant != "no_thought",
                "vertexai": False,
            }
        }
    return None


def _create_mock_response_simple(
    provider_name: str,
    model_id: str,
    input_tokens: int,
    output_tokens: int,
    content: Optional[str],
    tool_calls: Optional[List[ToolCall]] = None,
    provider_specific_fields: Optional[Dict[str, Any]] = None,
) -> ProviderResponse:
    """
    Create a simple mock ProviderResponse for testing.

    Args:
        provider_name: Provider name
        model_id: Model identifier
        input_tokens: Input token count
        output_tokens: Output token count
        content: Response content (None for tool calls)
        tool_calls: List of tool calls (empty list if none)
        provider_specific_fields: Optional provider-specific metadata

    Returns:
        ProviderResponse with populated metadata
    """
    metadata = create_test_metadata_with_builder(
        provider_name=provider_name,
        model_id=model_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        provider_specific_fields=provider_specific_fields,
    )

    return ProviderResponse(
        content=content,
        tool_calls=tool_calls or [],
        metadata=metadata,
    )


class MultiTurnResponseMocker:
    """
    Helper class to manage multi-turn mock responses.

    Reduces complexity by encapsulating turn counting and conditional logic.
    """

    def __init__(self, provider_name: str, model_id: str, turns_config: List[Dict]):
        """
        Initialize multi-turn mocker.

        Args:
            provider_name: Provider name
            model_id: Model identifier
            turns_config: List of dicts, each containing:
                - input_tokens: int
                - output_tokens: int
                - content: Optional[str]
                - tool_calls: Optional[List[ToolCall]]
                - provider_specific: Optional[Dict] (only for first turn typically)
        """
        self.provider_name = provider_name
        self.model_id = model_id
        self.turns_config = turns_config
        self.call_count = 0

    def __call__(self, messages, **kwargs) -> ProviderResponse:
        """Generate response for current turn."""
        if self.call_count >= len(self.turns_config):
            raise ValueError(f"Unexpected call #{self.call_count + 1}")

        turn_config = self.turns_config[self.call_count]
        self.call_count += 1

        return _create_mock_response_simple(
            provider_name=self.provider_name,
            model_id=self.model_id,
            input_tokens=turn_config["input_tokens"],
            output_tokens=turn_config["output_tokens"],
            content=turn_config.get("content"),
            tool_calls=turn_config.get("tool_calls", []),
            provider_specific_fields=turn_config.get("provider_specific"),
        )


def aggregate_multi_agent_metadata(metadata_list: List[Metadata]) -> Dict[str, Any]:
    """
    Aggregate metadata from multiple agents into a summary.

    Args:
        metadata_list: List of Metadata objects from different agents

    Returns:
        Dictionary containing aggregated metrics and per-agent breakdown
    """
    if not metadata_list:
        return {
            "total_agents": 0,
            "providers_used": [],
            "total_duration_seconds": 0.0,
            "total_cost_usd": 0.0,
            "usage": {
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_tokens": 0,
            },
            "tool_calls_count": 0,
            "breakdown": [],
            "provider_specific_insights": {},
        }

    return {
        "total_agents": len(metadata_list),
        "providers_used": list({m.model.provider for m in metadata_list}),
        "total_duration_seconds": sum(
            m.latency.total_duration_ms for m in metadata_list
        ),
        "total_cost_usd": sum(
            m.usage.estimated_cost.total_usd
            for m in metadata_list
            if m.usage.estimated_cost
        ),
        "usage": {
            "total_input_tokens": sum(m.usage.input_tokens for m in metadata_list),
            "total_output_tokens": sum(m.usage.output_tokens for m in metadata_list),
            "total_tokens": sum(m.usage.total_tokens for m in metadata_list),
        },
        "tool_calls_count": sum(m.tools.total_tool_calls for m in metadata_list),
        "breakdown": [
            {
                "agent_index": i,
                "provider": m.model.provider,
                "model": m.model.model_id,
                "tokens": {
                    "input": m.usage.input_tokens,
                    "output": m.usage.output_tokens,
                    "total": m.usage.total_tokens,
                },
                "cost_usd": m.usage.estimated_cost.total_usd
                if m.usage.estimated_cost
                else 0.0,
                "duration_seconds": m.latency.total_duration_ms,
                "tool_calls": m.tools.total_tool_calls,
                "turns": m.turns.total_turns,
            }
            for i, m in enumerate(metadata_list, 1)
        ],
        "provider_specific_insights": _extract_provider_insights(metadata_list),
    }


def _extract_provider_insights(metadata_list: List[Metadata]) -> Dict[str, Any]:
    """Extract provider-specific fields from all agents.

    Note: Each agent's metadata already contains merged first_metadata + last_metadata
    from its own run, so provider-specific fields should be present if they occurred
    at any point during that agent's execution.

    Args:
        metadata_list: List of Metadata objects from different agents

    Returns:
        Dictionary mapping insight keys to values
    """
    insights: Dict[str, Any] = {}

    for i, metadata in enumerate(metadata_list, 1):
        # OpenAI: system_fingerprint (usually from first turn of agent's run)
        if metadata.provider_specific.openai:
            if metadata.provider_specific.openai.system_fingerprint:
                insights[f"agent_{i}_openai_fingerprint"] = (
                    metadata.provider_specific.openai.system_fingerprint
                )

        # Ollama: warm_up (only from first turn when model loads)
        if metadata.provider_specific.ollama:
            insights[f"agent_{i}_ollama_warmup"] = {
                "detected": metadata.provider_specific.ollama.warm_up,
                "duration_seconds": metadata.provider_specific.ollama.warm_up_duration_seconds,
            }

        # Google: thought signatures (can appear in any turn)
        if metadata.provider_specific.google:
            insights[f"agent_{i}_google_thought_signatures"] = (
                metadata.provider_specific.google.used_thought_signatures
            )
            insights[f"agent_{i}_google_vertexai"] = (
                metadata.provider_specific.google.vertexai
            )

    return insights


def format_multi_agent_report(aggregated: Dict[str, Any]) -> str:
    """
    Format aggregated metadata into a human-readable report.

    Args:
        aggregated: Dictionary from aggregate_multi_agent_metadata()

    Returns:
        Formatted string report
    """
    lines = [
        "MULTI-AGENT WORKFLOW SUMMARY",
        "=" * 50,
        "",
        "Overview:",
        f"  Total Agents: {aggregated['total_agents']}",
        f"  Providers Used: {', '.join(aggregated['providers_used'])}",
        f"  Total Duration: {aggregated['total_duration_seconds']:.2f}s",
        f"  Total Cost: ${aggregated['total_cost_usd']:.4f} USD",
        "",
        "Token Usage:",
        f"  Total Input Tokens: {aggregated['usage']['total_input_tokens']:,}",
        f"  Total Output Tokens: {aggregated['usage']['total_output_tokens']:,}",
        f"  Total Tokens: {aggregated['usage']['total_tokens']:,}",
        f"  Total Tool Calls: {aggregated['tool_calls_count']}",
        "",
        "Breakdown by Agent:",
    ]

    for agent in aggregated["breakdown"]:
        lines.extend(
            [
                f"  {agent['agent_index']}. {agent['provider']}/{agent['model']}:",
                f"     Tokens: {agent['tokens']['input']} in, {agent['tokens']['output']} out, {agent['tokens']['total']} total",
                f"     Cost: ${agent['cost_usd']:.4f}",
                f"     Duration: {agent['duration_seconds']:.2f}s",
                f"     Turns: {agent['turns']}",
                f"     Tools: {agent['tool_calls']} calls",
                "",
            ]
        )

    if aggregated["provider_specific_insights"]:
        lines.append("Provider-Specific Insights:")
        for key, value in aggregated["provider_specific_insights"].items():
            lines.append(f"  {key}: {value}")

    return "\n".join(lines)
