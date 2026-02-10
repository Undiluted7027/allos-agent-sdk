# tests/e2e/test_cross_provider_workflows.py

"""End-to-end tests for cross-provider workflows.

Tests that verify the same task works consistently across all providers,
including metadata tracking, provider switching, and failover scenarios.

Test Coverage:
1. Same Task, All Providers - Execute identical tasks across all providers
2. Provider Switching Mid-Conversation - Verify context preservation when switching
3. Provider Fallback on Failure - Test resilience with automatic failback
4. Google Thought Signatures - Test Google-specific features (Python 3.10+)

Note: Google provider tests require Python 3.10+ and are automatically skipped on Python 3.9.
All providers are mocked at the client level, so no real API calls are made.
"""

import sys
from unittest.mock import MagicMock

import pytest

from allos import Agent, AgentConfig
from allos.providers.base import MessageRole, ProviderResponse, ToolCall
from allos.utils.errors import AllosError
from tests.conftest import PROVIDER_MODELS
from tests.e2e.conftest import (
    mock_provider_environment,
)

pytestmark = pytest.mark.e2e


@pytest.mark.e2e
class TestCrossProviderWorkflows:
    """Test identical tasks across all available providers."""

    def test_same_task_all_providers(
        self, available_providers, work_dir, mock_metadata_factory
    ):
        """Execute identical file write task across all providers and compare metadata."""
        task_prompt = "Write 'Hello from Allos' to greeting.txt"
        expected_content = "Hello from Allos"
        results = {}

        for provider_name in available_providers:
            # Skip Google on Python 3.9
            if provider_name == "google" and sys.version_info < (3, 10):
                continue

            with mock_provider_environment(provider_name):
                model_name = PROVIDER_MODELS.get(provider_name, "default")

                # Create agent
                agent = Agent(
                    AgentConfig(
                        provider_name=provider_name,
                        model=model_name,
                        tool_names=["write_file"],
                        auto_approve=True,
                    )
                )

                # Setup chat mock
                self._setup_write_file_mock(agent, provider_name, mock_metadata_factory)

                # Execute
                result = agent.run(task_prompt)

                # Collect results
                results[provider_name] = {
                    "result": result,
                    "metadata": agent.last_run_metadata,
                    "messages": len(agent.context.messages),
                }

                # Verify and cleanup
                test_file = work_dir / "greeting.txt"
                assert test_file.exists(), f"{provider_name}: File was not created"
                assert test_file.read_text() == expected_content, (
                    f"{provider_name}: File content mismatch"
                )
                test_file.unlink()

        # Verify all providers completed successfully
        assert len(results) == len(available_providers), (
            "Not all providers completed the task"
        )
        self._validate_metadata_consistency(results)

    def test_provider_switching_mid_conversation(
        self, work_dir, mock_metadata_factory, available_providers
    ):
        """Test switching providers mid-conversation maintains context."""
        if len(available_providers) < 2:
            pytest.skip("Need at least 2 providers for switching test")

        provider1, provider2 = available_providers[0], available_providers[1]

        # Check Python version requirements for Google
        if sys.version_info < (3, 10) and "google" in [provider1, provider2]:
            pytest.skip("Google provider requires Python 3.10+")

        session_file = work_dir / "session.json"
        test_file = work_dir / "language.txt"
        test_content = "Python is great for AI"
        initial_message_count = 0

        # Phase 1: Start with first provider
        with mock_provider_environment(provider1):
            agent1 = Agent(
                AgentConfig(
                    provider_name=provider1,
                    model=PROVIDER_MODELS.get(provider1, "default"),
                    tool_names=["write_file", "read_file"],
                    auto_approve=True,
                )
            )

            self._setup_write_file_mock(
                agent1,
                provider1,
                mock_metadata_factory,
                filename="language.txt",
                content=test_content,
            )

            agent1.run(f"Write '{test_content}' to language.txt")

            assert test_file.exists()
            assert test_file.read_text() == test_content

            agent1.save_session(str(session_file))
            assert session_file.exists()
            initial_message_count = len(agent1.context.messages)

        # Phase 2: Load session and switch to second provider
        with mock_provider_environment(provider2):
            # Load session
            agent2 = Agent.load_session(str(session_file))

            # Switch provider manually
            self._switch_agent_provider(agent2, provider2)

            # Mock provider to read file
            self._setup_read_file_mock(
                agent2,
                provider2,
                mock_metadata_factory,
                filename="language.txt",
                content=test_content,
            )

            agent2.run("Read language.txt and tell me what it says")

            # Verify context preservation
            assert len(agent2.context.messages) > initial_message_count, (
                "Context did not grow after second turn"
            )
            self._verify_conversation_history(agent2)

    def test_provider_fallback_on_failure(
        self, work_dir, mock_metadata_factory, available_providers
    ):
        """Test provider fallback when primary provider fails."""
        if len(available_providers) < 2:
            pytest.skip("Need at least 2 providers for fallback test")

        primary, backup = available_providers[0], available_providers[1]

        if sys.version_info < (3, 10) and "google" in [primary, backup]:
            pytest.skip("Google provider requires Python 3.10+")

        task_prompt = "Write 'Resilience test' to fallback.txt"

        # Attempt with primary provider (simulated failure)
        with mock_provider_environment(primary):
            agent_primary = Agent(
                AgentConfig(
                    provider_name=primary,
                    model=PROVIDER_MODELS.get(primary, "default"),
                    tool_names=["write_file"],
                    auto_approve=True,
                )
            )

            def failing_chat(*args, **kwargs):
                raise AllosError(f"Simulated {primary} provider failure")

            agent_primary.provider.chat = failing_chat

            with pytest.raises(AllosError, match="Simulated.*provider failure"):
                agent_primary.run(task_prompt)

        # Fallback to backup provider
        with mock_provider_environment(backup):
            agent_backup = Agent(
                AgentConfig(
                    provider_name=backup,
                    model=PROVIDER_MODELS.get(backup, "default"),
                    tool_names=["write_file"],
                    auto_approve=True,
                )
            )

            self._setup_write_file_mock(
                agent_backup,
                backup,
                mock_metadata_factory,
                filename="fallback.txt",
                content="Resilience test",
            )

            agent_backup.run(task_prompt)

            # Verify task completed
            test_file = work_dir / "fallback.txt"
            assert test_file.exists()
            assert test_file.read_text() == "Resilience test"
            assert agent_backup.last_run_metadata is not None

            # Check metadata provider match (handling chat_completions/openai mapping)
            actual_provider = agent_backup.last_run_metadata.model.provider
            assert actual_provider == backup or (
                backup == "chat_completions" and actual_provider == "openai"
            )

    @pytest.mark.skipif(
        sys.version_info < (3, 10), reason="Google requires Python 3.10+"
    )
    def test_google_thought_signatures_preservation(
        self, work_dir, mock_metadata_factory
    ):
        """Test Google Gemini thought signatures persist across turns."""
        if "google" not in PROVIDER_MODELS:
            pytest.skip("Google provider not available in configuration")

        with mock_provider_environment("google"):
            agent = Agent(
                AgentConfig(
                    provider_name="google",
                    model=PROVIDER_MODELS.get("google", "gemini-2.5-flash-lite"),
                    tool_names=["write_file"],
                    auto_approve=True,
                )
            )

            mock_signatures = {"call_1": b"mock_signature_data"}

            def chat_with_signatures(messages, **kwargs):
                return ProviderResponse(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            name="write_file",
                            arguments={"path": "test.txt", "content": "Test"},
                        )
                    ],
                    thought_signatures=mock_signatures,
                    metadata=mock_metadata_factory(),
                )

            agent.provider.chat = MagicMock(
                side_effect=[
                    chat_with_signatures([], tools=[]),
                    ProviderResponse(
                        content="File written successfully",
                        tool_calls=[],
                        metadata=mock_metadata_factory(),
                    ),
                ]
            )

            agent.run("Write 'Test' to test.txt")

            assistant_messages = [
                m
                for m in agent.context.messages
                if m.role == MessageRole.ASSISTANT and m.thought_signatures
            ]
            assert len(assistant_messages) > 0, "No thought signatures found in context"
            assert assistant_messages[0].thought_signatures == mock_signatures

    # --- Helper Methods ---

    def _switch_agent_provider(self, agent: Agent, new_provider_name: str):
        """Helper to manually switch an agent's provider and re-initialize it."""
        from allos.providers import ProviderRegistry

        model_name = PROVIDER_MODELS.get(new_provider_name, "default")
        agent.config.provider_name = new_provider_name
        agent.config.model = model_name

        agent.provider = ProviderRegistry.get_provider(
            agent.config.provider_name, model=agent.config.model
        )

    def _validate_metadata_consistency(self, results):
        """Validate metadata consistency across provider results."""
        for provider_name, data in results.items():
            metadata = data["metadata"]
            assert metadata is not None, f"{provider_name}: No metadata recorded"
            assert metadata.usage.total_tokens > 0, (
                f"{provider_name}: No tokens recorded"
            )
            assert metadata.model.provider == provider_name or (
                provider_name == "chat_completions"
                and metadata.model.provider == "openai"
            ), f"{provider_name}: Provider mismatch in metadata"

    def _verify_conversation_history(self, agent: Agent):
        """Verify the conversation history structure."""
        user_messages = [
            m for m in agent.context.messages if m.role == MessageRole.USER
        ]
        assert len(user_messages) >= 2, "Expected at least 2 user messages"

        assistant_messages = [
            m for m in agent.context.messages if m.role == MessageRole.ASSISTANT
        ]
        assert len(assistant_messages) >= 2, "Expected at least 2 assistant messages"

    def _get_client_patch_path(self, provider_name: str) -> str:
        """Get the full patch path for mocking the provider's client."""
        mapping = {
            "openai": "allos.providers.openai.openai.OpenAI",
            "anthropic": "allos.providers.anthropic.anthropic.Anthropic",
            "ollama": "allos.providers.ollama.ollama.Client",
            "google": "allos.providers.google.genai.Client",
            "chat_completions": "allos.providers.chat_completions.openai.OpenAI",
        }
        return mapping.get(provider_name, "allos.providers.base.BaseProvider")

    def _setup_write_file_mock(
        self,
        agent: Agent,
        provider_name: str,
        mock_metadata_factory,
        filename: str = "greeting.txt",
        content: str = "Hello from Allos",
    ):
        """Setup mock responses for a write_file task (2 turns)."""
        turn1_metadata = mock_metadata_factory(
            provider=provider_name, usage={"input_tokens": 50, "output_tokens": 30}
        )
        turn2_metadata = mock_metadata_factory(
            provider=provider_name, usage={"input_tokens": 100, "output_tokens": 20}
        )

        turn1_response = ProviderResponse(
            content=None,
            tool_calls=[
                ToolCall(
                    id="call_write_1",
                    name="write_file",
                    arguments={"path": filename, "content": content},
                )
            ],
            metadata=turn1_metadata,
        )

        turn2_response = ProviderResponse(
            content=f"I've successfully written the content to {filename}.",
            tool_calls=[],
            metadata=turn2_metadata,
        )

        agent.provider.chat = MagicMock(side_effect=[turn1_response, turn2_response])  # type: ignore[method-assign]

    def _setup_read_file_mock(
        self,
        agent: Agent,
        provider_name: str,
        mock_metadata_factory,
        filename: str = "language.txt",
        content: str = "Python is great for AI",
    ):
        """Setup mock responses for a read_file task (2 turns)."""
        turn1_metadata = mock_metadata_factory(
            provider=provider_name, usage={"input_tokens": 60, "output_tokens": 25}
        )
        turn2_metadata = mock_metadata_factory(
            provider=provider_name, usage={"input_tokens": 150, "output_tokens": 40}
        )

        turn1_response = ProviderResponse(
            content=None,
            tool_calls=[
                ToolCall(
                    id="call_read_1", name="read_file", arguments={"path": filename}
                )
            ],
            metadata=turn1_metadata,
        )

        turn2_response = ProviderResponse(
            content=f"The file {filename} contains: '{content}'",
            tool_calls=[],
            metadata=turn2_metadata,
        )

        agent.provider.chat = MagicMock(side_effect=[turn1_response, turn2_response])  # type: ignore[method-assign]
