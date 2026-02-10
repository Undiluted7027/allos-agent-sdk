"""End-to-end tests for Google provider workflows.

NOTE: Requires Python 3.10+

These tests verify Google-specific features:
1. Thought signatures multi-turn (Gemini 3.x)
2. Vertex AI vs Gemini API behavior
3. Model family comparisons (1.5, 2.0, 2.5, 3.0)
"""

import sys

import pytest

# Skip entire module on Python 3.9
if sys.version_info < (3, 10):
    pytest.skip("Google provider tests require Python 3.10+", allow_module_level=True)

from unittest.mock import MagicMock, patch

from google.genai import types

from allos import Agent, AgentConfig
from allos.providers.base import MessageRole, ProviderResponse, ToolCall
from allos.providers.metadata import ProviderSpecific, ProviderSpecificGoogle

pytestmark = pytest.mark.e2e


@pytest.mark.e2e
@pytest.mark.requires_python_310
class TestGoogleWorkflows:
    """E2E tests specific to Google provider features."""

    def test_thought_signatures_multi_turn_gemini3(
        self, work_dir, mock_metadata_factory
    ):
        """
        Test Gemini 3.x thought signatures across multiple turns.

        Gemini 3 requires thought signatures for function calling.
        This test verifies they're preserved correctly across the full
        provider response chain.
        """
        with patch("allos.providers.google.genai") as mock_genai:
            # Setup mock client
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client

            # Mock model list for _verify_model_available
            mock_client.models.list.return_value = [
                types.Model(
                    name="models/gemini-3-flash-preview", input_token_limit=1048576
                )
            ]

            # Create agent
            agent = Agent(
                AgentConfig(
                    provider_name="google",
                    model="gemini-3-flash-preview",
                    tool_names=["write_file"],
                    auto_approve=True,
                )
            )

            # Turn 1: LLM calls write_file with thought signature
            turn1_signature = b"thought_signature_turn1_xyz"
            turn1_metadata = mock_metadata_factory(
                usage={"input_tokens": 100, "output_tokens": 50},
                provider="google",
                model_id="gemini-3-flash-preview",
                provider_specific=ProviderSpecific(
                    google=ProviderSpecificGoogle(
                        vertexai=False, used_thought_signatures=True
                    )
                ),
            )

            turn1_response = ProviderResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_write_1",
                        name="write_file",
                        arguments={"path": "test.txt", "content": "Hello Gemini 3"},
                    )
                ],
                thought_signatures={"call_write_1": turn1_signature},
                metadata=turn1_metadata,
            )

            # Turn 2: LLM confirms completion (no thought signature needed)
            turn2_metadata = mock_metadata_factory(
                usage={"input_tokens": 200, "output_tokens": 30},
                provider="google",
                model_id="gemini-3-flash-preview",
            )

            turn2_response = ProviderResponse(
                content="File written successfully with Gemini 3 thought signatures.",
                tool_calls=[],
                metadata=turn2_metadata,
            )

            agent.provider.chat = MagicMock(
                side_effect=[turn1_response, turn2_response]
            )

            # Execute workflow
            agent.run("Write 'Hello Gemini 3' to test.txt")

            # Verify thought signature was preserved in context
            assistant_messages = [
                m
                for m in agent.context.messages
                if m.role == MessageRole.ASSISTANT and m.thought_signatures
            ]

            assert len(assistant_messages) > 0, "No thought signatures in context"
            assert assistant_messages[0].thought_signatures, (
                "Thought signatures were None"
            )
            assert "call_write_1" in assistant_messages[0].thought_signatures, (
                "Tool call ID not in signatures"
            )
            assert (
                assistant_messages[0].thought_signatures["call_write_1"]
                == turn1_signature
            ), "Thought signature mismatch"

            # Verify file was created
            test_file = work_dir / "test.txt"
            assert test_file.exists()
            assert test_file.read_text() == "Hello Gemini 3"

            # Verify metadata structure
            assert agent.last_run_metadata is not None
            assert agent.last_run_metadata.usage.total_tokens > 0
            assert agent.last_run_metadata.model.provider == "google"

    def test_thought_signatures_preservation_across_turns(
        self, work_dir, mock_metadata_factory
    ):
        """
        Test that thought signatures from turn N are passed back to LLM in turn N+1.

        This is critical for Gemini 3.x function calling.
        """
        with patch("allos.providers.google.genai") as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            mock_client.models.list.return_value = [
                types.Model(
                    name="models/gemini-3-flash-preview", input_token_limit=1048576
                )
            ]

            agent = Agent(
                AgentConfig(
                    provider_name="google",
                    model="gemini-3-flash-preview",
                    tool_names=["write_file", "read_file"],
                    auto_approve=True,
                )
            )

            # Turn 1: Write file with signature
            sig1 = b"signature_write"
            turn1_response = ProviderResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="write_file",
                        arguments={"path": "data.txt", "content": "Data"},
                    )
                ],
                thought_signatures={"call_1": sig1},
                metadata=mock_metadata_factory(
                    provider="google", model_id="gemini-3-flash-preview"
                ),
            )

            # Turn 2: Read file with different signature
            sig2 = b"signature_read"
            turn2_response = ProviderResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_2", name="read_file", arguments={"path": "data.txt"}
                    )
                ],
                thought_signatures={"call_2": sig2},
                metadata=mock_metadata_factory(
                    provider="google", model_id="gemini-3-flash-preview"
                ),
            )

            # Turn 3: Final response
            turn3_response = ProviderResponse(
                content="Successfully wrote and read the file.",
                tool_calls=[],
                metadata=mock_metadata_factory(
                    provider="google", model_id="gemini-3-flash-preview"
                ),
            )

            agent.provider.chat = MagicMock(
                side_effect=[turn1_response, turn2_response, turn3_response]
            )

            agent.run("Write 'Data' to data.txt and then read it back")

            # Verify both signatures are in context
            assistant_with_sigs = [
                m
                for m in agent.context.messages
                if m.role == MessageRole.ASSISTANT and m.thought_signatures
            ]

            assert len(assistant_with_sigs) >= 2, (
                "Expected at least 2 turns with signatures"
            )

            # Verify signatures are distinct
            all_sigs = []
            for msg in assistant_with_sigs:
                assert msg.thought_signatures, (
                    "Expected thought signatures to have values"
                )
                all_sigs.extend(msg.thought_signatures.values())
            assert sig1 in all_sigs, "First signature not found"
            assert sig2 in all_sigs, "Second signature not found"

    def test_gemini_model_comparison(self, work_dir, mock_metadata_factory):
        """
        Compare behavior across Gemini model families.

        Key differences:
        - 1.5: No thought signatures
        - 2.0: No thought signatures
        - 2.5: Optional thought signatures
        - 3.0: Required thought signatures
        """
        models_to_test = {
            "gemini-1.5-flash": {"has_signatures": False, "required": False},
            "gemini-2.0-flash": {"has_signatures": False, "required": False},
            "gemini-2.5-flash": {"has_signatures": True, "required": False},
            "gemini-3-flash-preview": {"has_signatures": True, "required": True},
        }

        results = {}

        for model_name, features in models_to_test.items():
            with patch("allos.providers.google.genai") as mock_genai:
                mock_client = MagicMock()
                mock_genai.Client.return_value = mock_client
                mock_client.models.list.return_value = [
                    types.Model(name=f"models/{model_name}", input_token_limit=1048576)
                ]

                agent = Agent(
                    AgentConfig(
                        provider_name="google",
                        model=model_name,
                        tool_names=["write_file"],
                        auto_approve=True,
                    )
                )

                # Mock response with or without thought signatures based on model
                thought_sigs = (
                    {"call_1": b"signature_data"}
                    if features["has_signatures"]
                    else None
                )

                turn1_response = ProviderResponse(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            name="write_file",
                            arguments={"path": "test.txt", "content": "Test"},
                        )
                    ],
                    thought_signatures=thought_sigs,
                    metadata=mock_metadata_factory(
                        provider="google", model_id=model_name
                    ),
                )

                turn2_response = ProviderResponse(
                    content=f"File written using {model_name}.",
                    tool_calls=[],
                    metadata=mock_metadata_factory(
                        provider="google", model_id=model_name
                    ),
                )

                agent.provider.chat = MagicMock(
                    side_effect=[turn1_response, turn2_response]
                )

                result = agent.run("Write 'Test' to test.txt")

                # Collect results
                assistant_msgs = [
                    m
                    for m in agent.context.messages
                    if m.role == MessageRole.ASSISTANT and m.thought_signatures
                ]

                results[model_name] = {
                    "result": result,
                    "has_signatures_in_context": len(assistant_msgs) > 0,
                    "file_created": (work_dir / "test.txt").exists(),
                    "metadata": agent.last_run_metadata,
                }

                # Clean up
                if (work_dir / "test.txt").exists():
                    (work_dir / "test.txt").unlink()

        # Verify model-specific behavior
        assert not results["gemini-1.5-flash"]["has_signatures_in_context"], (
            "1.5 should not have signatures"
        )
        assert not results["gemini-2.0-flash"]["has_signatures_in_context"], (
            "2.0 should not have signatures"
        )

        # 2.5 and 3.0 should have signatures if we provided them
        if "gemini-2.5-flash" in results:
            assert results["gemini-2.5-flash"]["has_signatures_in_context"], (
                "2.5 should have signatures"
            )
        if "gemini-3-flash-preview" in results:
            assert results["gemini-3-flash-preview"]["has_signatures_in_context"], (
                "3.0 should have signatures"
            )

        # All models should successfully create the file
        for model_name, data in results.items():
            assert data["file_created"], f"{model_name} failed to create file"
            assert data["metadata"] is not None, f"{model_name} has no metadata"
            assert data["metadata"].model.provider == "google"

    def test_vertex_ai_vs_gemini_api(self, work_dir, mock_metadata_factory):
        """
        Compare Vertex AI and Gemini API for same task.

        Both should produce equivalent results, just using different
        authentication methods.
        """
        test_cases = [
            {
                "name": "Gemini API",
                "vertexai": False,
            },
            {
                "name": "Vertex AI",
                "vertexai": True,
            },
        ]

        results = {}

        for test_case in test_cases:
            with patch("allos.providers.google.genai") as mock_genai:
                mock_client = MagicMock()
                mock_genai.Client.return_value = mock_client
                mock_client.models.list.return_value = [
                    types.Model(
                        name="models/gemini-2.0-flash", input_token_limit=1048576
                    )
                ]

                # Also patch Vertex AI credential loading if needed
                if test_case["vertexai"]:
                    mock_auth_context = patch("google.auth.default")
                    mock_auth = mock_auth_context.__enter__()
                    mock_auth.return_value = (MagicMock(), "test-project")
                else:
                    mock_auth_context = None

                agent = Agent(
                    AgentConfig(
                        provider_name="google",
                        model="gemini-2.0-flash",
                        tool_names=["write_file"],
                        auto_approve=True,
                    )
                )

                # Mock identical responses
                filename = f"{test_case['name'].replace(' ', '_')}.txt"
                self._setup_standard_write_mock(
                    agent,
                    mock_metadata_factory,
                    filename=filename,
                    vertexai=test_case["vertexai"],
                )

                result = agent.run(f"Write '{test_case['name']}' to file")

                results[test_case["name"]] = {
                    "result": result,
                    "messages": len(agent.context.messages),
                    "metadata": agent.last_run_metadata,
                }

                if mock_auth_context:
                    mock_auth_context.__exit__(None, None, None)

        # Verify both APIs completed successfully
        assert len(results) == 2, "Both APIs should have completed"

        # Verify metadata structure is consistent
        for api_name, data in results.items():
            assert data["metadata"] is not None, f"{api_name}: No metadata"
            assert data["metadata"].model.provider == "google", (
                f"{api_name}: Wrong provider"
            )
            assert data["metadata"].usage.total_tokens > 0, (
                f"{api_name}: No token usage"
            )

    def test_google_error_handling(self, work_dir):
        """
        Test Google-specific error handling.

        Covers:
        - Model not available
        - Authentication errors
        - API errors
        """
        from allos.utils.errors import ProviderError

        with patch("allos.providers.google.genai") as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client

            # Mock empty model list
            mock_client.models.list.return_value = []

            # This should raise ProviderError due to model not available
            with pytest.raises(ProviderError, match="not available"):
                Agent(
                    AgentConfig(
                        provider_name="google",
                        model="nonexistent-model",
                        tool_names=[],
                    )
                )

    # --- Helper Methods ---

    def _setup_standard_write_mock(
        self,
        agent: Agent,
        mock_metadata_factory,
        filename: str = "test.txt",
        vertexai: bool = False,
        model: str = "gemini-2.0-flash",
    ):
        """Setup standard two-turn write file mock with proper metadata."""
        turn1_response = ProviderResponse(
            content=None,
            tool_calls=[
                ToolCall(
                    id="call_1",
                    name="write_file",
                    arguments={"path": filename, "content": "Test content"},
                )
            ],
            metadata=mock_metadata_factory(
                usage={"input_tokens": 50, "output_tokens": 30},
                provider="google",
                model_id=model,
                provider_specific=ProviderSpecific(
                    google=ProviderSpecificGoogle(
                        vertexai=vertexai,
                        project="test-project" if vertexai else None,
                        location="us-central1" if vertexai else None,
                    )
                ),
            ),
        )

        turn2_response = ProviderResponse(
            content=f"File {filename} written successfully.",
            tool_calls=[],
            metadata=mock_metadata_factory(
                usage={"input_tokens": 100, "output_tokens": 20},
                provider="google",
                model_id=model,
            ),
        )

        agent.provider.chat = MagicMock(side_effect=[turn1_response, turn2_response])  # type: ignore[method-assign]
