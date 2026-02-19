import sys
from itertools import permutations
from pathlib import Path

import pytest

from allos import Agent, AgentConfig
from allos.providers import MessageRole, ProviderRegistry
from tests.conftest import PROVIDER_MODELS


def _provider_matches(expected: str, actual: str) -> bool:
    if expected == "chat_completions" and actual == "openai":
        return True
    return expected == actual


def _assert_metadata_shape(metadata, expected_provider: str) -> None:
    assert metadata is not None
    assert _provider_matches(expected_provider, metadata.model.provider)
    assert metadata.model.model_id != ""
    assert metadata.usage.input_tokens >= 0
    assert metadata.usage.output_tokens >= 0
    assert metadata.usage.total_tokens == (
        metadata.usage.input_tokens + metadata.usage.output_tokens
    )
    assert metadata.latency.total_duration_ms >= 0
    assert metadata.turns.total_turns >= 1


PROVIDER_CASES = [
    ("openai", PROVIDER_MODELS["openai"], [pytest.mark.requires_openai]),
    ("anthropic", PROVIDER_MODELS["anthropic"], [pytest.mark.requires_anthropic]),
    ("ollama", PROVIDER_MODELS["ollama"], [pytest.mark.requires_ollama]),
    ("cohere", PROVIDER_MODELS["cohere"], [pytest.mark.requires_cohere]),
    (
        "chat_completions",
        PROVIDER_MODELS["chat_completions"],
        [pytest.mark.requires_openai],
    ),
    (
        "google",
        PROVIDER_MODELS["google"],
        [
            pytest.mark.skipif(
                sys.version_info < (3, 10),
                reason="Google provider requires Python 3.10+",
            ),
            # Agent path currently targets Gemini-key auth by default.
            pytest.mark.requires_gemini,
        ],
    ),
]


def _build_provider_switch_pairs():
    """Generate all A->B provider combinations with per-case marker gating."""
    pairs = []
    for case_a, case_b in permutations(PROVIDER_CASES, 2):
        provider_a, model_a, marks_a = case_a
        provider_b, model_b, marks_b = case_b

        marks = [*marks_a, *marks_b]
        pair_id = f"{provider_a}_to_{provider_b}"

        pairs.append(
            pytest.param(
                provider_a,
                model_a,
                provider_b,
                model_b,
                marks=marks,
                id=pair_id,
            )
        )
    return pairs


PROVIDER_SWITCH_PAIRS = _build_provider_switch_pairs()


@pytest.mark.integration
@pytest.mark.parametrize(
    "provider_a, model_a, provider_b, model_b",
    PROVIDER_SWITCH_PAIRS,
)
def test_session_provider_switch_real(
    provider_a: str,
    model_a: str,
    provider_b: str,
    model_b: str,
    work_dir: Path,
):
    """Run on provider A, save/load session, switch to provider B, continue conversation."""
    agent = Agent(
        AgentConfig(
            provider_name=provider_a,
            model=model_a,
            no_tools=True,
            max_iterations=3,
        )
    )

    first_prompt = "In one short sentence, explain why tests matter."
    first_result = agent.run(first_prompt)

    assert isinstance(first_result, str)
    assert first_result.strip() != ""
    assert agent.last_run_metadata is not None
    _assert_metadata_shape(agent.last_run_metadata, expected_provider=provider_a)

    context_size_after_first_run = len(agent.context.messages)
    assert context_size_after_first_run >= 2

    session_path = work_dir / "provider_switch_session.json"
    agent.save_session(session_path)
    assert session_path.exists()

    loaded_agent = Agent.load_session(session_path)
    assert loaded_agent.config.provider_name == provider_a
    assert loaded_agent.config.model == model_a
    assert len(loaded_agent.context.messages) == context_size_after_first_run

    # Verify session continuity before switching provider.
    first_prompt_still_present = any(
        msg.role == MessageRole.USER and msg.content == first_prompt
        for msg in loaded_agent.context.messages
    )
    assert first_prompt_still_present

    # Switch provider for subsequent turns in the same loaded session.
    loaded_agent.config.provider_name = provider_b
    loaded_agent.config.model = model_b
    loaded_agent.provider = ProviderRegistry.get_provider(provider_b, model=model_b)

    second_prompt = "Now respond in one sentence and reference your previous answer."
    second_result = loaded_agent.run(second_prompt)

    assert isinstance(second_result, str)
    assert second_result.strip() != ""
    assert loaded_agent.last_run_metadata is not None
    _assert_metadata_shape(loaded_agent.last_run_metadata, expected_provider=provider_b)

    # Conversation should continue rather than reset.
    assert len(loaded_agent.context.messages) >= context_size_after_first_run + 2

    user_prompts = [
        msg.content
        for msg in loaded_agent.context.messages
        if msg.role == MessageRole.USER
    ]
    assert first_prompt in user_prompts
    assert second_prompt in user_prompts
