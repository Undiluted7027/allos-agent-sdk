# tests/performance/test_ollama_performance.py

"""Performance tests for Ollama provider.

These tests measure response times, throughput, and resource usage.
They are designed to help benchmark and optimize Ollama performance.
"""

import time
from pathlib import Path

import pytest

from allos import Agent, AgentConfig
from allos.providers import Message, MessageRole, ProviderRegistry

# Mark all tests in this file as performance tests
pytestmark = [pytest.mark.integration, pytest.mark.requires_ollama]


class PerformanceMetrics:
    """Container for performance measurement results."""

    def __init__(self) -> None:
        self.total_duration: float = 0.0
        self.first_token_latency: float = 0.0
        self.tokens_per_second: float = 0.0
        self.total_tokens: int = 0
        self.input_tokens: int = 0
        self.output_tokens: int = 0


def measure_sync_chat_performance(
    provider_name: str, model: str, prompt: str
) -> PerformanceMetrics:
    """Measure performance of synchronous chat completion."""
    metrics = PerformanceMetrics()

    provider = ProviderRegistry.get_provider(provider_name, model=model)
    messages = [Message(role=MessageRole.USER, content=prompt)]

    start_time = time.perf_counter()
    response = provider.chat(messages)
    end_time = time.perf_counter()

    metrics.total_duration = end_time - start_time

    # Extract token counts from metadata
    if response.metadata:
        metrics.input_tokens = response.metadata.usage.input_tokens
        metrics.output_tokens = response.metadata.usage.output_tokens
        metrics.total_tokens = metrics.input_tokens + metrics.output_tokens

        if metrics.total_duration > 0:
            metrics.tokens_per_second = metrics.output_tokens / metrics.total_duration

    return metrics


def measure_streaming_performance(
    provider_name: str, model: str, prompt: str
) -> PerformanceMetrics:
    """Measure performance of streaming chat completion."""
    metrics = PerformanceMetrics()
    first_token_received = False

    provider = ProviderRegistry.get_provider(provider_name, model=model)
    messages = [Message(role=MessageRole.USER, content=prompt)]

    start_time = time.perf_counter()
    first_token_time = 0.0

    for chunk in provider.stream_chat(messages):
        if not first_token_received and chunk.content:
            first_token_time = time.perf_counter()
            metrics.first_token_latency = first_token_time - start_time
            first_token_received = True

        # Count tokens from final metadata chunk
        if chunk.final_metadata:
            metrics.input_tokens = chunk.final_metadata.usage.input_tokens
            metrics.output_tokens = chunk.final_metadata.usage.output_tokens

    end_time = time.perf_counter()
    metrics.total_duration = end_time - start_time
    metrics.total_tokens = metrics.input_tokens + metrics.output_tokens

    if metrics.total_duration > 0 and metrics.output_tokens > 0:
        metrics.tokens_per_second = metrics.output_tokens / metrics.total_duration

    return metrics


@pytest.mark.performance
def test_ollama_sync_chat_latency(default_ollama_model):
    """
    Performance Test: Measure synchronous chat latency.

    Measures total response time for a simple query.
    Expected: < 5 seconds for local models on decent hardware.
    """
    prompt = "What is 2+2? Answer in one word."

    metrics = measure_sync_chat_performance("ollama", default_ollama_model, prompt)

    # Assertions
    assert metrics.total_duration > 0
    assert metrics.output_tokens > 0

    # Log results
    print("\n=== Ollama Sync Chat Performance ===")
    print(f"Model: {default_ollama_model}")
    print(f"Total Duration: {metrics.total_duration:.3f}s")
    print(f"Input Tokens: {metrics.input_tokens}")
    print(f"Output Tokens: {metrics.output_tokens}")
    print(f"Throughput: {metrics.tokens_per_second:.1f} tokens/s")


@pytest.mark.performance
def test_ollama_streaming_first_token_latency(default_ollama_model):
    """
    Performance Test: Measure time to first token in streaming.

    Measures how quickly the model starts responding.
    Expected: < 2 seconds for local models.
    """
    prompt = "Write a short poem about AI."

    metrics = measure_streaming_performance("ollama", default_ollama_model, prompt)

    # Assertions
    assert metrics.first_token_latency > 0
    assert metrics.first_token_latency < metrics.total_duration

    # Log results
    print("\n=== Ollama Streaming Performance ===")
    print(f"Model: {default_ollama_model}")
    print(f"First Token Latency: {metrics.first_token_latency:.3f}s")
    print(f"Total Duration: {metrics.total_duration:.3f}s")
    print(f"Throughput: {metrics.tokens_per_second:.1f} tokens/s")


@pytest.mark.performance
def test_ollama_agent_workflow_performance(default_ollama_model, work_dir: Path):
    """
    Performance Test: Measure full agent workflow with tools.

    Measures end-to-end performance including tool execution.
    """
    # Create a test file
    (work_dir / "test.txt").write_text("Hello World!")

    config = AgentConfig(
        provider_name="ollama",
        model=default_ollama_model,
        tool_names=["read_file"],
        auto_approve=True,
    )

    agent = Agent(config)

    start_time = time.perf_counter()
    response = agent.run("Read test.txt and tell me what it says")
    end_time = time.perf_counter()

    duration = end_time - start_time

    # Assertions
    assert "Hello World" in response or "hello world" in response.lower()
    assert agent.last_run_metadata is not None

    # Log results
    print("\n=== Agent Workflow Performance ===")
    print(f"Total Duration: {duration:.3f}s")
    print(f"Tool Calls: {len(agent.last_run_metadata.tools.tool_calls)}")
    print(f"Turns: {agent.last_run_metadata.turns.total_turns}")
    print(f"Total Tokens: {agent.last_run_metadata.usage.total_tokens}")
