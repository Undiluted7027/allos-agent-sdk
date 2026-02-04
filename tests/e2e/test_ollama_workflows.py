# tests/e2e/test_ollama_workflows.py

"""E2E tests for Ollama agent workflows.

These tests use mocked Ollama responses to test full agent workflows
without requiring a running Ollama server. They test the integration
between Agent, OllamaProvider, and Tools.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from allos import Agent, AgentConfig
from allos.cli.main import main
from allos.providers.ollama import _OLLAMA_CLIENT_POOL

pytestmark = pytest.mark.e2e


@pytest.fixture(autouse=True)
def clear_connection_pool():
    """Clear the Ollama connection pool before and after each test."""
    _OLLAMA_CLIENT_POOL.clear()
    yield
    _OLLAMA_CLIENT_POOL.clear()


@pytest.fixture
def runner() -> CliRunner:
    """Provides a CliRunner instance for invoking the CLI."""
    return CliRunner()


# --- E2E Tests with Mocked Ollama ---


@patch("allos.providers.ollama.Client")
def test_ollama_simple_chat_workflow(MockClient, work_dir: Path, mock_metadata_factory):
    """
    E2E Test: Simple chat with Ollama (no tools).
    Tests that Agent properly initializes Ollama provider and handles basic chat.
    """
    # Setup mock
    mock_instance = MockClient.return_value
    mock_instance.list.return_value = MagicMock(
        models=[MagicMock(model="llama3.1:latest")]
    )
    mock_instance.show.return_value = MagicMock(
        capabilities={}, modelinfo={"num_ctx": 128000}
    )
    mock_instance.chat.return_value = {
        "model": "llama3.1:latest",
        "message": {"role": "assistant", "content": "Hello! How can I help you?"},
        "done": True,
        "prompt_eval_count": 10,
        "eval_count": 8,
        "total_duration": 1000000000,
        "prompt_eval_duration": 500000000,
        "eval_duration": 500000000,
    }

    # Create agent with Ollama
    config = AgentConfig(provider_name="ollama", model="llama3.1:latest", no_tools=True)
    agent = Agent(config)

    # Run simple query
    response = agent.run("Hello, who are you?")

    # Assertions
    assert response == "Hello! How can I help you?"
    assert agent.last_run_metadata is not None
    assert agent.last_run_metadata.usage.input_tokens > 0
    assert agent.last_run_metadata.usage.output_tokens > 0
    mock_instance.chat.assert_called_once()


@patch("allos.providers.ollama.Client")
def test_ollama_agent_with_tools_workflow(
    MockClient, work_dir: Path, mock_metadata_factory
):
    """
    E2E Test: Ollama agent using tools to complete a task.
    Tests the full agentic loop: prompt -> tool call -> tool execution -> final response.
    """
    # Setup mock
    mock_instance = MockClient.return_value
    mock_instance.list.return_value = MagicMock(
        models=[MagicMock(model="llama3.1:latest")]
    )
    mock_instance.show.return_value = MagicMock(
        capabilities={"tools": True}, modelinfo={"num_ctx": 128000}
    )

    # First call: Model decides to use write_file tool
    first_response = {
        "model": "llama3.1:latest",
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "write_file",
                        "arguments": {
                            "path": "greeting.txt",
                            "content": "Hello from Ollama!",
                        },
                    }
                }
            ],
        },
        "done": True,
        "prompt_eval_count": 50,
        "eval_count": 10,
    }

    # Second call: Model provides final response after tool execution
    second_response = {
        "model": "llama3.1:latest",
        "message": {
            "role": "assistant",
            "content": "I have created the file 'greeting.txt' with the greeting message.",
        },
        "done": True,
        "prompt_eval_count": 60,
        "eval_count": 15,
    }

    mock_instance.chat.side_effect = [first_response, second_response]

    # Create agent with tools
    config = AgentConfig(
        provider_name="ollama",
        model="llama3.1:latest",
        tool_names=["write_file"],
        auto_approve=True,
    )
    agent = Agent(config)

    # Run task
    response = agent.run(
        "Create a file called greeting.txt with the message 'Hello from Ollama!'"
    )

    # Assertions
    assert "greeting.txt" in response
    assert mock_instance.chat.call_count == 2

    # Verify file was created
    greeting_file = work_dir / "greeting.txt"
    assert greeting_file.exists()
    assert greeting_file.read_text() == "Hello from Ollama!"

    # Check metadata
    assert agent.last_run_metadata is not None
    assert len(agent.last_run_metadata.tools.tool_calls) == 1
    assert agent.last_run_metadata.tools.tool_calls[0].tool_name == "write_file"


@patch("allos.cli.utils.ollama_running", return_value=True)
@patch("allos.providers.ollama.Client")
def test_ollama_cli_integration(
    MockClient, mock_ollama_running, runner: CliRunner, work_dir: Path
):
    """
    E2E Test: Ollama via CLI with file operations.
    Tests the full CLI -> Agent -> Provider -> Tools workflow.
    """
    # Setup mock
    mock_instance = MockClient.return_value
    mock_instance.list.return_value = MagicMock(
        models=[MagicMock(model="llama3.1:latest")]
    )
    mock_instance.show.return_value = MagicMock(
        capabilities={"tools": True}, modelinfo={"num_ctx": 128000}
    )

    # First call: Use read_file tool
    first_response = {
        "model": "llama3.1:latest",
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "read_file", "arguments": {"path": "test.txt"}}}
            ],
        },
        "done": True,
        "prompt_eval_count": 40,
        "eval_count": 8,
    }

    # Second call: Provide answer
    second_response = {
        "model": "llama3.1:latest",
        "message": {
            "role": "assistant",
            "content": "The file contains 3 lines of text.",
        },
        "done": True,
        "prompt_eval_count": 50,
        "eval_count": 12,
    }

    mock_instance.chat.side_effect = [first_response, second_response]

    # Create test file
    (work_dir / "test.txt").write_text("Line 1\nLine 2\nLine 3")

    # Run CLI command
    result = runner.invoke(
        main,
        [
            "--provider",
            "ollama",
            "--model",
            "llama3.1:latest",
            "--auto-approve",
            "Read test.txt and count the lines",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert "3 lines" in result.output.lower()
    assert mock_instance.chat.call_count == 2


@patch("allos.providers.ollama.Client")
def test_ollama_multi_turn_conversation(
    MockClient, work_dir: Path, mock_metadata_factory
):
    """
    E2E Test: Multi-turn conversation with context preservation.
    Tests that the agent maintains conversation context across multiple turns.
    """
    # Setup mock
    mock_instance = MockClient.return_value
    mock_instance.list.return_value = MagicMock(
        models=[MagicMock(model="llama3.1:latest")]
    )
    mock_instance.show.return_value = MagicMock(
        capabilities={}, modelinfo={"num_ctx": 128000}
    )

    # Define responses for multiple turns
    responses = [
        {
            "message": {"role": "assistant", "content": "My name is Ollama Assistant."},
            "done": True,
            "prompt_eval_count": 10,
            "eval_count": 6,
        },
        {
            "message": {
                "role": "assistant",
                "content": "I just told you - my name is Ollama Assistant!",
            },
            "done": True,
            "prompt_eval_count": 20,
            "eval_count": 12,
        },
    ]
    mock_instance.chat.side_effect = responses

    # Create agent
    config = AgentConfig(provider_name="ollama", model="llama3.1:latest", no_tools=True)
    agent = Agent(config)

    # First turn
    response1 = agent.run("What is your name?")
    assert "Ollama Assistant" in response1

    # Second turn - should have context from first turn
    response2 = agent.run("What did you just say your name was?")
    assert "Ollama Assistant" in response2

    # Verify context has both messages
    assert len(agent.context.messages) >= 4  # 2 user + 2 assistant messages
    assert mock_instance.chat.call_count == 2


@patch("allos.providers.ollama.Client")
def test_ollama_streaming_workflow(MockClient, work_dir: Path):
    """
    E2E Test: Streaming response from Ollama.
    Tests that streaming works correctly with the agent.
    """
    # Setup mock
    mock_instance = MockClient.return_value
    mock_instance.list.return_value = MagicMock(
        models=[MagicMock(model="llama3.1:latest")]
    )
    mock_instance.show.return_value = MagicMock(
        capabilities={}, modelinfo={"num_ctx": 128000}
    )

    # Mock streaming response
    stream_chunks = [
        {"message": {"role": "assistant", "content": "The "}, "done": False},
        {"message": {"role": "assistant", "content": "sky "}, "done": False},
        {"message": {"role": "assistant", "content": "is "}, "done": False},
        {"message": {"role": "assistant", "content": "blue."}, "done": False},
        {
            "done": True,
            "model": "llama3.1:latest",
            "total_duration": 1000000000,
            "prompt_eval_count": 15,
            "eval_count": 4,
        },
    ]
    mock_instance.chat.return_value = iter(stream_chunks)

    # Create agent
    config = AgentConfig(provider_name="ollama", model="llama3.1:latest", no_tools=True)
    agent = Agent(config)

    # Stream response
    chunks = list(agent.stream_run("What color is the sky?"))

    # Verify we got chunks
    assert len(chunks) > 0

    # Collect content
    content = "".join(chunk.content for chunk in chunks if chunk.content)
    assert "sky" in content.lower()
    assert "blue" in content.lower()


@patch("allos.providers.ollama.Client")
def test_ollama_error_recovery_workflow(
    MockClient, work_dir: Path, mock_metadata_factory
):
    """
    E2E Test: Agent recovers from tool execution errors.
    Tests that the agent handles tool failures gracefully.
    """
    # Setup mock
    mock_instance = MockClient.return_value
    mock_instance.list.return_value = MagicMock(
        models=[MagicMock(model="llama3.1:latest")]
    )
    mock_instance.show.return_value = MagicMock(
        capabilities={"tools": True}, modelinfo={"num_ctx": 128000}
    )

    # First call: Try to read non-existent file
    first_response = {
        "model": "llama3.1:latest",
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "read_file",
                        "arguments": {"path": "nonexistent.txt"},
                    }
                }
            ],
        },
        "done": True,
        "prompt_eval_count": 30,
        "eval_count": 8,
    }

    # Second call: Acknowledge the error
    second_response = {
        "model": "llama3.1:latest",
        "message": {
            "role": "assistant",
            "content": "I apologize, the file 'nonexistent.txt' does not exist.",
        },
        "done": True,
        "prompt_eval_count": 40,
        "eval_count": 15,
    }

    mock_instance.chat.side_effect = [first_response, second_response]

    # Create agent
    config = AgentConfig(
        provider_name="ollama",
        model="llama3.1:latest",
        tool_names=["read_file"],
        auto_approve=True,
    )
    agent = Agent(config)

    # Run task
    response = agent.run("Read nonexistent.txt")

    # Agent should complete without crashing
    assert "does not exist" in response.lower() or "not exist" in response.lower()
    assert mock_instance.chat.call_count == 2


@patch("allos.providers.ollama.Client")
def test_ollama_session_save_and_load(MockClient, work_dir: Path, tmp_path: Path):
    """
    E2E Test: Save and load agent session with Ollama.
    Tests session persistence and restoration.
    """
    # Setup mock
    mock_instance = MockClient.return_value
    mock_instance.list.return_value = MagicMock(
        models=[MagicMock(model="llama3.1:latest")]
    )
    mock_instance.show.return_value = MagicMock(
        capabilities={}, modelinfo={"num_ctx": 128000}
    )

    responses = [
        {
            "message": {"role": "assistant", "content": "The capital is Paris."},
            "done": True,
            "prompt_eval_count": 15,
            "eval_count": 5,
        },
        {
            "message": {"role": "assistant", "content": "I said the capital is Paris!"},
            "done": True,
            "prompt_eval_count": 25,
            "eval_count": 8,
        },
    ]
    mock_instance.chat.side_effect = responses

    # Create agent and have a conversation
    config = AgentConfig(provider_name="ollama", model="llama3.1:latest", no_tools=True)
    agent = Agent(config)

    response1 = agent.run("What is the capital of France?")
    assert "Paris" in response1

    # Save session
    session_file = tmp_path / "ollama_session.json"
    agent.save_session(session_file)
    assert session_file.exists()

    # Load session into new agent
    loaded_agent = Agent.load_session(session_file)

    # Verify config was preserved
    assert loaded_agent.config.provider_name == "ollama"
    assert loaded_agent.config.model == "llama3.1:latest"

    # Verify context was preserved
    assert len(loaded_agent.context.messages) == len(agent.context.messages)

    # Continue conversation with loaded agent
    response2 = loaded_agent.run("What did you just say?")
    assert "Paris" in response2

    # Verify total chat calls (1 from original agent, 1 from loaded agent)
    assert mock_instance.chat.call_count == 2
