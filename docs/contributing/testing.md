# Testing Guide

Our test suite is built with `pytest` and is divided into three main categories. A high standard of testing is crucial for the reliability of the SDK.

-   **Unit Tests (`tests/unit/`):** These are fast, run in complete isolation, and do not require any external services or API keys. They test individual functions and classes.
-   **End-to-End (E2E) Tests (`tests/e2e/`):** These tests validate the full application flow, primarily through the CLI. They use a **mocked LLM provider** but interact with the **real filesystem and tools**. They **do not** require real API keys to run.
-   **Integration Tests (`tests/integration/`):** These tests verify the interaction between different parts of our SDK by making **real API calls** to external services (like OpenAI and Anthropic). They are slower, may incur costs, and require API keys and a special flag to run.

## Running Tests

We provide a helper script for the most common testing scenario and command-line flags for more specific needs.

### The Default Test Suite (Unit + E2E)

This is the command you should run most often. It executes all unit and E2E tests quickly and without needing any API keys. This is the same command our CI/CD pipeline uses.

```bash
./scripts/run_tests.sh
```

This script is a convenient wrapper around `pytest -m "not integration"`.

### Running a Specific Category of Tests

If you are working on a specific area, you can run just the tests for that category using these flags:

```bash
# Run ONLY the End-to-End (E2E) tests
uv run pytest --run-e2e

# Run ONLY the Integration tests
uv run pytest --run-integration

# Run ONLY the Unit tests (by specifying the directory)
uv run pytest tests/unit/
```

### Running Integration Tests

These tests make **real API calls** and are skipped by default.

**Requirements:**
1.  You must have the appropriate API keys set in a `.env` file at the project root (e.g., `OPENAI_API_KEY=...`).
2.  You must pass the `--run-integration` flag to `pytest`.

```bash
# Run only the integration tests
uv run pytest --run-integration
```

If you are missing a required API key, pytest will skip the relevant tests and print a specific message telling you which environment variable needs to be set.

**Markers:**
*   `@pytest.mark.requires_openai`: Used for tests hitting OpenAI directly OR using the `ChatCompletionsProvider` (which uses the `openai` library).
*   `@pytest.mark.requires_anthropic`: Used for tests hitting Anthropic.

## Writing Tests

Always place new tests in the appropriate directory (`unit`, `e2e`, or `integration`).

-   **Unit Tests:** Place in `tests/unit/`. Use `pytest-mock` (`@patch`) extensively to isolate the component being tested.
-   **E2E Tests:** Place in `tests/e2e/`. Mark them with `@pytest.mark.e2e`. Use `click.testing.CliRunner` to invoke the CLI and assert against the output and exit codes.
-   **Integration Tests:** Place in `tests/integration/`. Mark the test or class with `@pytest.mark.integration`. For provider-specific tests, also add `@pytest.mark.requires_openai` or `@pytest.mark.requires_anthropic` to enable automatic, specific API key checks.
