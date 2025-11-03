# Testing Guide

Our test suite is built with `pytest` and is divided into three main categories. A high standard of testing is crucial for the reliability of the SDK.

-   **Unit Tests (`tests/unit/`):** These are fast, run in complete isolation, and do not require any external services or API keys. They test individual functions and classes.
-   **Integration Tests (`tests/integration/`):** These tests verify the interaction between different parts of our SDK or make **real API calls** to external services (like OpenAI and Anthropic). They require API keys and a special flag to run.
-   **End-to-End (E2E) Tests (`tests/e2e/`):** These tests validate the CLI application from an end-user's perspective, simulating command-line invocations. They use mocks to isolate the CLI logic from the agent's core and require API keys.

## Running Tests

We provide a helper script to run tests consistently.

### Running Unit Tests (Fastest)

This is the most common command you'll run during development. It's fast and checks the core logic. Our CI pipeline runs this on every commit.

```bash
uv run pytest tests/unit/
```
*(You can also use `./scripts/run_tests.sh`, which defaults to running all tests but will be slow if it includes E2E tests.)*

### Running CLI (E2E, Requires Keys) Tests

These tests are also fast and require API keys. They are essential for verifying any changes to the CLI.

```bash
uv run pytest tests/e2e/
```

### Running Integration Tests (Slowest, Requires Keys)

These tests make **real API calls**.

**Requirements:**
-   You must have the appropriate API keys set in your `.env` file.
-   You must pass the `--run-integration` flag to `pytest`.

```bash
# Run only the integration tests
uv run pytest tests/integration/ --run-integration
```

### Running the Full Test Suite Locally

To run every single test (unit, integration, and E2E) on your local machine:

```bash
./scripts/run_tests.sh --run-integration
```

## Writing Tests

-   **Unit Tests:** Place in `tests/unit/`. Use `pytest-mock` (`@patch`) extensively to isolate the component being tested.
-   **Integration Tests:** Place in `tests/integration/`. Mark them with the `@pytest.mark.integration` decorator (defined in `tests/conftest.py` and `pyproject.toml`) to ensure they are skipped by default.
-   **E2E Tests:** Place in `tests/e2e/`. Use `click.testing.CliRunner` to invoke the CLI and assert against the output and exit codes.
