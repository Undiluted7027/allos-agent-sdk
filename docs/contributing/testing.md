# Testing Guide

Our test suite is built with `pytest` and is divided into two main categories: **unit tests** and **integration tests**. A high standard of testing is crucial for the reliability of the SDK.

## Running Tests

We provide a helper script to run tests consistently.

### Running Unit Tests

Unit tests are fast, run in isolation, and do not require any external services or API keys. They should be run frequently during development.

```bash
./scripts/run_tests.sh
```
This command runs `pytest` and generates a coverage report in the terminal.

### Running Integration Tests

Integration tests make **real API calls** to external services (like OpenAI and Anthropic) to verify that our providers work correctly with the live APIs.

**Requirements:**
- You must have the appropriate API keys set in your `.env` file.
- You must pass the `--run-integration` flag.

```bash
# Run both unit and integration tests
./scripts/run_tests.sh --run-integration
```

If the flag is not provided or the required API keys are missing, these tests will be automatically skipped.

## Writing Tests

-   **Unit Tests:** Place unit tests in `tests/unit/`. These tests should extensively use mocking (e.g., `@patch`) to isolate the component being tested from its dependencies.
-   **Integration Tests:** Place integration tests in `tests/integration/`. These tests should be marked with our custom `run_integration_tests` marker to ensure they are skipped by default.

See `tests/conftest.py` for the definition of the `--run-integration` flag and the `run_integration_tests` marker.
