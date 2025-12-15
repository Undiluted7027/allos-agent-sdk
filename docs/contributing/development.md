# Development Setup

Thank you for your interest in contributing to the Allos Agent SDK! This guide will help you set up a local development environment.

## 1. Prerequisites

- Git
- Python 3.9+
- `uv` (our recommended package manager). If you don't have it, [install it here](https://astral.sh/uv).

## 2. Fork and Clone the Repository

First, fork the repository on GitHub. Then, clone your fork locally:

```bash
git clone https://github.com/YourUsername/allos-agent-sdk.git
cd allos-agent-sdk
```

## 3. Run the Setup Script (Recommended)

We provide a convenient script that automates the entire setup process.

```bash
# Make the script executable (if needed)
chmod +x scripts/setup_dev.sh

# Run the setup script
./scripts/setup_dev.sh
```

This script will:
1.  Check for `uv`.
2.  Create a virtual environment in `.venv/`, preferring Python 3.9 if available.
3.  Install all required and development dependencies using `uv`.
4.  Install pre-commit hooks for automated code formatting and linting.
5.  Create a `.env` file from the example for you to add your API keys.

## 4. Activate the Environment

Activate the virtual environment created by the script:

```bash
source .venv/bin/activate
```

## 5. Add API Keys

Open the newly created `.env` file and add your API keys for the providers you wish to test. This is required for running the integration tests.

```env
# .env
OPENAI_API_KEY="your_openai_key"
ANTHROPIC_API_KEY="your_anthropic_key"

# Optional: Add keys for compatible providers to test the universal adapter
TOGETHER_API_KEY="your_together_key"
GROQ_API_KEY="your_groq_key"
```

## 6. Run the Tests

To verify your setup is working correctly, run the test suite.

```bash
# Run all unit tests
./scripts/run_tests.sh

# Run unit and integration tests (requires API keys)
./scripts/run_tests.sh --run-integration
```

For more details, see the [Testing Guide](./testing.md).

---

## Manual Setup (Alternative)

If you prefer to configure your environment manually without the script, you can use the following commands:

1.  **Install Dependencies:**
    ```bash
    uv sync --all-extras --dev
    source .venv/bin/activate
    ```

2.  **Install Pre-commit Hooks:**
    ```bash
    pre-commit install
    ```

3.  **Setup Environment Variables:**
    ```bash
    cp .env.example .env
    # Edit .env with your keys
    ```


You are now ready to start developing!
