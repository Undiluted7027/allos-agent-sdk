# Contributing to the Allos Agent SDK

First off, thank you for considering contributing to Allos! We're building this project in the open and welcome all contributions, from bug reports and feature suggestions to documentation improvements and code pull requests.

## How Can I Contribute?

- ⭐ **Star the repo:** The easiest way to show your support and helps a lot with visibility!
- 🐛 **Report Bugs:** If you find a bug, please [open a bug report](https://github.com/Undiluted7027/allos-agent-sdk/issues/new?template=bug_report.md).
- 💡 **Suggest Features:** Have a great idea? [Open a feature request](https://github.com/Undiluted7027/allos-agent-sdk/issues/new?template=feature_request.md).
- 🔌 **Request a New Provider:** Want support for a new LLM provider? [Let us know](https://github.com/Undiluted7027/allos-agent-sdk/issues/new?template=provider_request.md).
- 📖 **Improve Documentation:** If you see a typo or think a section could be clearer, please open a PR.
- 🔧 **Submit Pull Requests:** If you want to contribute code, we'd love to have your help.

## Development Workflow

To contribute code, please follow these steps:

1.  **Fork the repository** on GitHub.
2.  **Clone your fork** locally: `git clone https://github.com/YourUsername/allos-agent-sdk.git`
3.  **Set up your development environment.** See our [Development Setup Guide](../docs/contributing/development.md) for detailed instructions.
4.  **Create a new branch** for your feature or bug fix: `git checkout -b feature/my-awesome-feature`.
5.  **Make your changes.** Please add or update tests as appropriate for your changes.
6.  **Run the tests** to ensure everything is still working. See our [Testing Guide](../docs/contributing/testing.md).
7.  **Commit your changes** with a clear and descriptive commit message.
8.  **Push your branch** to your fork on GitHub: `git push origin feature/my-awesome-feature`.
9.  **Open a Pull Request** against the `main` branch of the original repository.

## Code Style Guide

We use `black` for code formatting and `ruff` for linting. These are enforced automatically by pre-commit hooks. As long as you've run the setup script (`./scripts/setup_dev.sh`), your code will be automatically formatted and checked every time you commit.

## Testing Requirements

All contributions that add or modify code must include corresponding tests. Our goal is to maintain 100% test coverage.

-   **Unit Tests (`tests/unit/`)** are required for all new logic.
-   **Integration Tests (`tests/integration/`)** are required if your change interacts with a live API.
-   **E2E Tests (`tests/e2e/`)** are required if your change affects the CLI.

Please see our full [Testing Guide](../docs/contributing/testing.md) for more details on how to run and write tests.

## Adding New Features

If you're planning a significant new feature, it's a good idea to open a feature request issue first to discuss the design and implementation with the maintainers.

-   **For a new provider:** Please follow the [Guide to Adding Providers](../docs/contributing/adding-providers.md).
-   **For a new built-in tool:** Please follow the [Guide to Adding Tools](../docs/contributing/adding-tools.md).

More information is available at [docs/contributing](https://github.com/Undiluted7027/allos-agent-sdk/tree/main/docs/contributing). Thank you again for your contribution!
