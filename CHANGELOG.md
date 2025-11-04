# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.1] - YYYY-MM-DD - MVP Release (PLANNED)

### ✨ Added

-   **Initial MVP Release of the Allos Agent SDK!**
-   **Core Agent (`allos.agent`):**
    -   Implemented the core `Agent` class with a robust, multi-turn agentic loop.
    -   Added `AgentConfig` for easy and type-safe agent configuration.
    -   Implemented a human-in-the-loop permission system (`ToolPermission`) for secure tool execution.
    -   Added full session management with `agent.save_session()` and `Agent.load_session()`.
-   **Provider Agnostic Layer (`allos.providers`):**
    -   Created a flexible provider abstraction (`BaseProvider`) and `ProviderRegistry`.
    -   Added full support for the **OpenAI Responses API**.
    -   Added full support for the **Anthropic Messages API**.
-   **Tool System (`allos.tools`):**
    -   Created an extensible tool abstraction (`BaseTool`) and `ToolRegistry` with a `@tool` decorator for easy custom tool creation.
    -   Added a comprehensive set of built-in tools:
        -   `read_file` (with line range support)
        -   `write_file` (with append mode)
        -   `edit_file` (with uniqueness validation)
        -   `list_directory` (with recursive support)
        -   `shell_exec` (with timeouts and a security blocklist)
-   **Command-Line Interface (`allos.cli`):**
    -   Developed a powerful and user-friendly `allos` CLI tool.
    -   Supports direct task execution (e.g., `allos "do something"`).
    -   Includes a full-featured interactive mode (`allos -i`).
    -   Provides commands for discovering tools and providers (`--list-tools`, `--list-providers`).
    -   Supports all core agent features like provider switching, tool selection, session management, and auto-approval via flags.
-   **Testing and Quality:**
    -   Achieved 100% unit test coverage for the core Python library.
    -   Built comprehensive integration and E2E test suites.
    -   Established a CI/CD pipeline for testing across multiple platforms (Linux, macOS, Windows) and Python versions (3.9-3.13).
-   **Documentation and Examples:**
    -   Created a full documentation suite with guides, tutorials, and API references.
    -   Provided a rich set of runnable examples for all major features.
