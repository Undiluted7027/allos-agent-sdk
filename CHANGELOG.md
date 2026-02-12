# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.1] - 2025-11-08 - MVP Released

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

## [Unreleased] - 2026-02-08

### Added
- **Ollama Provider** (`allos.providers.ollama.OllamaProvider`): Native Ollama integration
    - Native Ollama client implementation (replacing HTTP-based approach)
    - Model warm-up system with automatic detection and retry logic
    - Connection pooling for 40% performance improvement
    - Automatic model pulling when not available locally
    - Hot reload support for models that need loading into VRAM
    - Enhanced error handling for malformed tool responses
    - Configurable retry logic with exponential backoff
    - Full streaming support
- **Google Provider**: Full support for Google Gemini models (Gemini API and Vertex AI)
  - Gemini 3.x models (preview): `gemini-3-flash-preview`, `gemini-3-pro-preview`
  - Gemini 2.5.x models: `gemini-2.5-flash`, `gemini-2.5-pro`
  - Gemini 2.0.x models: `gemini-2.0-flash`
  - Multiple authentication methods: API key, service accounts, ADC, impersonation
- **Thought Signatures**: Automatic handling of Google Gemini 3.x thought signatures
  - New optional fields in `Message`, `ProviderResponse`, `ProviderChunk`
  - Transparent preservation across conversation turns
  - Required for Gemini 3.x function calling

- **Comprehensive Metadata System** (`allos.providers.metadata`):
    - Unified `Metadata` schema for all LLM interactions with detailed tracking:
    - Token usage (input, output, cached tokens with hit rates)
    - Cost estimation with static pricing database
    - Latency metrics (total duration, time-to-first-token)
    - Tool call details with execution times and status
    - Multi-turn agentic loop tracking (turn history, token breakdown)
    - Quality signals (finish reasons, refusals, response truncation)
    - Provider-specific metadata (OpenAI, Ollama, Google, Anthropic)
    - `MetadataBuilder` factory pattern for consistent construction
    - Multi-modal support placeholders (images, audio, video)

- **Enhanced Streaming Support**:
    - Streaming now supported across ALL providers (OpenAI, Anthropic, Google, Ollama)
    - Unified `ProviderChunk` interface:
    - Incremental content streaming
    - Completed tool calls with `tool_call_done`
    - Tool call deltas for progressive updates
    - Thought signatures in chunks
    - Final metadata delivery
    - Better handling of interleaved text and function calls

- **CLI Enhancements**:
    - New `--metadata` flag for detailed metadata display per turn
    - Improved provider switching with validation
    - Enhanced interactive mode with metadata visualization
    - New utility functions in `allos.cli.utils`:
    - `format_metadata()` for pretty-printing
    - `format_tool_calls()` for tool visualization
    - Model/provider suggestion helpers

- **Testing & Quality**:
    - **99% test coverage** across entire codebase (15,000+ lines of new tests)
    - New comprehensive test suites:
    - 2,346 lines: Google provider tests
    - 1,209 lines: Ollama provider tests
    - 558 lines: Ollama performance tests
    - 485 lines: System integrity tests
    - 366 lines: Ollama warm-up tests
    - 264 lines: Connection pooling tests
    - 163 lines: Metadata system tests
    - 356 lines: CLI utility tests

### Changed
- **BREAKING**: Minimum Python version increased from 3.9 to 3.10
  - Required for Google GenAI SDK dependencies
  - All other providers still work with Python 3.10+
- Updated `Agent` class to handle thought signatures in multi-turn conversations
- Updated `ConversationContext` to store thought signatures alongside messages

- **Provider Interface Enhancements**:
    - `BaseProvider.chat()` returns thought signatures in `ProviderResponse`
    - `BaseProvider.stream_chat()` yields thought signatures in `ProviderChunk`
    - `Message` dataclass extended with `thought_signatures` field
    - All providers updated for unified metadata structure

- **Agent & Context Improvements**:
    - `Agent.run()` handles thought signatures across multi-turn conversations
    - `ConversationContext.add_assistant_message()` accepts thought signatures parameter
    - Enhanced streaming iteration logic for better chunk processing
    - Improved tool execution error handling and reporting

- **OpenAI Provider**:
    - Enhanced metadata extraction from responses
    - Better streaming responses with tool calls
    - Improved error messages for API failures
    - Better token counting for cost estimation

- **Anthropic Provider**:
    - Updated to use latest Messages API features
    - Enhanced streaming support
    - Improved cost estimation accuracy

- **Chat Completions Provider**:
    - Expanded backend support (OpenAI, Azure, together.ai, etc.)
    - Enhanced configuration flexibility
    - Better error handling and validation

- **Dependencies**:
    - Updated `urllib3` to patch security vulnerabilities (CVE fixes)
    - Added `google-genai>=1.47.0` for Google provider
    - Added `google-auth>=2.48.0` for Vertex AI authentication
    - Added `ollama>=0.6.1` for native Ollama client
    - Updated development dependencies for better tooling

### Fixed

- **Google Provider**:
    - Fixed metadata built before response parsing (thought signatures not tracked)
    - Fixed streaming final metadata chunk yielded inside loop instead of at end
    - Fixed warning about non-text parts by manually extracting text from parts
    - Fixed message conversion methods to handle thought signatures correctly
    - Fixed model verification to provide helpful suggestions
    - Converted message conversion from static to instance methods for thought signature tracking

- **Ollama Provider**:
    - Fixed model warm-up detection and retry logic
    - Fixed malformed tool response handling
    - Fixed connection pooling edge cases
    - Fixed streaming response chunk processing
    - Fixed mock test failures for `ollama_running` detection

- **Core Fixes**:
    - Fixed dynamic metadata construction in test mocks
    - Fixed type validation edge cases in provider responses
    - Improved error handling with better error messages across all providers
    - Fixed `max_tokens` handling in Responses API
    - Fixed ruff linting issues in API key validation
    - Fixed mypy type checking issues across codebase

- **Test Infrastructure**:
    - Fixed test mocks to match new Google API structure (parts-based)
    - Fixed static method vs instance method issues in provider tests
    - Updated all provider tests for new metadata structure
    - Fixed race conditions in concurrent test execution
    - Fixed coverage reporting for new code paths

### Documentation

- **New Documentation**:
    - Complete Google provider reference with authentication examples
    - Ollama provider guide with model management and performance tips
    - Metadata system reference with full schema documentation
    - OpenAI API comparison guide for migration from OpenAI SDK
    - Chat Completions provider compatibility guide
    - CLI reference updates for new flags and features

- **Updated Documentation**:
    - Installation guide for Python 3.10+ requirement
    - Quickstart guide with provider selection examples
    - Provider guide expanded with Google and Ollama
    - Agent API reference for thought signature support
    - Contributing guide with new testing requirements

- **Removed Documentation**:
    - Deleted `docs/plans/google_provider_integration.md` (implemented)
    - Deleted `docs/plans/test_coverage_fixes.md` (completed)

### Performance

- **Ollama Provider Optimizations**:
    - Connection pooling reduces overhead by ~40%
    - Model warm-up detection eliminates first-call latency (0-15s saved)
    - Automatic model preloading reduces time-to-first-token
    - Connection reuse eliminates handshake overhead on repeated calls

- **Metadata System**:
    - Efficient builder pattern minimizes object creation
    - Lazy cost calculation only when pricing data available
    - Optimized token counting with caching

### Internal

- **Code Quality**:
    - Added comprehensive docstrings across entire codebase
    - Improved type hints for better IDE support
    - Enhanced error handling with more informative messages
    - Refactored provider message conversion for consistency

- **Development Tools**:
    - Updated `.pre-commit-config.yaml` with better linting rules
    - Improved ruff configuration to ignore specific test files
    - Enhanced mypy configuration for stricter type checking
    - Added performance testing framework

- **Build & Packaging**:
    - Updated `pyproject.toml` with new provider dependencies
    - Enhanced dependency management with optional extras:
    - `pip install allos-agent-sdk[google]` for Google support
    - `pip install allos-agent-sdk[ollama]` for Ollama support
    - `pip install allos-agent-sdk[all]` for all providers

### Metrics

- **Test Coverage**: 99% (23,857 lines added/modified)
- **New Test Files**: 15+ comprehensive test suites
- **Documentation**: 10+ new/updated pages (2,000+ lines)
- **Examples**: 7 new comprehensive examples
- **Commits**: 22 feature/fix commits
