# Allos Agent SDK Documentation

Welcome to the official documentation for the Allos Agent SDK.

Allos is an open-source, provider-agnostic agentic SDK that gives you the power to build production-ready AI agents that work with **any LLM provider**.

## Key Sections

- **[Installation](./installation.md)**: How to install the SDK and its dependencies.
- **[Getting Started](./getting-started.md)**: A conceptual overview of the first steps.
- **[Quickstart Guide](./guides/quickstart.md)**: Get your first LLM response in under 5 minutes.
- **[Guides](./guides/)**: In-depth explanations of core features.
  - **[Agents](./guides/agents.md)**: Learn how to configure and run the core `Agent`.
  - **[Providers](./guides/providers.md)**: Learn how to use and switch between different LLM providers like OpenAI and Anthropic.
  - **[Tools](./guides/tools.md)**: Learn how to use the built-in tools for filesystem and shell operations.
  - **[Custom Tools](./guides/custom-tools.md)**: Learn how to create your own custom tools.
- **[Contributing](./contributing/)**: Find out how to contribute to the project.
  - **[Development Setup](./contributing/development.md)**: Set up your local development environment.
  - **[Adding a New Provider](./contributing/adding-providers.md)**: A guide for extending Allos with new LLM providers.
  - **[Adding a New Tool](./contributing/adding-tools.md)**: A guide for adding new built-in tools.
  - **Note:** The core agent logic is not yet extensible in the MVP. See our [Roadmap](./../ROADMAP.md) for future plans on sub-agents and custom agent architectures.
- **[Reference](./reference/)**: Detailed technical reference for the API.
  - **[CLI Reference](./reference/cli-reference.md)**: A complete reference for the `allos` command-line tool.
  - **[Provider API](./reference/provider-api.md)**: Technical details of the provider abstraction layer.
  - **[Tool API](./reference/tool-api.md)**: Technical details of the tool abstraction layer.
  - **[Agent API](./reference/agent-api.md)**: Technical details of the `Agent` class and its configuration.
