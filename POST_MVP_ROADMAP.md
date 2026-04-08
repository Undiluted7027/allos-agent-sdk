# Allos Agent SDK - POST-MVP Roadmap

**Goal**: Transform Allos from a working MVP into the most flexible, powerful, and developer-friendly agentic SDK

**Timeline**: Weeks 9-36+ (28+ weeks of active development)

**Current Status**: MVP Launched (Nov 8, 2025), Phase 2 in Progress

---

## 🎯 POST-MVP Vision

Build upon the solid MVP foundation to deliver:
1. **Universal Provider Support**: OpenAI, Anthropic, Ollama, Google, and more
2. **Rich Tool Ecosystem**: Web tools, database tools, and extensible plugins
3. **Production-Ready Features**: Advanced monitoring, error recovery, and scaling
4. **Best-in-Class DX**: Configuration files, IDE integrations, and templates
5. **Ecosystem Integration**: Pydantic AI, Smolagents, MCP, and framework compatibility

**Post-MVP Success Metric**: Allos becomes the go-to choice for building production-ready AI agents with any LLM provider.

---

## 📊 Progress Tracker

| Phase | Focus | Timeline | Status | Completion |
|-------|-------|----------|--------|------------|
| **Phase 1: MVP** | Core functionality | Weeks 1-8 | ✅ Complete | 100% |
| **Phase 2: Enhanced Features** | Essential capabilities | Weeks 9-14 | 🎯 Current | 50% |
| **Phase 3: Advanced Tooling** | Developer experience | Weeks 15-20 | 📋 Planned | 0% |
| **Phase 4: Enterprise & Scale** | Production features | Weeks 21-28 | 📋 Planned | 0% |
| **Phase 5: Ecosystem Integration** | Framework compatibility | Weeks 29-36 | 🔮 Future | 0% |
| **Phase 6: Innovation** | Cutting-edge features | Ongoing | 🔮 Future | 0% |

**Overall Post-MVP Progress**: 16.67% (1/6 phases complete)

---

## Phase 2: Enhanced Features

**Timeline**: Weeks 9-14 (6 weeks)

**Status**: 🏗️ In Progress (50%)

**Focus**: Essential capabilities for production use

### Goals
- Add local model support (Ollama)
- Expand provider ecosystem (Google, Cohere, Bedrock)
- Implement web tools (search, fetch)
- Build advanced context management
- Create configuration system
- Establish plugin architecture foundation

### Progress Breakdown (50%)

| Feature | Duration | Status | Completion |
|---------|----------|--------|------------|
| **2.0.1 Chat Completions Provider** | Days 57-80 | ✅ Complete | 100% |
| **2.0.2 Native Ollama Provider** | Days 80-100 | ✅ Complete | 100% |
| **2.0.3 Native Google Provider** | Days 90-130 | ✅ Complete | 100% |
| **2.0.4 Native Cohere Provider** | Days 130-140 | ✅ Complete | 100% |
| **2.1 CLI Enhancements** | Week 13 | ✅ Complete | 100% |
| **2.2 Cloud Providers** | Week 10 | 📋 Planned | 0% |
| **2.3 Web Tools** | Week 11 | 📋 Planned | 0% |
| **2.4 Advanced Context** | Week 12 | 📋 Planned | 0% |
| **2.5 Configuration System** | Week 13 | 📋 Planned | 0% |
| **2.6 Plugin System** | Week 14 | 📋 Planned | 0% |

---

## 2.0 Additional Providers (3 Months)

**Goal:** Expand Allos to cover more LLM providers. Take list of providers from OpenCode.

### 2.0.1 Chat Completions Provider (Days 57-80)

**Duration**: 23 days

**Status**: ✅ Complete

**Goal**: Create reusable Chat Completions API provider for OpenAI-compatible services

**Motivation**: Enable compatibility with Together AI, Anyscale, and provide alternative for testing

#### Chat Completions Architecture & Design

##### Research & Design
- [x] Study Chat Completions API vs Responses API differences
- [x] Document API contract differences
- [x] Design provider interface
- [x] Plan backward compatibility strategy
- [x] Create comparison matrix

##### Key Differences

| Feature | Responses API | Chat Completions API |
|---------|---------------|---------------------|
| Endpoint | `/v1/responses` | `/v1/chat/completions` |
| Tool Calling | Native multi-turn | Function calling |
| Message Format | Response items | Messages array |
| Streaming | Response events | SSE chunks |
| Tool Results | Structured | Function call messages |

#### Chat Completions Provider Implementation

##### Implementation
- [x] **`allos/providers/chat_completions.py`**
  - `ChatCompletionsProvider` class extending `BaseProvider`
  - OpenAI Chat Completions API integration
  - Message format conversion
  - Function calling (tool calling)
  - Streaming support
  - Configurable base_url for compatibility

```python
import openai
from typing import List, Optional
from ..base import BaseProvider, Message, ProviderResponse, ToolCall, MessageRole

@provider
class ChatCompletionsProvider(BaseProvider):
    """
    Provider for OpenAI Chat Completions API.

    This provider supports:
    - OpenAI's Chat Completions endpoint
    - OpenAI-compatible APIs (Together AI, Anyscale, etc.)

    Note: For OpenAI's advanced features, use OpenAIProvider (Responses API).
    This provider is for compatibility with OpenAI-compatible services.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model, **kwargs)
        self.base_url = base_url

        try:
            # Allow custom base_url for OpenAI-compatible APIs
            self.client = openai.OpenAI(
                api_key=api_key,
                base_url=base_url,
                **kwargs
            )

        except Exception as e:
            raise ProviderError(
                f"Failed to initialize ChatCompletions client: {e}",
                provider_name="chat_completions",
            ) from e


    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[dict]] = None
    ) -> ProviderResponse:
        """Send chat request using Chat Completions API"""

        # Convert messages to Chat Completions format
        chat_messages = self._convert_to_chat_format(messages)

        # Prepare request
        request_params = {
            'model': self.model,
            'messages': chat_messages,
            'temperature': self.temperature,
        }

        # Add tools as functions
        if tools:
            request_params['functions'] = self._convert_tools_to_functions(tools)
            request_params['function_call'] = 'auto'

        # Make request
        try:
            response = self.client.chat.completions.create(**request_params)
            return self._convert_response(response)

        except openai.APIError as e:
            raise ProviderError(f"OpenAI API error: {str(e)}")

    def _convert_to_chat_format(self, messages: List[Message]) -> List[dict]:
        """Convert Allos messages to Chat Completions format"""
        chat_messages = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                chat_messages.append({
                    'role': 'system',
                    'content': msg.content
                })
            elif msg.role == MessageRole.USER:
                chat_messages.append({
                    'role': 'user',
                    'content': msg.content
                })
            elif msg.role == MessageRole.ASSISTANT:
                message = {'role': 'assistant'}

                if msg.content:
                    message['content'] = msg.content

                # Handle function calls (tool calls)
                if msg.tool_calls:
                    message['tool_calls'] = [
                        {
                            'id': tc.id,
                            'type': 'function',
                            'function': {
                                'name': tc.name,
                                'arguments': json.dumps(tc.arguments),
                            },
                        } for tc in msg.tool_calls
                    ]

                chat_messages.append(message)

            elif msg.role == MessageRole.TOOL:
                # Tool results as function responses
                chat_messages.append({
                    'role': 'function',
                    "tool_call_id": msg.tool_call_id,
                    'content': msg.content
                })

        return chat_messages

    def _convert_tools_to_functions(self, tools: List[dict]) -> List[dict]:
        """Convert Allos tools to OpenAI functions format"""
        functions = []

        for tool in tools:
            properties = {}
            required_params = []
            for param in tool.parameters:
                properties[param.name] = {
                    "type": param.type,
                    "description": param.description,
                }
                if param.requried:
                    required_params.append(param.name)

            function_def = {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required_params
                },
            }

            # Wrap us in the tool object
            functions.append({
                "type": "function",
                "function": function_def
            })

        return functions

    def _convert_response(self, response) -> ProviderResponse:
        """Convert Chat Completions response to Allos format"""
        message = response.choices[0].message

        # Extract tool calls
        tool_calls = []
        if hasattr(message, 'function_call') and message.function_call:
            tool_calls.append(ToolCall(
                id=f"call_{response.id}",  # Generate ID
                name=message.function_call.name,
                arguments=json.loads(message.function_call.arguments)
            ))

        return ProviderResponse(
            content=message.content,
            tool_calls=tool_calls if tool_calls else None,
            raw_response=response
        )
```

##### Features
- [x] Support for custom base_url
- [x] Function calling (OpenAI-style tool calling)
- [x] Message format conversion
- [x] Streaming support (REQUIRED Implementing Streaming Across ALL PROVIDERS)
- [x] Error handling
- [x] Token counting

#### Testing & Documentation

##### Testing
- [x] **`tests/unit/test_chat_completions_provider.py`**
  - Mock OpenAI client
  - Test message conversion
  - Test function calling
  - Test with custom base_url
  - Test error handling

##### Integration Tests
- [x] Test with real OpenAI Chat Completions endpoint
- [x] Test with Together AI (if API key available)
- [x] Compare behavior with ResponsesAPI provider
- [x] Test alias endpoints

##### Documentation
- [x] **`docs/providers/chat-completions.md`**
  - When to use Chat Completions vs Responses API
  - Configuration for different services
  - Limitations compared to Responses API
  - Migration guide

#### Deliverables

- ChatCompletionsProvider class
- Support for custom base_url
- Function calling implementation
- Comprehensive tests
- Documentation and comparison guide

---

### 2.0.2 Native Ollama Provider (Days 80-100)

**Duration**: 20 days

**Status**: ✅ Complete

**Goal**: Enable completely local, private AI agents

**Motivation**: Users want to run agents offline without API costs or privacy concerns

#### Ollama Provider Core

##### Research & Design
- [x] Study Ollama Python library API documentation
- [x] Design provider interface to match BaseProvider
- [x] Plan token counting strategy (model-specific)
- [x] Design context window detection per model
- [x] Plan streaming implementation

##### Implementation
- [x] **`allos/providers/ollama.py`**
  - `OllamaProvider` class extending `BaseProvider`
  - Connection to local Ollama server (default: localhost:11434)
  - Support for custom endpoints via environment variable
  - Model availability checking
  - Context window detection per model family
  - Register with `@provider` decorator
  - Options like `temperature` and `num_predict` are passed as `options` parameter in model calls. Like so:


##### Testing
- [x] **`tests/unit/test_ollama_provider.py`**
  - Mock Ollama client (native library)
  - Test model availability checking
  - Test message conversion
  - Test tool calling (native format)
  - Test error handling
  - Test context window detection

#### Ollama Tool Calling & Streaming

##### Tool Calling Support
- [x] **Native tool calling implementation**
  - Convert Allos tool format to Ollama format (native, not OpenAI functions)
  - Handle tool call responses from Ollama's native format
  - Extract tool call IDs
  - Support multiple tool calls in one turn

### ADDON: Basic Metadata & Observability System

#### Metadata Schema
- [x] **`allos/providers/metadata.py`**
  - [x] `Metadata` Pydantic model with comprehensive schema
  - [x] `MetadataBuilder` factory pattern
  - [x] `Usage` tracking (input/output tokens, cache, cost)
  - [x] `Latency` metrics (total duration, TTFT)
  - [x] `ToolInfo` and `ToolCallDetail` for tool tracking
  - [x] `TurnsInfo` and `TurnLog` for turn history
  - [x] `ProviderSpecific` for provider-unique fields
  - [x] `QualitySignals` for finish reasons

#### Provider-Specific Metadata
- [x] **OpenAI**: `system_fingerprint`
- [x] **Ollama**: `warm_up`, `warm_up_duration_seconds`

#### First Metadata Preservation
- [x] Provider-specific traits preserved from first turn
- [x] Backfill mechanism in `_create_aggregate_metadata()`
- [x] Ensures warm_up, system_fingerprint retained across turns

#### Agent Integration
- [x] `agent.last_run_metadata` populated after each run
- [x] Cumulative token/cost tracking across turns
- [x] Turn history with per-turn metrics

**Code Example**:
```python
result = agent.run("Create a FastAPI app")
metadata = agent.last_run_metadata

print(f"Tokens: {metadata.usage.total_tokens}")
print(f"Cost: ${metadata.usage.estimated_cost.total_usd}")
print(f"Turns: {metadata.turns.total_turns}")
print(f"Tools used: {metadata.tools.total_tool_calls}")
```


### ADDON: Basic Streaming Support (Moved from Phase 3.6)

#### Streaming Architecture
- [x] **`allos/providers/base.py`**
  - [x] `ProviderChunk` dataclass for streaming chunks
  - [x] `stream_chat()` abstract method in BaseProvider
  - [x] Streaming protocol definition

- [x] **Provider Implementations**
  - [x] OpenAI `stream_chat()` with Responses API events
  - [x] Anthropic `stream_chat()` with streaming messages
  - [x] Ollama `stream_chat()` with native streaming
  - [x] Chat Completions `stream_chat()` with SSE

- [x] **Agent Streaming**
  - [x] `Agent.stream_run()` method
  - [x] `CumulativeState` TypedDict for tracking and metadata
  - [x] Streaming metadata aggregation
  - [x] Tool execution during streaming

- [x] **CLI Support**
  - [x] `--stream` flag for one-shot streaming

#### Testing
- [x] Unit tests for streaming providers
- [x] E2E streaming workflow tests
- [x] Integration tests with real APIs (`test_streaming_real.py`)
- [x] Streaming with tool calls (`test_agent_stream_tool_loop_real.py`)

**Code Example**:
```bash
# CLI streaming
allos --stream "Write a long story about space exploration"

# Python API
for chunk in agent.stream_run("Create a web app"):
    print(chunk.content, end='', flush=True)
```

##### Testing
- [x] **Integration tests with real Ollama**
  - Test with llama3.2:1b (fast, small model)
  - Test tool calling with real model
  - Test streaming responses
  - Verify context window handling

#### Context Window Detection & Model Support

##### Context Window Management
- [x] **Model-specific context windows**
  - Create model family mappings (Llama, Mistral, Qwen, etc.)
  - Detect context window from model name and actual model metadata
  - Add override via configuration
  - Implement token counting per model family


##### Token Counting
- [x] **Implement token counting**
  - Use tiktoken for estimation (by model family)
  - Fall back to word-based estimation
  - Cache token counts
  - Integrate with context manager

##### Popular Model Testing
- [x] Test with popular models:
  - `llama3.2:3b` - Latest small Llama
  - `qwen2.5-coder:7b` - Code-focused
  - `mistral:7b` - General purpose
  - `deepseek-coder:6.7b` - Code specialist

#### Documentation & Examples

##### Documentation
- [x] **`docs/providers/ollama.md`**
  - Installation instructions (Ollama + models)
  - Configuration guide
  - Model recommendations
  - Tool calling guide
  - Streaming examples
  - Troubleshooting section
  - Performance tips

##### Examples
- [x] **`examples/ollama_usage.py`**
  - Basic local model usage
  - Model comparison (local vs API)
  - Streaming example
  - Tool calling example

```python
"""
Ollama Provider Example - Local AI Agents

This example demonstrates how to use Ollama for completely
local, private AI agents with no API costs.
"""

from allos import Agent, AgentConfig

def basic_ollama_usage():
    """Run agent with local Ollama model"""
    print("\n=== Basic Ollama Usage ===")

    agent = Agent(AgentConfig(
        provider_name="ollama",
        model="qwen2.5-coder:7b",
        tool_names=["read_file", "write_file", "shell_exec"]
    ))

    result = agent.run("Create a simple FastAPI hello world app")
    print(f"Result: {result}")

def compare_providers():
    """Compare local vs cloud providers"""
    print("\n=== Comparing Providers ===")

    task = "Explain what FastAPI is in one sentence"

    providers = [
        ("ollama", "llama3.2:3b"),
        ("openai", "gpt-4"),
        ("anthropic", "claude-sonnet-4-5")
    ]

    for provider, model in providers:
        print(f"\nTesting {provider}/{model}...")
        agent = Agent(AgentConfig(
            provider_name=provider,
            model=model,
            tool_names=[]
        ))
        result = agent.run(task)
        print(f"Response: {result}")

if __name__ == "__main__":
    basic_ollama_usage()
    compare_providers()
```

- [x] **`examples/local_vs_cloud.py`**
  - Side-by-side comparison
  - Cost analysis
  - Performance benchmarks
  - Privacy considerations

##### Update Main Documentation
- [x] Update `README.md` provider table
- [x] Update `docs/guides/providers.md`
- [x] Add Ollama to quickstart guide

#### Testing & Polish

##### Comprehensive Testing
- [x] **Unit tests**: Mock-based, no Ollama required
- [x] **Integration tests**: Real Ollama server required
- [x] **E2E tests**: Full agent workflows with Ollama
- [x] **Performance tests**: Measure response times

##### Edge Cases
- [x] Model not available (suggest `ollama pull`)
- [x] Ollama server not running
- [x] Network timeout handling
- ~~[ ] Large context handling~~
- [x] Streaming interruption
- [x] Tool calling errors

##### Performance Optimization
- [x] Connection pooling
- ~~[ ] Request caching~~
- [x] Model warm-up detection
- ~~[ ] Memory usage monitoring~~

##### CLI Integration
- [x] Add `--list-ollama-models` command
- [x] Add Ollama-specific help text

##### Final Polish
- [x] Code review and refactoring
- [x] Documentation review
- [x] Example testing
- [x] Performance profiling
- ~~[ ] Security review~~

#### Success Criteria

✅ Ollama provider fully functional

✅ Native tool calling works with supported models

✅ Streaming responses implemented

✅ Context window detection working

✅ 10+ popular models tested and documented

✅ All tests pass (unit, integration, E2E)

✅ Documentation complete with examples

✅ Performance acceptable (< 2x API latency)

#### Deliverables

- Working Ollama provider with streaming support
- Tool calling for local models
- Comprehensive documentation
- 3+ working examples
- Model comparison guide
- Updated CLI with Ollama support
- Integration and E2E tests

#### Example Usage After This Phase

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull qwen2.5-coder:7b

# Use with Allos
allos --provider ollama --model qwen2.5-coder:7b "Create a web scraper"

# List available local models
allos --list-ollama-models

# Interactive mode with local model
allos --provider ollama --model llama3.2:3b --interactive
```

```python
# Python API
from allos import Agent, AgentConfig

# Local coding agent
agent = Agent(AgentConfig(
    provider_name="ollama",
    model="qwen2.5-coder:7b",
    tool_names=["read_file", "write_file", "shell_exec"],
    temperature=0.7
))

result = agent.run("Review the code in src/ and suggest improvements")
print(result)

# Streaming mode
agent_stream = Agent(AgentConfig(
    provider_name="ollama",
    model="llama3.2:3b",
    tool_names=["read_file"]
))

for chunk in agent_stream.stream_run("Explain this codebase"):
    print(chunk, end='', flush=True)
```

---

<!-- ## 2.2 Additional Providers (Week 10)

**Duration**: 1 week (Days 71-77)

**Status**: 📋 Planned

**Goal**: Expand provider ecosystem to 7+ providers

**Architecture Note**: Together AI will extend the `ChatCompletionsProvider` created in Days 57-59, leveraging the OpenAI-compatible Chat Completions API. -->

### 2.0.3 Google Gemini Provider (Days 90-130)

**Duration**: 40 days

**Status**: ✅ Complete

**Goal**: Create Google provider client to interact with Gemini (Vertex AI / Google Studio)

#### Core Implementation
- [x] **`allos/providers/google.py`** (~1078 lines)
  - [x] `GoogleProvider` class extending `BaseProvider`
  - [x] Google AI Studio API integration (Gemini API)
  - [x] Vertex AI support with multiple auth methods:
    - [x] API key (Gemini API)
    - [x] Service account JSON file
    - [x] Application Default Credentials (ADC)
    - [x] Service account impersonation
  - [x] Native tool calling
  - [x] **Thought signatures** for Gemini 3.x (required) and 2.5.x (optional)
  - [x] Streaming support via `stream_chat()` with genai
  - [x] Dynamic context window detection (1M-2M tokens)
  - [x] Python 3.10+ requirement (enforced at import)

#### Features
- [x] Support for Gemini 2.5 Pro, 2.5 Flash, 3.0 Flash, 3.0 Pro
- [x] Vertex AI integration
- [x] Native tool calling
- [x] **Thought signatures** for Gemini 3.x (required) and 2.5.x (optional)
- [x] Streaming support via `stream_chat()`
- [x] Dynamic context window detection
- [x] Python 3.10+ requirement (enforced at import)

#### Provider-Specific Metadata
- [x] **Google**: `vertexai`, `project`, `location`, `used_thought_signatures`

#### Testing
- [x] Unit tests with mocked responses
- [x] Integration tests with real API (`tests/integration/providers/test_google_real.py`)
- [x] Tool calling validation
- [x] Context window testing
- [x] Thought signature validation (`tests/integration/providers/test_google_thought_signatures_real.py`)

#### Examples
- [x] `examples/google_gemini_usage.py`
- [x] `examples/metadata_inspection.py`

#### Documentation
- [x] **`docs/providers/google.md`**
  - API key setup (AI Studio vs Vertex)
  - Model selection guide
  - Tool calling examples
  - Multimodal examples (future)

**Delivered:**
- ✅ Full support for Gemini models (1.5, 2.0, 2.5, 3.x)
- ✅ Gemini API and Vertex AI integration
- ✅ Multiple authentication methods (API key, service account, ADC, impersonation)
- ✅ Automatic thought signature handling for Gemini 3.x
- ✅ Model verification with helpful suggestions
- ✅ Comprehensive metadata tracking
- ✅ Full streaming support
- ✅ 2,346 lines of tests (99% coverage)
- ✅ Complete documentation
- ✅ Python 3.10+ requirement clearly documented
- ✅ CI/CD updated for Python version testing

**Python Version:** Requires Python 3.10+ (due to google-auth dependency)

### 2.0.4: Native Cohere Provider (Days 130-140)

**Duration**: 10 days

**Status**: ✅ Complete

**Goal**: Create native Cohere provider client to interact with Cohere APIs.

#### Implementation
- [x] **`allos/providers/cohere.py`**
  - `CohereProvider` class
  - Command R / R+ support
  - Native tool calling
  - Context window: 200K tokens

```python
import cohere

@provider
class CohereProvider(BaseProvider):
    """Provider for Cohere Command models"""

    def __init__(self, model: str = "command-r-plus", **kwargs):
        super().__init__(model, **kwargs)
        self.client = cohere.Client(os.getenv('COHERE_API_KEY'))
        self.context_window = 200_000
```

#### Features
- [x] Command R support
- [x] Command R+ support
- [x] Native tool calling
- [x] Streaming support

### Day 75: Amazon Bedrock Provider

#### Implementation
- [ ] **`allos/providers/bedrock.py`**
  - `BedrockProvider` class
  - AWS authentication (boto3)
  - Support for:
    - Claude via Bedrock
    - Llama via Bedrock
    - Mistral via Bedrock
    - Titan models
  - Region configuration

```python
import boto3

@provider
class BedrockProvider(BaseProvider):
    """Provider for Amazon Bedrock models"""

    def __init__(self, model: str, region: str = "us-east-1", **kwargs):
        super().__init__(model, **kwargs)
        self.client = boto3.client(
            'bedrock-runtime',
            region_name=region
        )
```

### Azure OpenAI
- [ ] **`allos/providers/azure_openai.py`**
  - Extends `OpenAIProvider` (Responses API)
  - Azure-specific endpoint configuration
  - Deployment name handling
  - API version management

```python
from .openai import OpenAIProvider
from .registry import provider

@provider
class AzureOpenAIProvider(OpenAIProvider):
    """
    Provider for Azure OpenAI Service.

    Extends OpenAIProvider (Responses API) with Azure-specific configuration.
    Note: Uses Responses API, not Chat Completions API.
    """

    def __init__(
        self,
        deployment_name: str,
        api_version: str = "2024-08-01-preview",
        **kwargs
    ):
        # Azure-specific configuration
        azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        azure_key = os.getenv('AZURE_OPENAI_API_KEY')

        super().__init__(
            model=deployment_name,  # Azure uses deployment names
            api_key=azure_key,
            base_url=f"{azure_endpoint}/openai/deployments/{deployment_name}",
            **kwargs
        )
        self.api_version = api_version
```

### Day 77: Testing & Documentation

#### Comprehensive Testing
- [ ] Test all current native providers (OpenAI, Anthropic, Chat Completions, Ollama, Google, Cohere)
- [ ] Add smoke coverage for selected alias providers (e.g., Together, Groq, OpenRouter, Cohere Compat)
- [ ] Provider switching tests
- [ ] Tool calling across providers
- [ ] Error handling
- [ ] Performance comparison

#### Documentation
- [ ] **Update provider comparison table**
  - API type (Responses vs Chat Completions vs Native)
  - Tool calling support
  - Context windows
  - Cost comparison
  - Speed benchmarks

- [ ] **Provider selection guide**
  - When to use each provider
  - Cost considerations
  - Feature comparison
  - Migration paths

```markdown
# Provider Comparison

| Provider | API Type | Tool Calling | Context Window | Cost ($/1M tokens) |
|----------|----------|--------------|----------------|-------------------|
| OpenAI | Responses API | Native multi-turn | 128K | $2.50 |
| Anthropic | Native | Native | 200K | $3.00 |
| Ollama | Native Python | Native | Varies | $0 (local) |
| Google | Native | Native | 2M | $1.25 |
| Cohere | Native | Native | 200K | $3.00 |
| Together AI | Chat Completions | Function calling | 32K-128K | $0.60 |
| Anyscale | Chat Completions | Function calling | 32K | $1.00 |
| Bedrock | Native | Varies | Varies | AWS pricing |
| Azure OpenAI | Responses API | Native multi-turn | 128K | Azure pricing |
```

### Success Criteria

✅ 9+ providers supported (OpenAI, Anthropic, Ollama, Google, Cohere, Bedrock, Together AI, Anyscale, Azure)

✅ ChatCompletionsProvider successfully extended for Together AI and Anyscale

✅ All providers support tool calling

✅ Seamless provider switching

✅ Comprehensive documentation

✅ All tests pass

### Deliverables

- 7+ new providers implemented
  - Google (native library)
  - Cohere (native library)
  - Bedrock (boto3)
  - Azure OpenAI
- Provider comparison guide
- Setup documentation for each
- Updated examples showing provider switching
- Architecture documentation explaining API differences

---

## 2.2 CLI & DX Enhancements

**Duration:** 3 days
**Status:** ✅ Complete
**Goal:** Improve interactive mode and developer experience

#### CLI Flags addition
- [x] `--active-providers` - Show provider readiness status
- [x] `--list-ollama-models` - List local Ollama models
- [x] `--stream` - Enable streaming mode
- [x] `--no-tools` - Disable all tools
- [x] `--max-tokens` - Set max output tokens
- [x] `--base-url` - Custom API endpoint
- [x] `--api-key` - Override API key
- [x] `--tool` - Specify individual tools (multiple allowed)

#### Validation System Utilities
- [x] `ValidationResult` Pydantic model
- [x] Field validators for data consistency
- [x] `validate_model_and_api_key()` function
- [x] `display_provider_info()` for status table
- [x] `display_validation_error()` with Rich panels for provider-specific error messages
- [x] Default model selection per provider

#### Interactive Mode Slash Commands
- [x] Command system implementation (`_handle_repl_command()`)
- [x] `/help` command with usage display
- [x] `/stream` status check
- [x] `/stream on` and `/stream off` toggle
- [x] `/exit` and `/quit` exit commands
- [x] Dynamic streaming state management
- [x] Enhanced welcome message with command list
- [x] Provider validation before session start
- [x] Model validation with API key checking

#### Type Safety Improvements
- [x] Added `py.typed` marker for PEP 561
- [x] Full type hint coverage for new code
- [x] Pydantic validation for validation results

#### Dependency Management
- [x] Modular dependency groups (test, lint, dev, publish)
- [x] Faster CI with targeted installations
- [x] Better contributor experience

#### Documentation
- [x] Google provider comprehensive docstrings
- [x] Provider utils documentation
- [x] Enhanced `.env.example`

**Deliverables:**
- ✅ Interactive mode with slash commands
- ✅ Dynamic streaming toggle
- ✅ Type-safe validation system
- ✅ Modular dependencies
- ✅ Comprehensive tests

## 2.3 Web Tools (Week 11)

**Duration**: 1 week (Days 78-84)

**Status**: 📋 Planned

**Goal**: Enable agents to search and fetch web content

### Day 78-79: Web Search Tool

#### Implementation
- [ ] **`allos/tools/web/search.py`**
  - `WebSearchTool` class
  - Multiple search engine support:
    - DuckDuckGo (default, no API key)
    - Brave Search API
    - Google Custom Search
    - SearXNG (self-hosted)
  - Result parsing and cleaning
  - Configurable result limits
  - Safe search filtering
  - Result ranking

```python
from ..base import BaseTool, tool, ToolParameter, ToolPermission

@tool
class WebSearchTool(BaseTool):
    """Search the web for information"""

    name = "web_search"
    description = "Search the web and return relevant results"
    permission = ToolPermission.ASK_USER

    parameters = [
        ToolParameter(
            "query",
            "string",
            "Search query",
            required=True
        ),
        ToolParameter(
            "num_results",
            "integer",
            "Number of results (1-10)",
            required=False
        )
    ]

    def execute(self, query: str, num_results: int = 5) -> dict:
        """Execute web search"""
        try:
            # Use DuckDuckGo as default (no API key needed)
            from duckduckgo_search import DDGS

            ddgs = DDGS()
            results = ddgs.text(query, max_results=num_results)

            # Format results
            formatted = []
            for r in results:
                formatted.append({
                    'title': r['title'],
                    'url': r['href'],
                    'snippet': r['body']
                })

            return {
                "success": True,
                "query": query,
                "results": formatted,
                "count": len(formatted)
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
```

#### Features
- [ ] Multiple search engines
- [ ] Rate limiting
- [ ] Caching results
- [ ] Result deduplication
- [ ] Safe search mode

#### Testing
- [ ] Unit tests with mocked search
- [ ] Integration tests with real searches
- [ ] Rate limit testing
- [ ] Error handling

### Day 80-81: Web Fetch Tool

#### Implementation
- [ ] **`allos/tools/web/fetch.py`**
  - `WebFetchTool` class
  - HTTP/HTTPS fetching with requests
  - HTML parsing with BeautifulSoup
  - Markdown conversion with html2text
  - PDF text extraction with PyPDF2
  - Image detection and description (future)
  - Rate limiting per domain
  - Respect robots.txt
  - User agent configuration
  - Timeout handling

```python
import requests
from bs4 import BeautifulSoup
import html2text

@tool
class WebFetchTool(BaseTool):
    """Fetch and parse web page content"""

    name = "web_fetch"
    description = "Fetch content from a URL and convert to text"
    permission = ToolPermission.ASK_USER

    parameters = [
        ToolParameter(
            "url",
            "string",
            "URL to fetch",
            required=True
        ),
        ToolParameter(
            "format",
            "string",
            "Output format: 'text', 'markdown', 'html'",
            required=False
        )
    ]

    def execute(self, url: str, format: str = "markdown") -> dict:
        """Fetch web content"""
        try:
            # Validate URL
            if not url.startswith(('http://', 'https://')):
                return {
                    "success": False,
                    "error": "Invalid URL scheme"
                }

            # Fetch with timeout
            response = requests.get(
                url,
                timeout=10,
                headers={'User-Agent': 'AllosAgent/1.0'}
            )
            response.raise_for_status()

            # Detect content type
            content_type = response.headers.get('content-type', '')

            if 'text/html' in content_type:
                # Parse HTML
                soup = BeautifulSoup(response.text, 'html.parser')

                # Remove scripts and styles
                for tag in soup(['script', 'style', 'nav', 'footer']):
                    tag.decompose()

                if format == 'markdown':
                    h = html2text.HTML2Text()
                    h.ignore_links = False
                    content = h.handle(str(soup))
                elif format == 'text':
                    content = soup.get_text(separator='\n', strip=True)
                else:
                    content = str(soup)

                return {
                    "success": True,
                    "url": url,
                    "content": content,
                    "title": soup.title.string if soup.title else None,
                    "format": format
                }

            elif 'application/pdf' in content_type:
                # Handle PDF (future: extract text)
                return {
                    "success": False,
                    "error": "PDF extraction coming soon"
                }

            else:
                # Return raw text
                return {
                    "success": True,
                    "url": url,
                    "content": response.text,
                    "format": "text"
                }

        except requests.Timeout:
            return {"success": False, "error": "Request timeout"}
        except requests.RequestException as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            return {"success": False, "error": f"Parse error: {str(e)}"}
```

#### Features
- [ ] HTML to Markdown conversion
- [ ] Text extraction and cleaning
- [ ] PDF text extraction
- [ ] Rate limiting
- [ ] robots.txt respect
- [ ] SSL verification
- [ ] Redirect handling
- [ ] Cookie management

### Day 82-83: Web Tool Integration & Testing

#### Integration
- [ ] Add web tools to ToolRegistry
- [ ] Update CLI to include web tools
- [ ] Add to tool configuration system
- [ ] Create tool categories (web, file, shell)

#### Testing
- [ ] **`tests/unit/test_web_tools.py`**
  - Mock HTTP responses
  - Test HTML parsing
  - Test markdown conversion
  - Test error handling

- [ ] **`tests/integration/test_web_tools.py`**
  - Real web searches
  - Real page fetching
  - Rate limit validation
  - robots.txt compliance

#### Security
- [ ] URL validation (prevent file://, javascript:)
- [ ] SSL certificate verification
- [ ] Maximum content size limits
- [ ] Timeout enforcement
- [ ] Domain blacklist support
- [ ] Content type validation

### Day 84: Documentation & Examples

#### Documentation
- [ ] **`docs/guides/web-tools.md`**
  - Web search guide
  - Web fetch guide
  - Search engine configuration
  - Best practices
  - Rate limiting guide
  - Security considerations

#### Examples
- [ ] **`examples/web_research.py`**
  - Research a topic
  - Compile findings
  - Write summary report

```python
"""
Web Research Agent Example

Demonstrates how to use web tools for research tasks.
"""

from allos import Agent, AgentConfig

def research_topic():
    """Research a topic and write a summary"""

    agent = Agent(AgentConfig(
        provider_name="anthropic",
        model="claude-sonnet-4-5",
        tool_names=["web_search", "web_fetch", "write_file"],
        auto_approve=False  # Ask permission for web access
    ))

    result = agent.run(
        "Research the latest developments in quantum computing. "
        "Search for recent articles, read the top 3, and write "
        "a summary report to quantum_computing_2025.md"
    )

    print(f"Research complete: {result}")

if __name__ == "__main__":
    research_topic()
```

- [ ] **`examples/competitive_analysis.py`**
  - Analyze competitor websites
  - Extract key information
  - Generate comparison report

- [ ] **`examples/fact_checking.py`**
  - Verify claims with web search
  - Cross-reference sources
  - Generate fact-check report

### Success Criteria

✅ Web search tool working with multiple engines

✅ Web fetch tool parsing HTML to markdown

✅ Rate limiting implemented

✅ Security measures in place

✅ All tests pass

✅ Documentation complete

✅ 3+ working examples

### Deliverables

- WebSearchTool with multi-engine support
- WebFetchTool with HTML parsing
- Comprehensive security measures
- Documentation and guides
- 3+ real-world examples
- Integration and E2E tests

---

## 2.4 Advanced Context Management (Week 12)

**Duration**: 1 week (Days 85-91)

**Status**: 📋 Planned

**Goal**: Handle large codebases and long conversations efficiently

> **Note**: MVP includes basic context window checking. This phase adds advanced optimization.

### Day 85-86: Context Compaction Strategies

#### Enhanced Compactor
- [ ] **`allos/context/compactor.py`** (enhance existing)
  - Importance scoring algorithm
  - Smart message retention
  - Summarization of old messages
  - Tool result compression
  - Provider-aware optimization

```python
class ContextCompactor:
    """Advanced context compaction strategies"""

    def compact(
        self,
        context: ConversationContext,
        target_tokens: int,
        provider: BaseProvider
    ) -> ConversationContext:
        """Compact context using multiple strategies"""

        # Strategy 1: Remove low-importance messages
        scored = self._score_importance(context.messages)

        # Strategy 2: Summarize old messages
        if self._should_summarize(scored):
            context = self._summarize_old_messages(context, provider)

        # Strategy 3: Compress tool results
        context = self._compress_tool_results(context)

        # Strategy 4: Keep essential messages
        context = self._retain_essential(context, target_tokens)

        return context

    def _score_importance(self, messages: List[Message]) -> List[Tuple[Message, float]]:
        """Score message importance"""
        scored = []
        for msg in messages:
            score = 0.0

            # System messages are critical
            if msg.role == MessageRole.SYSTEM:
                score = 1.0

            # Recent messages more important
            recency_bonus = self._recency_score(msg)
            score += recency_bonus

            # Messages with tool calls are important
            if msg.tool_calls:
                score += 0.3

            # Long messages might be important
            if len(msg.content or "") > 500:
                score += 0.2

            scored.append((msg, score))

        return scored
```

#### Features
- [ ] Importance scoring
- [ ] Summarization with LLM
- [ ] Tool result compression
- [ ] Message deduplication
- [ ] Context window optimization per provider

### Day 87-88: Context Analyzer

#### Implementation
- [ ] **`allos/context/analyzer.py`**
  - Conversation flow analysis
  - Key information extraction
  - Context optimization suggestions
  - Token usage analytics
  - Performance metrics

```python
class ContextAnalyzer:
    """Analyze conversation context for optimization"""

    def analyze(self, context: ConversationContext) -> AnalysisReport:
        """Analyze context and provide insights"""

        report = AnalysisReport()

        # Token usage breakdown
        report.total_tokens = context.total_tokens_used
        report.by_role = self._tokens_by_role(context)
        report.by_tool = self._tokens_by_tool(context)

        # Conversation flow
        report.turns = len([m for m in context.messages if m.role == MessageRole.USER])
        report.tool_calls = len([m for m in context.messages if m.tool_calls])

        # Optimization opportunities
        report.suggestions = self._generate_suggestions(context)

        return report

    def _generate_suggestions(self, context: ConversationContext) -> List[str]:
        """Suggest optimizations"""
        suggestions = []

        # Check for repetitive patterns
        if self._has_repetition(context):
            suggestions.append("Consider summarizing repetitive content")

        # Check for large tool results
        large_results = self._find_large_tool_results(context)
        if large_results:
            suggestions.append(f"Compress {len(large_results)} large tool results")

        # Check for old messages
        old_count = self._count_old_messages(context)
        if old_count > 10:
            suggestions.append(f"Summarize {old_count} old messages")

        return suggestions
```

#### Features
- [ ] Token usage breakdown
- [ ] Conversation pattern detection
- [ ] Optimization recommendations
- [ ] Cost analysis
- [ ] Performance tracking

### Day 89-90: Integration & Testing

#### Integration
- [ ] Integrate with Agent class
- [ ] Add automatic compaction triggers
- [ ] Add CLI flags for analysis
- [ ] Add configuration options

```yaml
# .allos/config.yaml
context:
  auto_compact: true
  compact_threshold: 0.8  # 80% of context window
  compaction_strategy: "smart"  # smart, aggressive, conservative
  keep_last_n_messages: 10
  summarize_old_messages: true
```

#### Testing
- [ ] **`tests/unit/test_compactor.py`**
  - Test importance scoring
  - Test summarization
  - Test compression
  - Test edge cases

- [ ] **`tests/integration/test_context_management.py`**
  - Test with large conversations
  - Test with multiple providers
  - Test performance
  - Validate token reduction

### Day 91: Documentation & Examples

#### Documentation
- [ ] **`docs/guides/context-management.md`**
  - Context compaction guide
  - Optimization strategies
  - Configuration options
  - Best practices
  - Performance tuning

#### Examples
- [ ] **`examples/large_codebase_analysis.py`**
  - Analyze large codebase
  - Handle context efficiently
  - Generate comprehensive report

### Success Criteria

✅ Context compaction reduces token usage by 50%+

✅ Importance scoring accurately retains key information

✅ Summarization maintains conversation coherence

✅ Performance acceptable (< 500ms overhead)

✅ All tests pass

### Deliverables

- Enhanced ContextCompactor with smart strategies
- ContextAnalyzer with insights
- Configuration options
- Documentation and guides
- Performance benchmarks

---

## 2.5 Configuration System (Week 13)

**Duration**: 1 week (Days 92-98)

**Status**: 📋 Planned

**Goal**: Make Allos easily configurable for teams

### Day 92-93: Configuration File Support

#### YAML Configuration
- [ ] **`allos/config/loader.py`**
  - Load `.allos/config.yaml`
  - Environment variable interpolation
  - Schema validation with Pydantic
  - Config merging (system → user → project)
  - Config inheritance

```python
from pydantic import BaseModel, Field
import yaml

class AllosConfig(BaseModel):
    """Allos configuration schema"""

    default: ProviderConfig
    tools: ToolsConfig
    context: ContextConfig
    providers: Dict[str, ProviderConfig]
    system_prompt: Optional[str] = None

    @classmethod
    def load(cls, path: Path = None) -> "AllosConfig":
        """Load configuration from file"""
        if path is None:
            path = cls._find_config_file()

        with open(path) as f:
            data = yaml.safe_load(f)

        # Interpolate environment variables
        data = cls._interpolate_env_vars(data)

        return cls(**data)
```

#### JSON Support
- [ ] Support `.allos/config.json` as alternative
- [ ] Automatic format detection
- [ ] JSON schema for validation

### Day 94: Configuration Features

#### Config Hierarchy
- [ ] **System config**: `/etc/allos/config.yaml`
- [ ] **User config**: `~/.config/allos/config.yaml`
- [ ] **Project config**: `.allos/config.yaml`
- [ ] **Environment-specific**: `.allos/config.dev.yaml`

#### Config Schema
```yaml
# .allos/config.yaml
default:
  provider: anthropic
  model: claude-sonnet-4-5
  temperature: 0.7
  max_tokens: 4096

tools:
  enabled:
    - read_file
    - write_file
    - edit_file
    - list_directory
    - shell_exec
    - web_search
    - web_fetch

  permissions:
    shell_exec: ask
    write_file: ask
    edit_file: ask
    web_fetch: ask
    read_file: always_allow
    list_directory: always_allow
    web_search: ask

  safe_auto_approve:
    - read_file
    - list_directory

context:
  auto_compact: true
  compact_threshold: 0.8
  strategy: smart
  keep_last_n: 10

system_prompt: |
  You are an expert software engineer.
  Write clean, tested, documented code.
  Follow best practices and coding standards.

providers:
  openai:
    api_key: ${OPENAI_API_KEY}
    base_url: null
    organization: null

  anthropic:
    api_key: ${ANTHROPIC_API_KEY}

  ollama:
    host: http://localhost:11434
    models:
      - qwen2.5-coder:7b
      - llama3.2:3b

  google:
    api_key: ${GOOGLE_API_KEY}
    project_id: ${GOOGLE_PROJECT_ID}

cli:
  default_session_dir: ~/.allos/sessions
  verbose: false
  color: true
```

### Day 95-96: Configuration CLI

#### Config Commands
- [ ] `allos config init` - Create template config
- [ ] `allos config validate` - Validate config file
- [ ] `allos config show` - Display current config
- [ ] `allos config edit` - Open in editor
- [ ] `allos config get <key>` - Get specific value
- [ ] `allos config set <key> <value>` - Set value

```bash
# Initialize config
allos config init

# Validate
allos config validate

# Show current configuration
allos config show

# Edit in default editor
allos config edit

# Get specific value
allos config get default.provider

# Set value
allos config set default.model claude-opus-4
```

### Day 97: Testing & Integration

#### Testing
- [ ] **`tests/unit/test_config_loader.py`**
  - Test YAML/JSON parsing
  - Test environment variable interpolation
  - Test config merging
  - Test validation

- [ ] **`tests/integration/test_config_integration.py`**
  - Test config loading in Agent
  - Test CLI config commands
  - Test config hierarchy

### Day 98: Documentation

#### Documentation
- [ ] **`docs/guides/configuration.md`**
  - Configuration file guide
  - All available options
  - Environment-specific configs
  - Best practices
  - Examples for different use cases

#### Config Templates
- [ ] Create config templates:
  - `templates/config/default.yaml`
  - `templates/config/dev.yaml`
  - `templates/config/prod.yaml`
  - `templates/config/team.yaml`

### Success Criteria

✅ YAML and JSON config files supported

✅ Environment variable interpolation works

✅ Config hierarchy and merging correct

✅ CLI config commands functional

✅ Validation with helpful error messages

✅ Documentation complete

### Deliverables

- Configuration loader with YAML/JSON support
- CLI configuration commands
- Config templates
- Documentation and guides
- Comprehensive tests

---

## 2.6 Plugin System Foundation (Week 14)

**Duration**: 1 week (Days 99-105)

**Status**: 📋 Planned

**Goal**: Enable community extensions without forking

### Day 99-100: Plugin Architecture

#### Base Plugin System
- [ ] **`allos/plugins/base.py`**
  - `BasePlugin` abstract class
  - Plugin lifecycle (init, load, unload)
  - Plugin metadata (name, version, author)
  - Plugin dependencies
  - Plugin hooks

```python
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any

class PluginMetadata(BaseModel):
    """Plugin metadata"""
    name: str
    version: str
    author: str
    description: str
    requires: List[str] = []
    provides: List[str] = []

class BasePlugin(ABC):
    """Base class for all plugins"""

    metadata: PluginMetadata

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize plugin"""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup plugin resources"""
        pass

    def on_agent_start(self, agent: "Agent") -> None:
        """Called when agent starts"""
        pass

    def on_agent_end(self, agent: "Agent") -> None:
        """Called when agent ends"""
        pass
```

#### Plugin Types
- [ ] **Tool Plugins**: Add new tools
- [ ] **Provider Plugins**: Add new providers
- [ ] **Hook Plugins**: Intercept events
- [ ] **Command Plugins**: Add CLI commands

### Day 101-102: Plugin Loader & Manager

#### Plugin Loader
- [ ] **`allos/plugins/loader.py`**
  - Auto-discover plugins
  - Load from directories
  - Dependency resolution
  - Plugin validation

```python
class PluginLoader:
    """Load and manage plugins"""

    def __init__(self):
        self.plugins: Dict[str, BasePlugin] = {}
        self.plugin_dirs = [
            Path.home() / ".allos" / "plugins",
            Path.cwd() / ".allos" / "plugins",
        ]

    def discover_plugins(self) -> List[PluginMetadata]:
        """Discover available plugins"""
        discovered = []

        for plugin_dir in self.plugin_dirs:
            if not plugin_dir.exists():
                continue

            for entry in plugin_dir.iterdir():
                if entry.is_dir() and (entry / "plugin.yaml").exists():
                    metadata = self._load_metadata(entry / "plugin.yaml")
                    discovered.append(metadata)

        return discovered

    def load_plugin(self, name: str, config: Dict = None) -> BasePlugin:
        """Load and initialize a plugin"""
        # Find plugin
        plugin_path = self._find_plugin(name)

        # Load plugin module
        plugin_module = self._load_module(plugin_path)

        # Instantiate plugin class
        plugin_class = plugin_module.get_plugin_class()
        plugin = plugin_class()

        # Initialize
        plugin.initialize(config or {})

        self.plugins[name] = plugin
        return plugin
```

#### Plugin Manager
- [ ] **`allos/plugins/manager.py`**
  - Enable/disable plugins
  - Plugin configuration
  - Plugin isolation
  - Plugin updates

### Day 103: Plugin Examples & Templates

#### Example Plugins
- [ ] **Tool Plugin Example**
  - Create sample database tool plugin
  - Show plugin structure
  - Demonstrate best practices

```python
# my_db_plugin/plugin.py
from allos.plugins.base import BasePlugin, PluginMetadata
from allos.tools.base import BaseTool, tool

class DatabasePlugin(BasePlugin):
    """Example database plugin"""

    metadata = PluginMetadata(
        name="database",
        version="1.0.0",
        author="Example Author",
        description="Database query tools",
        requires=["sqlalchemy"],
        provides=["query_database", "list_tables"]
    )

    def initialize(self, config):
        """Initialize database connection"""
        from sqlalchemy import create_engine
        db_url = config.get('database_url')
        self.engine = create_engine(db_url)

    def cleanup(self):
        """Close database connection"""
        self.engine.dispose()

    def get_tools(self) -> List[BaseTool]:
        """Return tools provided by this plugin"""
        return [
            QueryDatabaseTool(self.engine),
            ListTablesTool(self.engine)
        ]
```

#### Plugin Template
- [ ] Create plugin template generator
- [ ] `allos plugin create <name>` command
- [ ] Template includes:
  - Plugin structure
  - setup.py
  - Tests
  - Documentation
  - CI/CD

### Day 104: Testing & Integration

#### Testing
- [ ] **`tests/unit/test_plugin_loader.py`**
  - Test plugin discovery
  - Test plugin loading
  - Test dependency resolution

- [ ] **`tests/integration/test_plugin_system.py`**
  - Test real plugin loading
  - Test plugin isolation
  - Test plugin lifecycle

### Day 105: Documentation

#### Documentation
- [ ] **`docs/guides/plugins.md`**
  - Plugin system overview
  - Creating plugins
  - Plugin types
  - Distribution guide
  - Best practices

#### Developer Guide
- [ ] **`docs/contributing/creating-plugins.md`**
  - Plugin development guide
  - API reference
  - Testing plugins
  - Publishing plugins

### Success Criteria

- [ ] Plugin architecture implemented
- [ ] Plugin loader auto-discovers plugins
- [ ] Plugin manager handles lifecycle
- [ ] Example plugins working
- [ ] Plugin template generator functional
- [ ] Documentation complete

### Deliverables

- Plugin base classes and interfaces
- Plugin loader with auto-discovery
- Plugin manager
- Example plugin (database)
- Plugin template generator
- Comprehensive documentation

---

## Phase 2: Summary & Success Metrics

### Phase 2 Completion Criteria

- [x] **Native providers shipped**: OpenAI (Responses), Anthropic, Chat Completions, Ollama, Google, Cohere
- [x] **OpenAI-compatible aliases**: Registry-backed alias support via Chat Completions (broad ecosystem coverage)
- [x] **Streaming foundation**: Provider, Agent, and CLI stream paths implemented for core providers
- [x] **Metadata foundation**: MetadataBuilder + `agent.last_run_metadata` integration shipped
- [x] **Examples**: 20+ examples available in `examples/`
- [ ] **Web Tools**: Search/fetch implementation and tests
- [ ] **Context Management**: Advanced compaction/analyzer
- [ ] **Configuration**: YAML/JSON loader and CLI config commands
- [ ] **Plugins**: Plugin foundation and lifecycle management

### Technical Metrics (Current + Targets)

- [x] **Native Provider Count (current)**: 6 (`openai`, `anthropic`, `chat_completions`, `ollama`, `google`, `cohere`)
- [x] **Alias Coverage (current)**: 70+ OpenAI-compatible aliases in registry
- 🎯 **Test Coverage**: Maintain high coverage as new phase-2 modules land
- 🎯 **Context Efficiency**: Target 50%+ token reduction once compaction ships
- 🎯 **Performance**: Keep <2x latency vs direct API calls for equivalent flows
- 🎯 **Tool Count**: 10+ tools (5 existing + 5 new)

### Adoption Metrics (End of Phase 2)

- 🎯 **PyPI Downloads**: 1,000+ (from 374)
- 🎯 **GitHub Stars**: 50+ (from 6)
- 🎯 **Contributors**: 3+ active contributors
- 🎯 **Community Plugins**: 2+ community-created plugins
- 🎯 **Production Users**: 5+ teams using in production

---

## Phase 3: Advanced Tooling

**Timeline**: Weeks 15-20 (6 weeks)

**Status**: 📋 Planned

**Goal**: Best-in-class developer experience and advanced features

### Progress Breakdown

| Feature | Duration | Status | Completion |
|---------|----------|--------|------------|
| **3.1 Hooks System** | Week 15 | 📋 Planned | 0% |
| **3.2 Subagents & Delegation** | Week 16 | 📋 Planned | 0% |
| **3.3 MCP Support** | Week 17 | 📋 Planned | 0% |
| **3.4 Skills System** | Week 18 | 📋 Planned | 0% |
| **3.5 Advanced Monitoring** | Week 19 | 📋 Planned | 0% |
| **3.6 IDE Integrations** | Week 20 | 📋 Planned | 0% |

### 3.1 Hooks System (Week 15)

**Goal**: Deep customization without modifying core code

#### Implementation Highlights
- Event registration system
- Before/after hooks for tools and LLM calls
- Error handling hooks
- Custom hook plugins
- Async hook support

#### Key Features
```python
# Register hooks
agent.hooks.register('before_tool_call', log_tool_call)
agent.hooks.register('after_llm_call', track_tokens)
agent.hooks.register('on_error', notify_team)
```

### 3.2 Subagents & Delegation (Week 16)

**Goal**: Complex tasks with specialized agents

#### Implementation Highlights
- Subagent creation and management
- Task delegation protocols
- Result aggregation
- Agent communication
- Agent templates (code review, security, research)

#### Example Usage
```python
# Create specialized agents
code_reviewer = Agent(AgentConfig(
    provider_name="claude-opus-4",
    tool_names=["read_file", "write_file"],
    system_prompt="You are a code review expert..."
))

security_auditor = Agent(AgentConfig(
    provider_name="gpt-4",
    tool_names=["read_file", "shell_exec"],
    system_prompt="You are a security expert..."
))

# Main agent delegates to subagents
main_agent = Agent(AgentConfig(
    provider_name="claude-sonnet-4-5",
    subagents={
        "code_review": code_reviewer,
        "security_audit": security_auditor
    }
))
```

### 3.3 MCP Support (Week 17)

**Goal**: Model Context Protocol integration

#### Implementation Highlights
- MCP client implementation
- MCP server discovery
- Resource access protocols
- Tool exposure via MCP
- Integration with MCP ecosystem

#### Features
- Connect to MCP servers
- Use MCP resources
- Expose Allos tools as MCP tools
- MCP-based agent coordination

### 3.4 Skills System (Week 18)

**Goal**: Reusable, shareable agent capabilities

#### Implementation Highlights
- Skill definition format
- Skill composition
- Skill marketplace (future)
- Skill versioning
- Skill dependencies

#### Example Skills
```yaml
# skills/code_review.yaml
name: code_review
description: Comprehensive code review
tools:
  - read_file
  - write_file
  - list_directory
system_prompt: |
  You are an expert code reviewer...
steps:
  - Read all source files
  - Check for common issues
  - Review security concerns
  - Write detailed report
```

### 3.5 Advanced Monitoring (Week 19)

**Goal**: Production-grade observability

#### Features
- Token usage tracking
- Cost analysis per task
- Performance metrics
- Error rate monitoring
- Integration with observability platforms (Datadog, New Relic)
- Distributed tracing
- Logging aggregation

### 3.6 IDE Integrations (Week 20)

**Goal**: Seamless editor integration

#### Integrations
- VS Code extension
- JetBrains plugin
- Vim/Neovim plugin (future)
- Emacs integration (future)

#### Features
- Inline agent assistance
- Code suggestions
- Error explanations
- Refactoring support
- Documentation generation

---

## Phase 4: Enterprise & Scale

**Timeline**: Weeks 21-28 (8 weeks)

**Status**: 📋 Planned

**Goal**: Production-ready enterprise features

### Progress Breakdown

| Feature | Duration | Status |
|---------|----------|--------|
| **4.1 Production Safeguards** | Week 21 | 📋 Planned |
| **4.2 Cost Management** | Week 22 | 📋 Planned |
| **4.3 Advanced Error Recovery** | Week 23 | 📋 Planned |
| **4.4 Async Support** | Week 24 | 📋 Planned |
| **4.5 Team Collaboration** | Week 25 | 📋 Planned |
| **4.6 Deployment Options** | Weeks 26-27 | 📋 Planned |
| **4.7 Enterprise Security** | Week 28 | 📋 Planned |

### 4.1 Production Safeguards (Week 21)

#### Features
- Rate limiting per provider
- Automatic retries with backoff
- Circuit breakers
- Graceful degradation
- Health checks
- Rollback capabilities

### 4.2 Cost Management (Week 22)

#### Features
- Budget controls
- Cost alerts
- Usage quotas
- Provider cost optimization
- Cost attribution by user/team/project
- Cost forecasting

### 4.3 Advanced Error Recovery (Week 23)

#### Features
- Intelligent retry strategies
- Self-correction mechanisms
- Alternative provider fallback
- Error pattern detection
- Automatic bug reports

### 4.4 Async Support (Week 24)

#### Features
- Async Agent class
- Async tool execution
- Concurrent task handling
- Background processing
- Event-driven architecture

```python
# Async agent
agent = AsyncAgent(AgentConfig(...))
result = await agent.run("Create a FastAPI app")

# Concurrent tasks
results = await agent.run_many([
    "Task 1",
    "Task 2",
    "Task 3"
])
```

### 4.5 Team Collaboration (Week 25)

#### Features
- Shared sessions
- Team workspaces
- Agent sharing
- Collaborative debugging
- Knowledge sharing

### 4.6 Deployment Options (Weeks 26-27)

#### Docker Support
- Official Docker images
- Docker Compose templates
- Kubernetes manifests

#### Cloud Deployment
- AWS deployment guide
- GCP deployment guide
- Azure deployment guide
- Fly.io one-click deploy

### 4.7 Enterprise Security (Week 28)

#### Features
- SSO integration
- Role-based access control
- Audit logging
- Secrets management
- Compliance reporting (SOC 2, GDPR)
- Data residency controls

---

## Phase 5: Ecosystem Integration

**Timeline**: Weeks 29-36 (8 weeks)

**Status**: 🔮 Future

**Goal**: Framework compatibility and ecosystem leadership

### Progress Breakdown

| Feature | Duration | Status |
|---------|----------|--------|
| **5.1 Pydantic AI Integration** | Week 29-30 | 📋 Planned |
| **5.2 LangChain Compatibility** | Week 31 | 📋 Planned |
| **5.3 Smolagents Compatibility** | Week 32 | 📋 Planned |
| **5.4 CrewAI Integration** | Week 33 | 📋 Planned |
| **5.5 AutoGPT Compatibility** | Week 34 | 📋 Planned |
| **5.6 Framework Adapters** | Weeks 35-36 | 📋 Planned |

### 5.1 Pydantic AI Integration (Weeks 29-30)

#### Features
- Pydantic AI agent compatibility
- Use Allos providers in Pydantic AI
- Use Pydantic AI agents in Allos
- Shared tool ecosystem
- Unified configuration

### 5.2 LangChain Compatibility (Week 31)

#### Features
- LangChain tool adapters
- Use LangChain tools in Allos
- Use Allos tools in LangChain
- Chain composition
- Memory integration

### 5.3 Smolagents Compatibility (Week 32)

#### Features
- Smolagents adapter
- Provider interop
- Tool sharing
- Agent collaboration

### 5.4 CrewAI Integration (Week 33)

#### Features
- CrewAI agent integration
- Multi-agent orchestration
- Task delegation
- Result synthesis

### 5.5 AutoGPT Compatibility (Week 34)

#### Features
- AutoGPT plugin
- Task planning integration
- Memory system integration
- Tool ecosystem sharing

### 5.6 Framework Adapters (Weeks 35-36)

#### Universal Adapter Pattern
- Generic framework adapter
- Plug-and-play integration
- Community adapters
- Adapter marketplace

---

## Phase 6: Innovation

**Timeline**: Ongoing (Weeks 37+)

**Status**: 🔮 Future

**Goal**: Cutting-edge features and research collaborations

### 6.1 Multi-Modal Support

#### Features
- Image analysis (vision models)
- Image generation integration
- Document understanding (PDF, images)
- Audio transcription
- Text-to-speech
- Video analysis (future)

### 6.2 Code Generation Templates

#### Features
- Project scaffolding
- Component generators
- Framework-specific templates
- Best practices automation

### 6.3 Agent Marketplace

#### Features
- Public agent registry
- Agent discovery
- One-click deployment
- Ratings and reviews
- Monetization (future)

### 6.4 Fine-tuning Support

#### Features
- Training data collection
- Fine-tuning pipeline
- Model evaluation
- A/B testing
- Performance optimization

### 6.5 Advanced Planning

#### Features
- Hierarchical planning
- Graph-based planning
- Constraint satisfaction
- Resource optimization
- Plan visualization

### 6.6 Evaluation Framework

#### Features
- Benchmark suite
- Task success metrics
- Cost efficiency tracking
- Quality scoring
- Comparison reports

---

## 🎯 Success Metrics by Phase

### Phase 2: Enhanced Features (Weeks 9-14)

- [x] **Native Providers**: 6 shipped
- [x] **Alias Provider Ecosystem**: broad OpenAI-compatible coverage
- [x] **Streaming + Metadata Foundations**: shipped
- [ ] **Context Efficiency:** 50% improvement
- [ ] **Config**
- [ ] **Additional Tools**


### Phase 3: Advanced Tooling (Weeks 15-20) - Targets
- [ ] **Community Plugins**: 20+
- [ ] **Skills Available**: 50+
- [ ] **Downloads**: 5,000+
- [ ] **GitHub Stars**: 200+
- [ ] **Contributors**: 10+

### Phase 4: Enterprise (Weeks 21-28) - Targets
- [ ] **Production Deployments**: 10+
- [ ] **Monthly Requests**: 1M+
- [ ] **Uptime**: 99.9%
- [ ] **Downloads**: 10,000+
- [ ] **GitHub Stars**: 500+

### Phase 5: Ecosystem (Weeks 29-36) - Targets
- [ ] **Framework Integrations**: 5+
- [ ] **Community Tools**: 100+
- [ ] **Downloads**: 50,000+
- [ ] **GitHub Stars**: 1,000+
- [ ] **Contributors**: 50+

### Phase 6: Innovation (Ongoing) - Long-term Targets
- [ ] **Industry Recognition**: Conference talks
- [ ] **Community**: 500+ contributors
- [ ] **Downloads**: 100,000+
- [ ] **GitHub Stars**: 10,000+
- [ ] **Standard**: De facto framework for AI agents

---

## 📝 Daily Log Template

Keep a development log to track progress:

```markdown
## Day X - [Date]

### Completed
- [ ] Task 1
- [ ] Task 2

### In Progress
- [ ] Task 3

### Blockers
- Issue 1: Description and workaround

### Tomorrow
- [ ] Next task

### Notes
- Any insights or learnings
```

---

## 🎓 Learning Resources

### For Phase 2
- **Ollama Docs**: https://ollama.ai/docs
- **Ollama Python Library**: https://github.com/ollama/ollama-python
- **Google AI Studio**: https://ai.google.dev/
- **Brave Search API**: https://brave.com/search/api/
- **DuckDuckGo Search**: https://github.com/deedy5/duckduckgo_search

### For Phase 3
- **MCP Specification**: https://modelcontextprotocol.io/
- **Pydantic AI**: https://ai.pydantic.dev/
- **OpenTelemetry**: https://opentelemetry.io/

### For Phase 4
- **Docker**: https://docs.docker.com/
- **Kubernetes**: https://kubernetes.io/docs/
- **AWS ECS**: https://aws.amazon.com/ecs/

### For Phase 5
- **LangChain**: https://python.langchain.com/
- **Smolagents**: https://huggingface.co/docs/smolagents
- **CrewAI**: https://docs.crewai.com/

---

## 🆘 Getting Help

If you get stuck during Post-MVP development:

1. Check this roadmap for guidance
2. Review MVP_ROADMAP.md for reference patterns
3. Look at existing provider/tool implementations
4. Test with simple examples first
5. Add comprehensive logging
6. Consult relevant framework documentation
7. Ask in GitHub Discussions

<!-- ---

## 🎉 Celebrating Milestones

- **Phase 2 Complete**: Essential features shipped! 🎯
- **Phase 3 Complete**: Advanced tooling ready! 🛠️
- **Phase 4 Complete**: Enterprise-ready! 🏢
- **Phase 5 Complete**: Ecosystem leader! 🌟
- **Phase 6 Ongoing**: Innovation continues! 🚀 -->

---

## 🔄 Roadmap Updates

This POST-MVP roadmap will be updated:
- **Bi-weekly**: Based on development progress
- **Monthly**: Based on community feedback
- **Quarterly**: Based on ecosystem changes
- **After Phase Completion**: Based on learnings

### Version History
- **v1.0** (Current) - November 19, 2025 - Initial POST-MVP roadmap

---

## 💬 Community Involvement

### How to Contribute to Post-MVP

1. **Pick a Feature**: Choose from Phase 2 or Phase 3
2. **Discuss First**: Open a GitHub Discussion
3. **Create RFC**: For major features
4. **Submit PR**: Follow CONTRIBUTING.md
5. **Iterate**: Based on review feedback

### Priority Features for Community

- 🔥 **High Priority**: Web tools, context management, configuration loader/CLI, plugin system
- 🌟 **Good First Issues**: Documentation alignment, examples expansion, alias-provider smoke tests
- 🎯 **Advanced**: MCP enhancements, subagents, async support

---

<div align="center">

**Building the future of AI agents, together** 🚀

Progress: ██████████░░░░░░░░░░░░░░░░░░░░░░ 25% (Phase 2 of 6)

[Back to README](./README.md) • [MVP Roadmap](./MVP_ROADMAP.md) • [Full Roadmap](./ROADMAP.md) • [Contributing](./.github/CONTRIBUTING.md)

---

*Created: November 19, 2025*

*Last Updated: February 20, 2026*

*Next Review: March 1, 2026*

</div>
