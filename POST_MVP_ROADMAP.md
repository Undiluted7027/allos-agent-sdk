# Allos Agent SDK - POST-MVP Roadmap

> **Goal**: Transform Allos from a working MVP into the most flexible, powerful, and developer-friendly agentic SDK
> **Timeline**: Weeks 9-36+ (28+ weeks of active development)
> **Current Status**: MVP Launched (Nov 8, 2025), Ready for Phase 2

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
| **Phase 2: Enhanced Features** | Essential capabilities | Weeks 9-14 | 🎯 Current | 0% |
| **Phase 3: Advanced Tooling** | Developer experience | Weeks 15-20 | 📋 Planned | 0% |
| **Phase 4: Enterprise & Scale** | Production features | Weeks 21-28 | 📋 Planned | 0% |
| **Phase 5: Ecosystem Integration** | Framework compatibility | Weeks 29-36 | 🔮 Future | 0% |
| **Phase 6: Innovation** | Cutting-edge features | Ongoing | 🔮 Future | 0% |

**Overall Post-MVP Progress**: 0% (0/6 phases complete)

**Current Date**: November 19, 2025 (11 days post-launch)

---

## 📈 MVP Launch Results (Nov 8 - Nov 19, 2025)

**11 Days Post-Launch Performance**:
- 🌟 **GitHub**: 6 stars
- 💼 **LinkedIn**: 39 likes, 8 comments
- 🔴 **Reddit**: 18 upvotes, 8 comments (across threads)
- 📝 **Dev.to**: 11 likes, 3 comments, 162 views, +264 followers
- 📰 **Hashnode**: 8 views (all-time)
- 📖 **Medium + Towards AI**: 206 presentations, 49 views, 12 reads
- 🎥 **YouTube**: 24 views (unlisted demo)
- 📰 **HackerNoon**: Story in editorial
- 📦 **PyPI**: 374 downloads
- 🐦 **X.com**: No reach/views

**Key Insights**:
- Dev.to showing strongest engagement (+264 followers)
- Medium showing good reach (206 presentations)
- PyPI downloads indicate real usage (374 in 11 days)
- Need to improve X.com strategy for Phase 2 launch

---

## Phase 2: Enhanced Features

**Timeline**: Weeks 9-14 (6 weeks)

**Status**: 🎯 Current Phase (Starting Soon)

**Focus**: Essential capabilities for production use

### Goals
- Add local model support (Ollama)
- Expand provider ecosystem (Google, Cohere, Bedrock)
- Implement web tools (search, fetch)
- Build advanced context management
- Create configuration system
- Establish plugin architecture foundation

### Progress Breakdown

| Feature | Duration | Status | Completion |
|---------|----------|--------|------------|
| **2.0 Chat Completions Provider** | Days 57-59 | 📋 Planned | 0% |
| **2.1 Ollama Provider** | Days 60-70 | 📋 Planned | 0% |
| **2.2 Additional Providers** | Week 10 | 📋 Planned | 0% |
| **2.3 Web Tools** | Week 11 | 📋 Planned | 0% |
| **2.4 Advanced Context** | Week 12 | 📋 Planned | 0% |
| **2.5 Configuration System** | Week 13 | 📋 Planned | 0% |
| **2.6 Plugin System** | Week 14 | 📋 Planned | 0% |

---

## 🏗️ Important Architectural Note

### OpenAI API Architecture in Allos

**Current State**: Allos uses OpenAI's **Responses API** (released July-August 2025) for the OpenAI provider to support all advanced capabilities including native multi-turn tool calling, streaming, and other features.

**Challenge**: Many OpenAI-compatible providers use the older **Chat Completions API**, not the Responses API:
- ✅ **Together AI**: Chat Completions API compatible
- ✅ **Anyscale**: Chat Completions API compatible (certain endpoints)
- ⚠️ **Ollama**: Has its own Python library AND OpenAI-compatible Chat Completions endpoint

**Solution**: Create a separate `ChatCompletionsProvider` that implements the Chat Completions API interface. This provider can be:
1. Used directly with OpenAI's Chat Completions endpoint
2. Extended/configured for OpenAI-compatible APIs (Together AI, Anyscale)
3. Referenced as an alternative approach (Ollama will use native Python library)

### Provider API Strategy

```
OpenAI Ecosystem:
├── OpenAIProvider (Responses API) ← Current, for OpenAI with advanced features
└── ChatCompletionsProvider (Chat Completions API) ← New, for OpenAI-compatible services

Provider Implementations:
├── OllamaProvider → Uses ollama Python library (native)
├── TogetherAIProvider → Extends ChatCompletionsProvider
├── AnyscaleProvider → Extends ChatCompletionsProvider
├── GoogleProvider → Uses google-generativeai library
├── CohereProvider → Uses cohere library
├── BedrockProvider → Uses boto3
└── AzureOpenAIProvider → Extends OpenAIProvider (Responses API)
```

**Impact on Roadmap**: We'll implement the `ChatCompletionsProvider` base class first (Days 57-59), then proceed with provider-specific implementations.

### Decision Guide: Which API to Use?

**Use OpenAI Responses API (`OpenAIProvider`) when:**
- Using OpenAI's latest models
- Need advanced multi-turn tool calling
- Want best OpenAI feature support
- Provider: `openai` with models like `gpt-4o`, `gpt-4`

**Use Chat Completions API (`ChatCompletionsProvider`) when:**
- Using OpenAI-compatible services (Together AI, Anyscale)
- Need broad compatibility
- Working with standard function calling
- Provider: `together`, `anyscale`, or custom endpoint

**Use Native Libraries when:**
- Provider has its own Python SDK (Ollama, Google, Cohere, Bedrock, Anthropic)
- Best feature support from provider
- Optimal performance
- Provider-specific capabilities needed

---

## 2.0 Chat Completions Provider (Days 57-59)

**Duration**: 3 days (Days 57-59)

**Status**: 📋 Planned

**Goal**: Create reusable Chat Completions API provider for OpenAI-compatible services

**Motivation**: Enable compatibility with Together AI, Anyscale, and provide alternative for testing

### Day 57: Chat Completions Architecture & Design

#### Research & Design
- [ ] Study Chat Completions API vs Responses API differences
- [ ] Document API contract differences
- [ ] Design provider interface
- [ ] Plan backward compatibility strategy
- [ ] Create comparison matrix

#### Key Differences

| Feature | Responses API | Chat Completions API |
|---------|---------------|---------------------|
| Endpoint | `/v1/responses` | `/v1/chat/completions` |
| Tool Calling | Native multi-turn | Function calling |
| Message Format | Response items | Messages array |
| Streaming | Response events | SSE chunks |
| Tool Results | Structured | Function call messages |

### Day 58: Chat Completions Provider Implementation

#### Implementation
- [ ] **`allos/providers/chat_completions.py`**
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

        # Allow custom base_url for OpenAI-compatible APIs
        self.client = openai.OpenAI(
            api_key=api_key or os.getenv('OPENAI_API_KEY'),
            base_url=base_url  # None = OpenAI, custom = compatible service
        )

        # Detect context window
        self.context_window = self._detect_context_window(model)

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
                    message['function_call'] = {
                        'name': msg.tool_calls[0].name,
                        'arguments': json.dumps(msg.tool_calls[0].arguments)
                    }

                chat_messages.append(message)

            elif msg.role == MessageRole.TOOL:
                # Tool results as function responses
                chat_messages.append({
                    'role': 'function',
                    'name': msg.name,
                    'content': msg.content
                })

        return chat_messages

    def _convert_tools_to_functions(self, tools: List[dict]) -> List[dict]:
        """Convert Allos tools to OpenAI functions format"""
        functions = []

        for tool in tools:
            functions.append({
                'name': tool['name'],
                'description': tool['description'],
                'parameters': tool['parameters']
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

#### Features
- [ ] Support for custom base_url
- [ ] Function calling (OpenAI-style tool calling)
- [ ] Message format conversion
- [ ] Streaming support
- [ ] Error handling
- [ ] Token counting

### Day 59: Testing & Documentation

#### Testing
- [ ] **`tests/unit/test_chat_completions_provider.py`**
  - Mock OpenAI client
  - Test message conversion
  - Test function calling
  - Test with custom base_url
  - Test error handling

#### Integration Tests
- [ ] Test with real OpenAI Chat Completions endpoint
- [ ] Test with Together AI (if API key available)
- [ ] Compare behavior with ResponsesAPI provider

#### Documentation
- [ ] **`docs/providers/chat-completions.md`**
  - When to use Chat Completions vs Responses API
  - Configuration for different services
  - Limitations compared to Responses API
  - Migration guide

```markdown
# Chat Completions Provider

## Overview

The `ChatCompletionsProvider` implements OpenAI's Chat Completions API, which is widely
supported by OpenAI-compatible services like Together AI and Anyscale.

## When to Use

**Use ChatCompletionsProvider for:**
- Together AI
- Anyscale Endpoints
- Other OpenAI-compatible APIs
- Testing compatibility

**Use OpenAIProvider (Responses API) for:**
- OpenAI with advanced features
- Native multi-turn tool calling
- Latest OpenAI capabilities

## Configuration

```python
# OpenAI Chat Completions
provider = ChatCompletionsProvider(
    model="gpt-4",
    api_key=os.getenv('OPENAI_API_KEY')
)

# Together AI
provider = ChatCompletionsProvider(
    model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    api_key=os.getenv('TOGETHER_API_KEY'),
    base_url="https://api.together.xyz/v1"
)

# Anyscale
provider = ChatCompletionsProvider(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    api_key=os.getenv('ANYSCALE_API_KEY'),
    base_url="https://api.endpoints.anyscale.com/v1"
)
```

## Limitations

Compared to the Responses API provider:
- Simpler function calling (not native multi-turn)
- No response streaming events (uses SSE chunks)
- Different message format handling

For most use cases, these limitations are not significant.


### Success Criteria

✅ Chat Completions provider working with OpenAI

✅ Custom base_url configuration works

✅ Function calling (tool calling) implemented

✅ Message format conversion correct

✅ Tests pass (unit and integration)

✅ Documentation complete

### Deliverables

- ChatCompletionsProvider class
- Support for custom base_url
- Function calling implementation
- Comprehensive tests
- Documentation and comparison guide

---

## 2.1 Ollama Provider (Days 60-70)

**Duration**: 11 days (Days 60-70, adjusted from original 57-70)

**Status**: 📋 Planned

**Goal**: Enable completely local, private AI agents

**Motivation**: Users want to run agents offline without API costs or privacy concerns

**Implementation Note**: Ollama will use its **native Python library** (`ollama` package), NOT the OpenAI-compatible endpoint, for best feature support and native tool calling.

### Day 60-61: Ollama Provider Core (adjusted from Day 57-58)

#### Research & Design
- [ ] Study Ollama Python library API documentation
- [ ] Design provider interface to match BaseProvider
- [ ] Plan token counting strategy (model-specific)
- [ ] Design context window detection per model
- [ ] Plan streaming implementation
- [ ] Note: Using native `ollama` library, NOT OpenAI-compatible endpoint

#### Implementation
- [ ] **`allos/providers/ollama.py`**
  - `OllamaProvider` class extending `BaseProvider`
  - Connection to local Ollama server (default: localhost:11434)
  - Support for custom endpoints via environment variable
  - Model availability checking
  - Context window detection per model family
  - Register with `@provider` decorator
  - Options like `temperature` and `num_predict` are passed as `options` parameter in model calls. Like so:

```python
from ollama import Client
client = Client()
response = client.chat(
    model='<model_name>',
    messages=[{'role': 'user', 'content': 'Say this is a test.'}],
    options={'temperature': 0.5} # Passed as options
)
```


```python
# Key implementation details for Ollama
# NOTE: Using native ollama library, NOT OpenAI-compatible Chat Completions API
from typing import Set
from ollama import Client
from ..base import BaseProvider, Message, ProviderResponse, ToolCall

# Params that can be passed as options
OLLAMA_SUPPORTED_OPTIONS: Set = {
    'mirostat', 'mirostat_eta', 'mirostat_tau', 'num_ctx', 'repeat_last_n', 'repeat_penalty', 'temperature', 'seed', 'stop', 'num_predict', 'top_k', 'top_p', 'min_p'
}

OLLAMA_TOOL_SUPPORTED_MODELS: Set = {'qwen3-vl:32b-instruct', 'granite3.1-dense:2b-instruct-q4_1', 'granite3.1-moe:3b', 'llama4:scout', 'smollm2:360m-instruct-q2_K'} # I have more models. These are the models that natively support tool calling via Ollama

@provider
class OllamaProvider(BaseProvider):
    """
    Provider for Ollama local models.

    Uses Ollama's native Python library for:
    - Best feature support
    - Native tool calling (when model supports it)
    - Optimal performance
    - Direct model control

    Note: While Ollama has an OpenAI-compatible endpoint, we use the
    native library for better integration and feature support.
    """

    def __init__(self, model: str, **kwargs):
        super().__init__(model, **kwargs)
        # Connect to Ollama server (native client)
        self.client = Client(
            host=os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        )
        self._verify_model_available()

    def _verify_model_available(self):
        """Check if model is available locally"""
        models = self.client.list()
        available = [m['name'] for m in models['models']]
        if self.model not in available:
            raise ProviderError(
                f"Model {self.model} not available. "
                f"Run: ollama pull {self.model}"
            )

    def chat(self, messages: List[Message], tools: List[dict] = None) -> ProviderResponse:
        """
        Send chat request to Ollama using native Python library.

        This uses ollama.Client.chat(), not OpenAI-compatible endpoint.
        """
        # Convert messages to Ollama format
        ollama_messages = self._convert_to_ollama_messages(messages)

        # Prepare request with native Ollama options format
        request = {
            'model': self.model,
            'messages': ollama_messages,
            'options': {
                'temperature': self.temperature,  # Native options format
            }
        }

        # Add tools if provided (Ollama supports native tool calling)
        if tools:
            request['tools'] = tools

        # Make request using native client
        response = self.client.chat(**request)

        # Convert response
        return self._convert_ollama_response(response)
```

#### Testing
- [ ] **`tests/unit/test_ollama_provider.py`**
  - Mock Ollama client (native library)
  - Test model availability checking
  - Test message conversion
  - Test tool calling (native format)
  - Test error handling
  - Test context window detection

### Day 62-63: Ollama Tool Calling & Streaming (adjusted from Day 59-60)

#### Tool Calling Support
- [ ] **Native tool calling implementation**
  - Convert Allos tool format to Ollama format (native, not OpenAI functions)
  - Handle tool call responses from Ollama's native format
  - Extract tool call IDs
  - Support multiple tool calls in one turn

```python
def _convert_tools_to_ollama_format(self, tools: List[dict]) -> List[dict]:
    """
    Convert Allos tools to Ollama's native tool format.

    Note: This is Ollama's native format, not OpenAI functions format.
    Ollama supports tools natively with its own schema.
    """
    ollama_tools = []
    for tool in tools:
        ollama_tools.append({
            'type': 'function',
            'function': {
                'name': tool['name'],
                'description': tool['description'],
                'parameters': tool['parameters']
            }
        })
    return ollama_tools
```

#### Streaming Support
- [ ] **Implement streaming chat using native library**
  - Stream tokens as they're generated
  - Yield partial responses
  - Handle tool calls in streaming mode
  - Add `stream` parameter to chat method

```python
def chat_stream(self, messages: List[Message], tools: List[dict] = None):
    """
    Stream chat responses from Ollama using native Python library.

    Uses ollama.Client.chat() with stream=True.
    """
    request = {
        'model': self.model,
        'messages': self._convert_to_ollama_messages(messages),
        'stream': True,  # Native streaming support
        'options': {'temperature': self.temperature}
    }

    if tools:
        request['tools'] = tools

    # Stream using native client
    for chunk in self.client.chat(**request):
        yield self._convert_ollama_chunk(chunk)
```

#### Testing
- [ ] **Integration tests with real Ollama**
  - Test with llama3.2:1b (fast, small model)
  - Test tool calling with real model
  - Test streaming responses
  - Verify context window handling

### Day 64-65: Context Window Detection & Model Support (adjusted from Day 61-62)

### Day 64-65: Context Window Detection & Model Support (adjusted from Day 61-62)

#### Context Window Management
- [ ] **Model-specific context windows**
  - Create model family mappings (Llama, Mistral, Qwen, etc.)
  - Detect context window from model name
  - Add override via configuration
  - Implement token counting per model family

```python
# Model context windows
OLLAMA_CONTEXT_WINDOWS = {
    'llama3.2': 128000,
    'llama3.1': 128000,
    'llama3': 8192,
    'mistral': 32768,
    'mixtral': 32768,
    'qwen2.5': 32768,
    'codellama': 16384,
    'deepseek-coder': 16384,
    # Add more as needed
}

def _get_context_window(self, model: str) -> int:
    """Detect context window from model name"""
    for family, window in OLLAMA_CONTEXT_WINDOWS.items():
        if model.startswith(family):
            return window
    # Default to conservative 8K
    return 8192
```

#### Token Counting
- [ ] **Implement token counting**
  - Use tiktoken for estimation (by model family)
  - Fall back to word-based estimation
  - Cache token counts
  - Integrate with context manager

#### Popular Model Testing
- [ ] Test with popular models:
  - `llama3.2:3b` - Latest small Llama
  - `qwen2.5-coder:7b` - Code-focused
  - `mistral:7b` - General purpose
  - `deepseek-coder:6.7b` - Code specialist

### Day 66-67: Documentation & Examples (adjusted from Day 63-64)

#### Documentation
- [ ] **`docs/providers/ollama.md`**
  - Installation instructions (Ollama + models)
  - Configuration guide
  - Model recommendations
  - Tool calling guide
  - Streaming examples
  - Troubleshooting section
  - Performance tips

```markdown
# Ollama Provider

## Installation

1. Install Ollama:
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

2. Pull a model:
```bash
ollama pull qwen2.5-coder:7b
```

3. Use with Allos:
```bash
allos --provider ollama --model qwen2.5-coder:7b "Create a FastAPI app"
```

## Recommended Models

### For Coding
- `qwen2.5-coder:7b` - Best for code generation
- `deepseek-coder:6.7b` - Excellent coding assistant
- `codellama:13b` - Strong code understanding

### For General Tasks
- `llama3.2:3b` - Fast, capable
- `mistral:7b` - Well-rounded
- `llama3.1:8b` - Latest Llama
```

#### Examples
- [ ] **`examples/ollama_usage.py`**
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
        provider="ollama",
        model="qwen2.5-coder:7b",
        tools=["read_file", "write_file", "shell_exec"]
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
            provider=provider,
            model=model,
            tools=[]
        ))
        result = agent.run(task)
        print(f"Response: {result}")

if __name__ == "__main__":
    basic_ollama_usage()
    compare_providers()
```

- [ ] **`examples/local_vs_cloud.py`**
  - Side-by-side comparison
  - Cost analysis
  - Performance benchmarks
  - Privacy considerations

#### Update Main Documentation
- [ ] Update `README.md` provider table
- [ ] Update `docs/guides/providers.md`
- [ ] Add Ollama to quickstart guide
- [ ] Update architecture diagrams

### Day 68-69: Testing & Polish (adjusted from Day 65-66)

#### Comprehensive Testing
- [ ] **Unit tests**: Mock-based, no Ollama required
- [ ] **Integration tests**: Real Ollama server required
- [ ] **E2E tests**: Full agent workflows with Ollama
- [ ] **Performance tests**: Measure response times

#### Edge Cases
- [ ] Model not available (suggest `ollama pull`)
- [ ] Ollama server not running
- [ ] Network timeout handling
- [ ] Large context handling
- [ ] Streaming interruption
- [ ] Tool calling errors

#### Performance Optimization
- [ ] Connection pooling
- [ ] Request caching
- [ ] Model warm-up detection
- [ ] Memory usage monitoring

#### CLI Integration
- [ ] Add `--list-ollama-models` command
- [ ] Show model status in provider list
- [ ] Add Ollama-specific help text

### Day 70: Additional Model Families & Final Polish (adjusted from Days 67-70)

#### Expand Model Support
- [ ] **Llama family**:
  - llama3.2 (1b, 3b, 11b)
  - llama3.1 (8b, 70b, 405b)
  - llama3 (8b, 70b)

- [ ] **Mistral family**:
  - mistral (7b)
  - mixtral (8x7b, 8x22b)

- [ ] **Qwen family**:
  - qwen2.5 (0.5b-72b)
  - qwen2.5-coder (1.5b-32b)

- [ ] **Code models**:
  - deepseek-coder (1.3b-33b)
  - codellama (7b-34b)
  - starcoder2 (3b-15b)

#### Model Capabilities Matrix
- [ ] Create model comparison guide
- [ ] Document tool calling support by model
- [ ] Performance benchmarks
- [ ] Cost analysis (electricity vs API)

#### Final Polish
- [ ] Code review and refactoring
- [ ] Documentation review
- [ ] Example testing
- [ ] Performance profiling
- [ ] Security review

### Success Criteria

✅ Ollama provider fully functional

✅ Native tool calling works with supported models

✅ Streaming responses implemented

✅ Context window detection working

✅ 10+ popular models tested and documented

✅ All tests pass (unit, integration, E2E)

✅ Documentation complete with examples

✅ Performance acceptable (< 2x API latency)

### Deliverables

- Working Ollama provider with streaming support
- Tool calling for local models
- Comprehensive documentation
- 3+ working examples
- Model comparison guide
- Updated CLI with Ollama support
- Integration and E2E tests

### Example Usage After This Phase

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
    provider="ollama",
    model="qwen2.5-coder:7b",
    tools=["read_file", "write_file", "shell_exec"],
    temperature=0.7
))

result = agent.run("Review the code in src/ and suggest improvements")
print(result)

# Streaming mode
agent_stream = Agent(AgentConfig(
    provider="ollama",
    model="llama3.2:3b",
    tools=["read_file"]
))

for chunk in agent_stream.run_stream("Explain this codebase"):
    print(chunk, end='', flush=True)
```

---

## 2.2 Additional Providers (Week 10)

**Duration**: 1 week (Days 71-77)

**Status**: 📋 Planned

**Goal**: Expand provider ecosystem to 7+ providers

**Architecture Note**: Together AI and Anyscale will extend the `ChatCompletionsProvider` created in Days 57-59, leveraging the OpenAI-compatible Chat Completions API.

### Day 71-73: Google Gemini Provider

#### Core Implementation
- [ ] **`allos/providers/google.py`**
  - `GoogleProvider` class
  - Google AI Studio API integration
  - Vertex AI support (optional)
  - Native tool calling
  - Token counting with tiktoken
  - Context window: 2M tokens (Gemini 1.5 Pro)

```python
import google.generativeai as genai

@provider
class GoogleProvider(BaseProvider):
    """Provider for Google Gemini models"""

    def __init__(self, model: str = "gemini-1.5-pro", **kwargs):
        super().__init__(model, **kwargs)
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        self.client = genai.GenerativeModel(model)
        self.context_window = 2_000_000  # 2M tokens!
```

#### Features
- [ ] Support for Gemini 1.5 Pro, Flash
- [ ] Gemini 2.0 support (when available)
- [ ] Native tool calling
- [ ] Multimodal support (future)
- [ ] Vertex AI integration

#### Testing
- [ ] Unit tests with mocked responses
- [ ] Integration tests with real API
- [ ] Tool calling validation
- [ ] Context window testing

#### Documentation
- [ ] **`docs/providers/google.md`**
  - API key setup (AI Studio vs Vertex)
  - Model selection guide
  - Tool calling examples
  - Multimodal examples (future)

### Day 74: Cohere Provider

#### Implementation
- [ ] **`allos/providers/cohere.py`**
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
- [ ] Command R support
- [ ] Command R+ support
- [ ] Native tool calling
- [ ] Streaming support

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

### Day 76: OpenAI-Compatible Providers (Chat Completions API)

**Architecture Note**: These providers extend `ChatCompletionsProvider` with custom base URLs.

#### Together AI Provider
- [ ] **`allos/providers/together.py`**
  - Extends `ChatCompletionsProvider`
  - Pre-configured base_url: `https://api.together.xyz/v1`
  - Support for Together AI models (Llama, Mixtral, Qwen, etc.)
  - Auto-configured API key from environment

```python
from .chat_completions import ChatCompletionsProvider
from .registry import provider

@provider
class TogetherAIProvider(ChatCompletionsProvider):
    """
    Provider for Together AI using OpenAI-compatible Chat Completions API.

    Extends ChatCompletionsProvider with Together AI's endpoint.
    """

    def __init__(self, model: str, **kwargs):
        # Pre-configure base_url for Together AI
        super().__init__(
            model=model,
            api_key=os.getenv('TOGETHER_API_KEY'),
            base_url="https://api.together.xyz/v1",
            **kwargs
        )

        # Together AI specific context windows
        self.context_window = self._detect_together_context_window(model)

    def _detect_together_context_window(self, model: str) -> int:
        """Detect context window for Together AI models"""
        if 'llama-3.1' in model.lower():
            return 128_000
        elif 'mixtral' in model.lower():
            return 32_768
        elif 'qwen' in model.lower():
            return 32_768
        return 8_192  # Conservative default
```

**Usage Example**:
```python
# Together AI automatically uses Chat Completions API
agent = Agent(AgentConfig(
    provider="together",
    model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    tools=["read_file", "write_file"]
))
```

#### Anyscale Provider
- [ ] **`allos/providers/anyscale.py`**
  - Extends `ChatCompletionsProvider`
  - Pre-configured base_url: `https://api.endpoints.anyscale.com/v1`
  - Support for Anyscale Endpoints models

```python
from .chat_completions import ChatCompletionsProvider
from .registry import provider

@provider
class AnyscaleProvider(ChatCompletionsProvider):
    """
    Provider for Anyscale Endpoints using OpenAI-compatible Chat Completions API.

    Extends ChatCompletionsProvider with Anyscale's endpoint.
    """

    def __init__(self, model: str, **kwargs):
        # Pre-configure base_url for Anyscale
        super().__init__(
            model=model,
            api_key=os.getenv('ANYSCALE_API_KEY'),
            base_url="https://api.endpoints.anyscale.com/v1",
            **kwargs
        )
```

**Usage Example**:
```python
# Anyscale automatically uses Chat Completions API
agent = Agent(AgentConfig(
    provider="anyscale",
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    tools=["read_file", "shell_exec"]
))
```

#### Azure OpenAI
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
- [ ] Test all 7 providers (OpenAI, Anthropic, Ollama, Google, Cohere, Bedrock, Together AI, Anyscale, Azure)
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
  - Together AI (extends ChatCompletionsProvider)
  - Anyscale (extends ChatCompletionsProvider)
  - Azure OpenAI (extends OpenAIProvider)
- Provider comparison guide
- Setup documentation for each
- Updated examples showing provider switching
- Architecture documentation explaining API differences

---

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
        provider="anthropic",
        model="claude-sonnet-4-5",
        tools=["web_search", "web_fetch", "write_file"],
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

✅ Plugin architecture implemented

✅ Plugin loader auto-discovers plugins

✅ Plugin manager handles lifecycle

✅ Example plugins working

✅ Plugin template generator functional

✅ Documentation complete

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

✅ **Providers**: 9+ supported (OpenAI Responses API, OpenAI Chat Completions, Anthropic, Ollama, Google, Cohere, Bedrock, Together AI, Anyscale, Azure)

✅ **Chat Completions API**: Foundation provider for OpenAI-compatible services working

✅ **Local Models**: Ollama fully integrated with native Python library and streaming

✅ **Web Tools**: Search and fetch working

✅ **Context Management**: Advanced compaction and analysis

✅ **Configuration**: YAML/JSON config files supported

✅ **Plugins**: Foundation ready for community extensions

✅ **Documentation**: Complete for all new features

✅ **Tests**: Unit, integration, and E2E for all features

✅ **Examples**: 10+ working examples

### Adoption Metrics (End of Phase 2)

- 🎯 **PyPI Downloads**: 1,000+ (from 374)
- 🎯 **GitHub Stars**: 50+ (from 6)
- 🎯 **Contributors**: 3+ active contributors
- 🎯 **Community Plugins**: 2+ community-created plugins
- 🎯 **Production Users**: 5+ teams using in production

### Technical Metrics

- 🎯 **Provider Count**: 9+ working providers (OpenAI Responses, Chat Completions, Anthropic, Ollama, Google, Cohere, Bedrock, Together AI, Anyscale, Azure)
- 🎯 **Tool Count**: 10+ tools (5 existing + 5 new)
- 🎯 **Test Coverage**: Maintain 100%
- 🎯 **Context Efficiency**: 50%+ token reduction
- 🎯 **Performance**: <2x latency vs direct API calls

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
    provider="claude-opus-4",
    tools=["read_file", "write_file"],
    system_prompt="You are a code review expert..."
))

security_auditor = Agent(AgentConfig(
    provider="gpt-4",
    tools=["read_file", "shell_exec"],
    system_prompt="You are a security expert..."
))

# Main agent delegates to subagents
main_agent = Agent(AgentConfig(
    provider="claude-sonnet-4-5",
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
- ✅ **Providers**: 7+ supported
- ✅ **Tools**: 10+ available
- ✅ **Context Efficiency**: 50%+ improvement
- ✅ **Downloads**: 1,000+
- ✅ **GitHub Stars**: 50+

### Phase 3: Advanced Tooling (Weeks 15-20)
- ✅ **Community Plugins**: 20+
- ✅ **Skills Available**: 50+
- ✅ **Downloads**: 5,000+
- ✅ **GitHub Stars**: 200+
- ✅ **Contributors**: 10+

### Phase 4: Enterprise (Weeks 21-28)
- ✅ **Production Deployments**: 10+
- ✅ **Monthly Requests**: 1M+
- ✅ **Uptime**: 99.9%
- ✅ **Downloads**: 10,000+
- ✅ **GitHub Stars**: 500+

### Phase 5: Ecosystem (Weeks 29-36)
- ✅ **Framework Integrations**: 5+
- ✅ **Community Tools**: 100+
- ✅ **Downloads**: 50,000+
- ✅ **GitHub Stars**: 1,000+
- ✅ **Contributors**: 50+

### Phase 6: Innovation (Ongoing)
- ✅ **Industry Recognition**: Conference talks
- ✅ **Community**: 500+ contributors
- ✅ **Downloads**: 100,000+
- ✅ **GitHub Stars**: 10,000+
- ✅ **Standard**: De facto framework for AI agents

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

---

## 🎉 Celebrating Milestones

- **Phase 2 Complete**: Essential features shipped! 🎯
- **Phase 3 Complete**: Advanced tooling ready! 🛠️
- **Phase 4 Complete**: Enterprise-ready! 🏢
- **Phase 5 Complete**: Ecosystem leader! 🌟
- **Phase 6 Ongoing**: Innovation continues! 🚀

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

- 🔥 **High Priority**: Ollama provider, web tools, Google provider
- 🌟 **Good First Issues**: Documentation, examples, tool plugins
- 🎯 **Advanced**: MCP support, subagents, async support

---

<div align="center">

**Building the future of AI agents, together** 🚀

Progress: ████░░░░░░░░░░░░░░░░░░░░ 16.7% (Phase 2 of 6)

[Back to README](./README.md) • [MVP Roadmap](./MVP_ROADMAP.md) • [Full Roadmap](./ROADMAP.md) • [Contributing](./.github/CONTRIBUTING.md)

---

*Created: November 19, 2025*

*Last Updated: November 19, 2025*

*Next Review: December 1, 2025*

</div>
