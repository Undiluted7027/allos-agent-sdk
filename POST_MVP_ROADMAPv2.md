# Allos Agent SDK - POST-MVP Roadmap

> **Goal**: Transform Allos from a working MVP into the most flexible, powerful, and developer-friendly agentic SDK

## POST-MVP Vision
Build upon the solid MVP foundation to deliver:
1. **Universal Provider Support**: OpenAI, Anthropic, Ollama, Google, and more
2. **Rich Tool Ecosystem**: Web tools, database tools, and extensible plugins
3. **Production-Ready Features**: Advanced monitoring, error recovery, and scaling
4. **Best-in-Class DX**: Configuration files, IDE integrations, and templates
5. **Ecosystem Integration**: Pydantic AI, Smolagents, MCP, and framework compatibility

**Post-MVP Success Metric**: Allos becomes the go-to choice for building production-ready AI agents with any LLM provider.

---

For developments in MVP, refer [MVP Roadmap](./MVP_ROADMAP.md).

## 📊 Progress Tracker

| Phase | Status | Completion |
| ----- | ------ | ---------- |
| **Phase 1: Chat Completions** | ✅ Complete | 100% |
| **Phase 2: Streaming & Basic Metadata** | ✅ Complete | 100% |
| **Phase 3: Additional Native Providers** | ✅ Complete | 100% |
| **Phase 4: CLI & DX Enhancements** | 🏗️ In Progress | 75% |
| **Phase 5: Test Coverage Expansion** | 🏗️ In Progress | % |
| **Phase 6: Cloud-Based Providers** | 🗓️ Planned | |
| **Phase 7: Shell Tools** | 🗓️ Planned | |
| **Phase 8: Web-based & ToDo Related Tools** | 🗓️ Planned | |
| **Phase 9: Advanced Context Management** | 🗓️ Planned | |
| **Phase 10: Configuration System** | 🗓️ Planned | |
| **Phase 11: CLI Command Expansion & DX** | 🗓️ Planned | |
| **Phase 12: Plugin System** | 🗓️ Planned | |

**Overall Progress**: 25% (3/12 phases complete)

---

## Phase 1: Chat Completions Provider ✅ COMPLETE

**Status**: ✅ Complete

**Focus**: Chat Completions Provider for OpenAI and other provider extensibility

### Goals
- Implement Chat Completions Provider
- Support legacy OpenAI workflows
- Support OpenAI compatible provider endpoints


### Tasks

#### Chat Completions Provider
- [x] **`allos/providers/chat_completions.py`**
    - `ChatCompletionsProvider` class
    - Message format conversion
    - Tool calling support
    - Basic error handling
    - Register with decorator
    - Configurable `base_url` for compatibility
- [x] **`allos/agent/agent.py`**
    - Update `AgentConfig` class
        - `max_tokens`: support for providers that explicitly require this (Anthropic via chat completions)
        - `no_tools`: support for explictly marking agent run as requiring no tools (for `ollama_compat` via ChatCompletions)
        - `base_url`: support for custom endpoints
        - `api_key`: support for authentication (EXCLUDED FROM session files)
    - Update `save_session` method to not save api keys
    - Update `_get_llm_response` method to pass arguments for provider chat (provider specific)

#### CLI Updates to Support Provider Overrides
- [x] **`allos/cli/interactive.py`**
    - Update `start_interactive_session`, `_load_or_create_agent`, `_override_agent_config` to support new parameters of `AgentConfig`: `base_url`, `api_key`, `no_tools`, `max_tokens`.
- [x] **`allos/cli/main.py`**
    - `active-providers` command
    - LLM provider specific overrides
        - Base URL (`--base-url`)
        - API Key (`--api-key`)
        - Max Tokens (`--max-tokens`)
        - No Tools (`--no-tools`)
    - Validation functions to determine default model based on provider (`_determine_model`, `_validate_api_key`)
    - Helpers to abstract the agent run logic (`_initialize_agent`, `_update_agent_config`)

#### Registry Updates to Support OpenAI Compatible Providers
- [x] **`allos/providers/registry.py`**
    - `OPENAI_COMPATIBLE_PROVIDERS` dictionary with:
        - `env_var`: The environment variable SDK should look for to register this provider
        - `base_url`: Base URL to target for this provider (chat completions compatible)
        - `implementation`: `chat_completions`
        - `base_url_env_var`: The environment variable used to override `base_url`
        - `requires_auth`: For providers that won't require any API keys (like ollama local)
    - Update `provider` decorator to intelligently find and register native and OpenAI Chat Completions Compatible providers
    - `list_providers` lists all known registered and aliased providers
    - `get_env_var_name` helper to get the expected env var name for the provider. Used to show status for `--active-providers` flag in CLI
    - Update `get_provider` to support `base_url` override via `base_url_env_var` and support no api key via `requires_auth: False`

#### Testing & Polish
- [x] **Write tests**: `tests/e2e/test_cli.py`
    - Test active providers command
    - Test run command arguments (max tokens, base url, api key, no tools)
    - Test run command with overrides
    - Test interactive session with overrides
- [x] **Integration testing**:
    - Test Real API Endpoint with Chat Completions Provider
    - Add Chat Completions to `tests/integration/test_provider_switching.py`
- [x] **Unit Tests**:
    - Unit tests for `ChatCompletionsProvider`
    - Coverage expansion to cover agent overrides from CLI
    - Tests to cover new registry logic for OpenAI compatible providers

#### Documentation
- [x] **`docs/providers/chat-completions.md`**
    - In depth description about Chat Completions Provider
- [x] **`docs/providers/openai.md`**
    - References with Chat Completions and OpenAI API Comparison documents
- [x] **`docs/providers/ollama.md`**
    - Change from documentation in progress to support for local Ollama models via Chat Completions endpoint
- [x] **`docs/guides/openai-api-comparison.md`**
    - Comparison of Allos usage between OpenAI Responses API and OpenAI Chat Completions API
- [x] **`docs/guides/providers.md`**
    - Description about OpenAI Chat Completions and external provider compatibility
- [x] **`docs/reference/tool-api.md`**
    - Description about different formats and nuances across Anthropic, OpenAI Responses, and OpenAI Chat Completions APIs.
- [x] Update all other relevant docs like: `docs/index.md`, `docs/installation.md`, `docs/contributing/{development.md,testing.md}`, `docs/guides/{quickstart.md,configuration.md,agents.md}`, `docs/reference/{agent-api.md,cli-reference.md,tool-api.md}`

#### Examples
- [x] Update **`examples/cli_workflow.sh`** to demonstrate Chat Completions
- [x] **`examples/fast_inference.py`**
    - Benchmark comparing performance across providers
- [x] Update **`examples/local_models.py`** to mention about local model usage via Ollama and Chat Completions
- [x] **`examples/omnibus.py` and `examples/omnibus_cli.sh`**
    - Multiple providers
    - Single session
- [x] Update **`examples/provider_switching.py`**
- [x] **`examples/universal_chat.py`** ask one question and get answers from all providers

### Pending
- [ ] Token counting with tiktoken (VERIFICATION PENDING)
- [ ] Robust Error Handling (VERIFICATION PENDING)

*Note: This provider's compatibility matrix (as defined by `OPENAI_COMPATIBLE_PROVIDERS` in `allos/providers/registry.py`) was expanded further upon referencing other open-source projects like [Open Code](https://opencode.ai/).*

### Success Criteria
✅ Chat Completions provider working with OpenAI

✅ Custom base_url and api_key configuration works

✅ Function calling (tool calling) implemented

✅ Message format conversion correct

✅ Tests pass (unit, e2e and integration)

✅ Documentation complete

### Deliverables
- Working ChatCompletions provider
- Provider registry with additional provider and alias registration
- Comprehensive tests
- Manual testing verified

### Example Usage After This Phase (VERIFIED)
```python
from allos.providers import Message, MessageRole
from allos.providers.registry import ProviderRegistry

# Chat Completions Provider (requires GROQ API KEY)
chat_completion = ProviderRegistry.get_provider("groq", model="llama-3.1-8b-instant")
response = chat_completion.chat([Message(role=MessageRole.USER, content="Hello")])

print(response)
```

#### Changes Post Manual Testing
The above manual testing script revealed latent bugs in terms of Agent config pollution and reset. Therefore, the core changes made were:
1. If provider is changed, reset Agent Config to safe defaults.
2. Recreate agent instance based on updated config
3. Re-initialize tools to avoid state pollution

## Phase 2: Streaming & Basic Metadata ✅ COMPLETE

**Status**: ✅ Complete

**Focus**: Streaming support across OpenAI, ChatCompletions, Anthropic, basic metadata integration

### Goals
- Token-by-token streaming enabled for providers
- Metadata capture in agent
- Metadata integration with providers

### Tasks

#### Metadata Schema
- [x] **`allos/providers/metadata.py`**
  - [x] `Metadata` Pydantic model with comprehensive schema
    - `request_id`: `req_` + UUIDv4 (limited to 16 chars)
    - `status`
  - [x] `ModelInfo` Information about the model
  - [x] `MetadataBuilder` factory pattern
  - [x] `Usage` tracking (input/output tokens, cache, cost)
  - [x] `Latency` metrics (total duration, TTFT)
  - [x] `ToolInfo` and `ToolCallDetail` for tool tracking
  - [x] `TurnsInfo` and `TurnLog` for turn history
  - [x] `ProviderSpecific` for provider-unique fields
  - [x] `ProviderSpecificOpenAI` for OpenAI specific fields (`system_fingerprint`, `log_probs`)
  - [x] `QualitySignals` for finish reasons
  - [x] `MetadataBuilder` class
    - `__init__`: Initialize with provider_name, request_kwargs, and start_time
    - `_with_response_obj`: Assign Response object to builder's instance
    - `build`: Build Metadata
    - `_build_model_info`: Build ModelInfo Metadata
    - `_build_usage_info`: Build Usage Metadata
    - `_calculate_cost`: Calculate Cost (MAY REQUIRE MORE IDEATION)
    - `_build_latency_info`: Build Latency Metadata
    - `_build_tool_info`: Build ToolInfo from ToolCallDetail
    - `_build_quality_signals`: Build QualitySignal Metadata
    - `_build_provider_specific`: Build ProviderSpecific Metadata
    - `_build_sdk_info`: Build Allos SDK version related metadata

#### Basic Streaming Implementation
- [x] **`allos/providers/base.py`**
  - [x] `ProviderChunk` dataclass for streaming chunks
  - [x] `stream_chat()` abstract method in BaseProvider
  - [x] Streaming protocol definition
- [x] **Provider Specific Implementations:** `allos/providers/openai.py`, `allos/providers/anthropic.py`, `allos/providers/chat_completions.py`
  - [x] OpenAI `stream_chat()` with Responses API events
  - [x] Anthropic `stream_chat()` with streaming messages
  - [x] Chat Completions `stream_chat()` with SSE
- [x] **`allos/agent/agent.py`**
  - [x] `Agent.stream_run()` method
  - [x] **Private helpers**:
    - `Agent._process_streaming_iteration()` for processing a single streaming iteration, yielding chunks and accumulating state of iteration
    - `Agent._get_provider_stream` for getting the stream iterator from the provider with configured parameters
    - `Agent._update_chunk_metadata` for updating chunk metadata with tool call information for an agent iteration
    - `Agent._accumulate_usage_stats` for accumulating token usage and cost statistics across iterations
    - `Agent._update_context_after_streaming` for updating conversation context with streaming iteration results
    - `Agent._yield_final_aggregate_metadata` for yielding final aggregate metadata if tool calls were made
    - `Agent._execute_and_record_tools` for executing tool calls and recording their results in context
  - [x] **Metadata**:
    - `ToolExecutionResult = Tuple[Dict[str, Any], ToolCallDetail]`
    - `CumulativeState` TypedDict for tracking and metadata (tracks totals)
      - `all_tool_details`: list of tool call objects
      - `input_tokens`: Number of input tokens
      - `output_tokens`: Number of output tokens
      - `cost`: Cost of usage
      - `last_metadata`: Metadata from last token
      - `turn_history`: list of tool call turns
  - [x] `Agent._record_turn` method for recording a turn in turn history
  - [x] `Agent._create_aggregate_metadata` method for creating aggregated metadata for an entire agent run

#### Updates to CLI to Provider Streaming
- [x] **`allos/cli/main.py`**
    - `--stream` CLI flag
    - `run_agent_stream` helper function

#### Updates to Providers
- [x] **`allos/providers/registry.py`**
    - Use provider registry to get env var names for natively registered providers instead of static dict

#### Updates to Tools
- [x] **`allos/tools/base.py`**: Enable type coercion and refactor `validate_arguments`
    - `_check_required_arguments`: Check that all required arguments are provided
    - `_validate_and_coerce_types`: Validate and coerce argument types
    - `_coerce_type`: Attempt to coerce the value to expected type
    - `_coerce_boolean`: Coerce string to boolean (common with LLM calls)
    - `_is_valid_type`: Check if value matches expected type

#### Testing & Polish
- [x] **Update configuration**: `tests/conftest.py`
    - Create `mock_metadata_factory` to create a baseline, valid Metadata object for tests
    - Update `mock_provider_factory` to inject `mock_metadata_factory` into `ProviderResponse` using chat side_effect
    - Update `mock_tool_factory` to return tools of type `BaseTool` instead of `MagicMocks`
- [x] **E2E Test Update**: `tests/e2e/test_real_tasks.py`
    - Update tests to use mock_metadata_factory fixture
    - Create new test suite (`tests/e2e/test_cli.py`) to test streaming capabilities with CLI
    - Create new test suite (`tests/e2e/test_system_integrity`) to test for ciritical system boundaries, error recovery, and state persistence
- [x] **Integration Test Update**: `tests/integration/{test_agent_workflow.py,test_session.py,test_tool_execution.py}`
    - Update tests to use mock_metadata_factory fixture
- [x] **Unit Test Update**:
    - Update provider specific tests to assert metadata
    - Update agent related tests to assert metadata
    - Create new test suite for testing argument validation for tools (`tests/unit/test_tools.py`)
    - Create new test suite for testing metadata across all fields and potential edge cases (`tests/unit/providers/test_metadata.py`)
    - Add test cases to providers and agent to test for streaming

*Note: Integration tests were later on updated to use actual Real API calls throughout rather than mocking any part of it.*

#### Documentation
- [x] **Archive `docs/guides/guide.md` and `docs/guides/initial-architecture-design.md`** since they no longer describe the actual architecture and plan for Allos.
- [x] Update **`docs/contributing/adding-providers.md`** to reflect that new providers now require `get_context_window` and `stream_chat` methods to be implemented
- [x] Update **`docs/contributing/development.md`** to reflect updated development workflow
- [x] Update **`docs/reference/agent-api.md`** to reflect the new `stream_run` method
- [x] Update **`docs/reference/cli-reference.md`** to reflect the new `--stream` one-shot flag
- [x] Add documentation about metadata `docs/reference/metadata.md`:
    - Metadata access via Agent
    - Metadata access via provider (per-request)
    - Metadata Schema reference
- [x] Update **`docs/reference/provider-api.md`** to reflect addition of `stream_chat` method for providers and `ProviderChunk` datatype

#### Examples
- [x] Update `examples/fast_inference.py` to show metadata
- [x] Update `examples/provider_switching.py`, `examples/omnibus.py`, `examples/omnibus_cli.sh` for better output and DX

### Pending
- [ ] Additional metadata capture and augmentation across providers

### Experimental
- [ ] Decide which metadata options to support and which to mark experimental
- [ ] Modality support for metadata `calculate_modality_usage` of `allos/utils/modality_counter.py` (COULD BE DEFFERED AS WELL)

### Success Criteria
✅ Stream works across all core providers implemented so far

✅ Metadata for token usage is correctly captured

✅ CLI supports streaming mode

✅ Tests pass (unit, e2e and integration)

✅ Documentation complete

### Deliverables
- Working basic streaming implementation
- CLI with one shot streaming
- Metadata aggregation

### Example Usage After This Phase (VERIFIED)
```python
from allos import Agent, AgentConfig

# Create agent
agent = Agent(
    AgentConfig(
        provider_name="openai", model="gpt-4", tool_names=["read_file", "write_file"]
    )
)

result = agent.run("Create a main.py python script that prints hello world")
metadata = agent.last_run_metadata

if metadata:
    print(f"Tokens: {metadata.usage.total_tokens}")
    if metadata.usage.estimated_cost:
        print(f"Cost: ${metadata.usage.estimated_cost.total_usd}")
    print(f"Turns: {metadata.turns.total_turns}")
    print(f"Tools used: {metadata.tools.total_tool_calls}")
else:
    raise ValueError("Metadata was not recorded properly")
```

#### Changes Post Manual Testing
Edge cases and tests revealed that provider specific arguments if they were not returned in last run of agent, then they are not preserved. Therefore to fix this:
1. Provider-specific traits preserved from first run by capturing the traits early on `first_metadata` in `CumulativeState`
2. Backfill mechanism in `_create_aggregate_metadata`
3. `agent.last_run_metadata` is populated per run/stream completion
4. Provider specific metadata is populated by `_create_aggregate_metadata`

## Phase 3: Additional Native Providers ✅ COMPLETE

**Status**: ✅ Complete

**Focus**: Native provider implementations for Ollama, Google Gemini and Cohere

### Goals
- Implement Ollama native provider
- Implement Google Gemini / Google Vertex AI provider
- Implement Cohere native provider

### Tasks

#### Native Ollama Provider
- [x] Update **`pyproject.toml`**:
    - Add ollama dependency
    - ollama optional dependency marker
- [x] **`allos/providers/ollama.py`**
    - `OllamaProvider` class
    - `_model_context_window` and `_model_supports_tools_capability` member variables
    - env_var `OLLAMA_HOST` handling (default: localhost:11434)
    - Custom endpoint config via OLLAMA_HOST env_var
    - Model availability and context window detection by calling ollama client list method
    - `_extract_model_family` private helper to get base model family
    - `_model_supports_tool` private helper to check if model supports tools using model's metadata, fallback to static defined tool compatible models
    - `retrieve_context_length` helper function to extract context window of model using known paths, fallback to known statically provided model information
    - Tool calling support
    - Token counting with tiktoken (Ollama model families) `get_encoding_for_ollama_model` in `allos/utils/token_counter.py`
    - Context window detection
    - Error handling
    - Register with decorator `ollama`, Ollama via Chat Completions should become `ollama_compat` to signal compatiblility endpoint
    - Connection pooling and model hot reload
    - Streaming exponential backoff
- [x] **`allos/providers/utils.py`**
    - `ollama_running` to check if locally, ollama is running or not
- [x] **`allos/tools/base.py`**
    - Add ollama provider format to `to_provider_format` for tool calling
- [x] **`allos/providers/metadata.py`**
    - Add ollama specific metadata `ProviderSpecificOllama` with `warm_up` and `warm_up_duration_seconds` fields

##### Updates to CLI for api key and model detection
- [x] Create **`allos/cli/utils.py`**
    - `determine_model`: Select a default model based on provider if model is not specified
    - `validate_api_key`: Check if a valid API key is available for the provider
    - `validate_model_and_api_key`: Validate both model and API key config
- [x] **`allos/cli/main.py`**
    - `--list-ollama-models` command to list available local ollama models (`_format_size`, `_format_modified_date`, `_get_model_capabilities`, `_create_models_table`, `print_ollama_models`)

##### Testing & Polish
- [x] **Update configuration**: `tests/conftest.py`
    - Add pytest marker for Ollama real API tests `requires_ollama` that marks tests requiring Ollama
    - Refactor `_apply_integration_key_skips` to become more readable and add support for Ollama by using `ollama_running` function from `allos.providers.utils`
    - Default Ollama model fixture
    - Update `_create_metadata` function in `mock_metadata_factory` to be more robust and use actual datatypes
- [x] **Performance Tests**:
    - Add `performance` marker to `pyproject.toml`
    - Add pytest flag `--run-performance` to run only those tests that are marked for performance test since these are slow and write `_skip_performance_tests` function to skip performance related tests unless `--run-performance` flag is passed
    - Create new test suite for measuring response time, throughput, and resource usage (`tests/performance/test_ollama_performance.py`)
    - Create new test suite for comparing performance between ollama with model hot reload and streaming backoff vs ollama without those add ons (`tests/performance/test_ollama_improvements.py`)
    - Create new test suite for model warmup detection (`tests/performance/test_ollama_warmup.py`)
- [x] **E2E Test Updates**: `tests/e2e/test_cli.py`
    - Add tests for new api key and model detection logic
    - Add tests for `--active-providers` command
    - Add tests for validation for `agent.run`
    - Add tests for validation for interactive mode
    - Create test suite for `ollama` focused workflows (`tests/e2e/test_ollama_workflows.py`) and mock both Ollama Client and `ollama_running` function
- [x] **Integration Tests**:
    - `tests/integration/providers/test_ollama_real.py`
      - Simple chat
      - Tool calling
    - `tests/integration/test_provider_switching.py`
        - Add ollama provider in `parameters_to_test`
    - `tests/integration/test_session.py`
        - Add `ollama` specific patches `allos.providers.ollama.OllamaProvider.chat` and `allos.providers.ollama.OllamaProvider._verify_model_available`
        - Add ollama provider parameter to `test_session_save_and_load_with_filesystem`
- [x] **Unit Tests**:
    - Add tests for missing ollama library in `tests/unit/test_providers.py`
    - `tests/unit/providers/test_ollama_provider.py`: test suite for unit tests for ollama
    - Add tests to test for edge cases for `ollama_running` in `tests/unit/test_utils.py`
    - `tests/unit/cli/test_cli_utils.py`: test suite for cli validation utils
    - `tests/unit/utils/test_token_counter.py`: add token counting with native Ollama tests
    - `tests/unit/providers/test_metadata.py`: add provider specific ollama metadata handling tests
    - `tests/unit/providers/test_ollama_connection_pooling.py`: test suite for ollama connection pooling
    - `tests/unit/providers/test_ollama_malformed_tools.py`: test suite for ollama native handling of malformed tool calls
    - `tests/unit/providers/test_ollama_warmup_coverage.py`: test suite to cover model warm-up detection code paths

##### Documentation
- [x] Add a section about Native and Chat Completions Ollama in `docs/guides/providers.md`
- [x] Update `docs/installation.md` to mention installation for allos with only ollama
- [x] Update `docs/providers/ollama.md`
    - Mention ways of calling Ollama (Native & Chat Completions) and differences between them
    - Model Capability Matrix
    - Tool calling support by model
    - Model Selection Guide
    - Model warmup detection
    - Streaming retry logic
- [x] Update `docs/guides/cli-reference.md`
    - Mention about `--list-ollama-models` command
    - Mention about `ollama_compat` and `ollama` providers
- [x] Update `docs/reference/metadata.md`
    - Mention about `ProviderSpecificOllama` schema

##### Examples
- [x] Update `examples/local_models.py` to use native Ollama provider
- [x] Create `examples/local_vs_cloud.py` to demonstrate the tradeoffs between cloud and local providers and show metadata for both
- [x] Create `examples/ollama_compat.py` to demonstrate Ollama interaction via Chat Completions compatible endpoint
- [x] Create `examples/ollama_usage.py` to demonstrate all capabilities of native Ollama

#### Native Google Gemini Provider
- [x] Update **`pyproject.toml`**:
    - Added `google-genai` dependency
    - Mark `google` optional dependency as the one requiring Python >= 3.10 because of `google-auth` dependency
- [x] **`allos/providers/base.py`**
    - Add `check_env_config` method that checks if environment is properly configured for a provider, should return a tuple of whether the provider is configured (bool) and a user-friendly message about environment variable status to print to CLI. Subclasses extending this class will override this method for custom logic (to support multiple authentication strategies for Google Provider)
    - Add `thought_signatures` as a field in `Message` class to intercept thought signatures within LLM messages when Gemini uses tool calls
    - Add `thought_signatures` as a field in `ProviderResponse` and `ProviderChunk` classes
- [x] **`allos/providers/google.py`**
    - `GoogleProvider` class
    - `check_env_config`: override base class' method to check for Gemini API keys, Vertex AI authentication via project, location
        - Priority 1: Google Gemini API keys `GEMINI_API_KEY`
        - Priority 2: Service Account JSON file
        - Priority 3: Vertex AI ADC
    - env_var: `GOOGLE_API_KEY`
    - `sub_provider` argument for constructor (handles sub-providers in vertex AI like Anthropic, Llama etc.)
    - `vertexai`, `project`, `location`, `sub_provider` member variables
    - `_model_context_window` private member variable
    - Model availability and context window detection by calling genai client list method
    - `_load_vertex_credentials`: load vertex ai credentials
        - Priority 1: Explicit credentials object
        - Priority 2: Service account JSON path
        - Priority 3: Service account JSON path via environment variable
        - Priority 4: Service account JSON content
        - Priority 5: Service account impersonation
        - Priority 6: Application Default Credentials
    - `_load_credentials_from_file`: load credentials from service account JSON file
    - `_load_credentials_from_json`: load credentials from JSON content
    - `_load_impersonated_credentials`: load impersonated service account credentials
    - `_load_adc_credentials`: load ADC
    - `_verify_model_available`: similar to ollama but uses `models/` prefix for Gemini and `publishers` prefix for vertex AI
    - `_find_similar_models`: find closest matching model names
    - `_extract_model_id`: extract clean model ID from full model name
    - Message conversion
    - Tool calling support
    - Error handling
    - Register with decorator `google`, vertexAI configuration enabled by passing a flag or argument to provider
    - Streaming support with genai
    - **Thought signatures** for Gemini 3.x (required) and 2.5 (optional) `Dict[str, bytes]`
    - Python 3.10+ requirement (enforced at import)
- [x] **`allos/providers/metadata.py`**
    - Add google specific metadata `ProviderSpecificGoogle` with `vertexai`, `project`, `location`, `used_thought_signatures`
- [x] **`allos/providers/registry.py`**
    - Update `list_providers` function to show all providers based on a flag `include_unavailable` (default: True), should show all providers regardless of their Python version requirements, else should not show `google` if python version < 3.10
    - Add `check_provider_env` classmethod to check environment configuration for a provider, delegates to `check_env_config` classmethod for a provider
    - `get_provider` must raise error if `google` provider is called and Python version < 3.10
- [x] **`allos/agent/agent.py`**
    - In `run` method, pass LLM response's `thought_signatures` with tool calls, and content to `context.add_message`
    - In `stream_run`, pass thought_signatures to the context when the stream ends and aggregate thought signatures during streaming
    - Update `AgentConfig` class
        - Respect additional options for providers via `provider_call_options: Dict[str, Any]`
        - `_validate_provider_call_options` to respect provider-managed options and fail fast if reserved keys are passed
        - `_build_provider_call_kwargs` build effective provider kwargs with precedence and validation
        - Connect provider call args to agent run and stream_run methods
- [x] **`allos/context/manager.py`**
    - Pass `thought_signatures` to `add_assistant_message`

##### Updates to CLI for better DX
- [x] **`allos/cli/main.py`**
    - In `print_active_providers`, change logic to use `check_provider_env` instead of `get_env_var_name` which should return whether provider is configured and dev friendly message about configuration
- [x] **`allos/cli/utils.py`**
    - Add default model for google provider: `gemini-2.5-flash-lite`
    - Refactor logic of `validate_api_key` to use `check_provider_env` and `OPENAI_COMPATIBLE_PROVIDERS`


##### Testing & Polish
- [x] **Update configuration**: `tests/conftest.py`
    - Add pytest marker for Gemini (`requires_gemini`) and Vertex AI (`requires_vertexai`) real API tests that marks tests requiring Gemini or Vertex AI credentials/API keys
    - Add `_check_vertexai_conf` method to check if the environment has the required keys to run Gemini/Vertex AI related tests, link with `_apply_integration_test_skips` via `requires_vertexai` and `requires_gemini`
    - Add mock api key for `gemini` and mock `APPLICATION_CREDENTIALS` in `mock_api_keys`
    - Add markers to skip tests if Python version < 3.10 (`requires_python310`, `skip_on_python_310`)
    - Create `get_available_provider_params` to get pytest parametrized values for providers
    - Create `available_providers` to get list of providers available with current Python version
- [x] **New dedicated configuration**: `tests/e2e/conftest.py` to handle configurations and validations across providers
    - `mock_provider_environment`: add context manager to setup mocks for a specific provider
    - `get_client_patch_patch`: get full patch patch to patch a provider client
    - Provider-specific dedicated mocks to setup clients for providers
    - `create_test_metadata_with_builder`: create realistic metadata using `MetadataBuilder`
    - `get_provider_specific_fields`: to generate provider-specific metadata fields
    - Create a simple mock ProviderResponse
    - Create a helper class to manage multi-turn response and conditional logic
- [x] **E2E Test Updates**:
    - **`tests/e2e/test_cli.py`**
      - Update tests to assert for new validation messages per `print_active_providers` of `allos/cli/main.py`
      - Update `mock_agent_and_load_session` to use monkeypatch to monkeypatch environment variables and satisfy `validate_model_and_api_key`, attach `stream_run` method and make it return an empty iterator
      - Add `mock_validation` and `mock_anthropic_validation` fixtures and use them across the tests
      - Add test to validate that switching providers during a session resets the agent config
    - **`tests/e2e/test_cross_provider_workflows`**
        - Same task, all providers
        - Provider switching mid-conversation
        - Provider fallback on failure
        - Google Thought Signatures feature
    - **`tests/e2e/test_google_workflows.py`**
        - Thought signatures multi-turn with Gemini 3.0
        - Vertex AI vs Gemini API behavior
        - Model family comparisons
    - **`tests/e2e/test_metadata_workflows.py`**
        - Metadata lifecycle
        - Provider-specific fields
        - Metadata aggregation across multi-turn conversations
    - **`tests/e2e/test_streaming_workflows.py`**
        - Streaming across all major providers
        - Streaming with function/tool calling
        - Streaming cancellation and resource cleanup
    - `tests/e2e/test_session.py`: Update to use correct markers and parameterize with `requires_*` flags, significant refactor to leverage e2e conftest
- [x] **Integration Tests**:
    - **`tests/integration/providers/test_google_real.py`**: with both Gemini and Vertex AI APIs
        - Simple chat
        - Tool calling
    - `tests/integration/test_provider_switching.py`: Add gemini and vertex ai markers, add gemini and vertex ai to list of LLM APIs to be tested in `providers_to_test`, refactor to use `get_available_provider_params` from conftest
    - `tests/integration/test_session.py`: Add patches for genai client and `_verify_model_available` of `GoogleProvider` class
    - **`tests/integration/test_agent_stream_tool_loop_real.py`**
        - Tool loop with streaming
    - **`tests/integration/test_agent_tool_loop_real.py`**
        - Tool loop with normal run
    - **`tests/integration/test_agent_workflow_real.py`**
        - Agent.run returns content and metadata
    - **`tests/integration/test_cli_stream_real.py`**
        - Stream mode success and session save/load continues context
    - **`tests/integration/test_cli_tool_loop_real.py`**
        - run mode executes tools and persists session
        - stream mode executes tools and persists session
    - **`tests/integration/test_metadata_aggregate_real.py`**
        - aggregate metadata is internally consistent for tool loops
    - **`tests/integration/test_provider_switching.py`**
        - Add metadata assertions and helper functions to validate metadata
    - **`tests/integration/test_session_provider_switch_real.py`**
        - Run on provider A, save/load session, switch to provider B, continue conversation across all implemented providers
    - **`tests/integration/test_session_real.py`**
        - Real worklow: run -> save -> load -> continue conversation
    - **`tests/integration/test_streaming_real.py`**
        - Real streaming: stream_run yields content and records metadata
        - Real provider streaming should surface tool-calling chunks
    - **`tests/integration/providers/test_alias_endpoints_real.py`**
        - Each configured alias should complete one basic chat run
        - Tool calling on alias endpoints
    - `tests/integration/providers/{test_anthropic_real.py,test_openai_real.py,test_chat_completions_real.py,test_google_real.py,test_ollama_real.py}`: helper functions for asserting metadata
    - **`tests/integration/test_error_normalization_real.py`**
        - Validate shared error normalization shape across providers
    - **`tests/integration/providers/test_google_thought_signatures_real.py`**
        - Validate real thought signature emmission for google provider tool calling with Gemini 3.x
- [x] **Unit Test Updates**:
    - **`tests/unit/test_cli_utils.py`**
      - Update tests to assert for new validation messages per `print_active_providers` of `allos/cli/main.py`
      - Add tests for missing google-genai library in `tests/unit/test_providers.py`
    - **`tests/unit/providers/test_google_provider.py`** test suite for unit tests for google provider
        - Fixtures for mocking service account, credentials, and genai client
        - Tests for different methods of authentication
        - Mirror genai error formats for exception handling tests
    - `tests/unit/test_agent.py`: add tests for thought signatures handling, tests for provider_call_options, tests for first metadata preservation
    - `tests/unit/test_providers.py`:
        - Add tests for handling providers via `include_unavailable` for `list-providers`
        - Add tests for provider registry env checks and model determination

*Note: Integration tests that were using mocks such as `test_session.py`, `test_agent_workflow.py`, and `test_tool_execution.py` were moved to E2E to keep integration tests purely realistic.*

- [x] **CI, Config Updates, & DX**:
    - Add `py.typed` marker file to support static typing
    - Update `scripts/setup_dev.sh` to install all dependencies for development
    - **`pyproject.toml`**:
      - Follow latest PEP standards
      - Update dependencies based on security vulnerabilities
      - Simplify optional dependencies to follow PEP and pip standards since uv also follows pip
      - Add lint, test, optional markers
      - Configure black to target Python 3.9 - Python 3.14
      - Remove dependency-groups to not confuse package managers like uv, pip
    - Update publish workflow to use Python version 3.10 for publishing since all core packages work on aleast Python 3.10, install `--all-extras`
    - Update CI to intelligently decide which dependency groups to install based on Python version (use GitHub Actions conditional branches, NO SHELL scripting), update installation flags via `--extra`


##### Documentation
- [x] Update `docs/providers/google.md`
    - Remove note about documentation in progress
    - Mention about new native Google Gemini and Vertex AI APIs
    - Thought signature support
    - Python 3.10 requirement
    - Authentication methods and priority
    - Models, their context window and best for
    - Argument difference Agent vs provider
- [x] Update `docs/reference/agent-api.md`
    - Mention about thought signature handling in Gemini 3.x models
    - Mention about `provider_call_options`
    - Update function signature for `run` and `stream_run` methods to respect `provider_call_options`
- [x] Update `docs/reference/provider-api.md`
    - Mention about schema of `thought_signatures: Dict[str, bytes]`
    - Mention about thought signature handling in `ProviderResponse`, `ProviderChunk` and `Message`
    - Mention about metadata and link it to metadata documentation (in `ProviderResponse`)
- [x] Update `docs/reference/tool-api.md`
    - Mention about Ollama native function calling conversion
    - Mention about Alias providers' tool call formatting path (via chat_completions)
- [x] Update `README.md`
    - Mention about Google Provider and its requirement of Python>= 3.10, Ollama Native Provider
    - Mark Google Provider, Native Ollama Provider as complete
    - Update installation and development commands to include google, ollama
    - Update examples to use correct SDK APIs
- [x] Update `docs/contributing/testing.md` to mention about Python requirements for google provider and additional pytest markers, add quick testing commands
- [x] Update `docs/contributing/adding-provider.md` to describe the new functions and requirements for a provider (stream_chat, check_env_config classmethod)
- [x] Update `docs/installation.md` to mention about google provider python requirements
- [x] Update `docs/guides/agents.md`
      - Update an example to show usage of `provider_call_options`
      - Describe `provider_call_options`

##### Examples
- [x] Create `examples/README.md` to serve as index for all examples.
- [x] Update `examples/cli_workflow.sh` for better DX in terms of what is being shown
- [x] Create `examples/google_direct_provider.py`
    - Agent with AgentConfig
    - ProviderRegister.get_provider
    - Google Provider specific options
- [x] Create `examples/metadata_inspection.py` to inspect normalized and provider-specific metadata
- [x] Create `examples/provider_call_options.py` to show usage of provider_call_options
- [x] Create `examples/streaming_chunks.py`
    - Content deltas
    - Tool call start/delta/done events
    - Final metadata

#### Native Cohere Provider
- [x] Update **`pyproject.toml`**:
    - Added `cohere` dependency
    - Mark `cohere` optional dependency
- [x] **`allos/providers/registry.py`**
    - Change `cohere` ChatCompletions alias to `cohere_compat`
- [x] **`allos/providers/cohere.py`**
    - Register with `cohere`
    - `_safe_error_body`, `_raise_provider_error` private helpers to get and print cohere specific errors converted to `ProviderError`
    - `env_var`: `COHERE_API_KEY`
    - `_model_context_window` member variable
    - `_verify_model_available` pulls models from Cohere client list
    - `_find_similar_models` for better DX in errors
    - Helpers to convert tool calls, messages, system prompt
    - Helpers to parse API response, build metadata
    - Streaming support

##### Testing
- [x] **Update configuration**:
    - `tests/conftest.py`
      - Add pytest marker for Chere real API tests `requires_cohere` that marks tests requiring Cohere
      - Update `mock_api_keys` fixture to also monkeypatch cohere api key
    - `tests/e2e/conftest.py`
        - Add cohere branch to `mock_provider_environment`
        - Add fixture to setup cohere mocks
- [x] **Update E2E tests**
    - `tests/e2e/{test_cross_provider_workflows.py,test_metadata_workflows.py}`: Add cohere client patch
    - `tests/e2e/test_session.py`: Patch CohereProvider's chat and `_verify_model_available` methods
- [x] **Integration tests**
    - `tests/integration/providers/test_cohere_real.py`: Cohere specific tests with real-API calls
    - `tests/integration/providers/test_error_normalization.py`: Cohere specific tests for error normalization
    - `tests/integration/test_provider_switch_real.py`: Cohere for provider switching
- [x] **Unit tests**
    - `tests/unit/providers/test_cohere_provider.py`: Unit tests for Cohere provider logic
    - `tests/unit/test_providers.py`: Tests for missing cohere library

##### Documentation
- [x] Add an example related to Cohere provider `docs/guides/providers.md`
- [x] Add installation command for Cohere `docs/installation.md`
- [x] Mention about Cohere Chat Completions Compatibility `docs/providers/chat-completions.md`
- [x] Describe native Cohere provider `docs/providers/cohere.md`
- [x] Update `docs/reference/cli-reference.md` to mention model_name requirement for Cohere, Ollama and Chat completions alias endpoints

##### Examples
- [x] Create `examples/cohere_usage.py`
    - Chat
    - Streaming
    - Tool-calling
    - Metadata inspection
- [x] Create `examples/cohere_compat.py`
    - Chat
    - Streaming
- [x] Update `examples/fast_inference.py`
    - Add Cohere env
    - Update messages
    - Add Cohere specific cost metrics

#### Changes Post Manual Testing
E2E tests revealed that state of ollama was getting corrupted because of cache. This caused some of the tests to become flaky. Solution: add to `tests/e2e/conftest.py`:
1. Function `_reset_ollama_global_state`: Clear module-level Ollama caches used across provider instances
2. Fixture `clear_ollama_global_state_between_e2e_tests`: Prevent cross-test leakage of pooled Ollama clients/warm-up state
3. Anytime ollama is chosen as provider_name, reset ollama state before and after

### Pending
- [ ] Streaming exponential backoff for other providers (See further phases)
- [ ] Cross-check model selection guide and capability matrix of `docs/providers/ollama.py` with actual Ollama model library and documentation (PENDING VERIFICATION)
- [ ] Handling of thought signatures in interactive session files

### Experimental
- [ ] Function calling with FunctionGemma for Ollama models that do not natively support function calling

### Success Criteria
✅ Can initialize ollama native, ollama compat, google (Gemini/VertexAI), Cohere native providers

✅ CI functioning for Python 3.14 across all OSes

✅ Can send messages and get responses

✅ Provider specific options work

✅ Tool calling works

✅ Error handling is robust

✅ All tests pass

### Deliverables
- Working Ollama provider
- Working Google provider
- Working Cohere native provider
- Provider specific options support
- Proactive env validation works and is correct
- Comprehensive tests

### Example Usage After This Phase (VERIFIED)
```python
"""Phase 3 smoke test: native Ollama + Google + Cohere providers."""

import os
import sys

from allos import Agent, AgentConfig
from allos.cli.utils import validate_model_and_api_key
from allos.providers import Message, MessageRole, ProviderRegistry

MODELS = {
    "ollama": os.getenv("ALLOS_OLLAMA_MODEL", "llama3.1:latest"),
    "google": os.getenv("ALLOS_GOOGLE_MODEL", "gemini-2.5-flash-lite"),
    "cohere": os.getenv("ALLOS_COHERE_MODEL", "command-r7b-12-2024"),
}


def verify_provider(provider: str, model: str) -> None:
    validation = validate_model_and_api_key(provider, model, api_key=None)
    if not validation.success:
        print(f"SKIP {provider}: {validation.error}")
        return

    p = ProviderRegistry.get_provider(provider, model=validation.model)
    msg = [Message(role=MessageRole.USER, content="Reply in one short sentence.")]
    r = p.chat(msg, temperature=0)
    assert r.content or r.tool_calls, f"{provider}: empty chat response"

    saw_output = False
    for chunk in p.stream_chat(msg, temperature=0):
        saw_output = saw_output or bool(chunk.content) or chunk.final_metadata is not None
    assert saw_output, f"{provider}: no streaming output"

    if provider == "google":
        assert r.metadata.provider_specific.google is not None
    print(f"OK {provider} ({validation.model})")


providers = ProviderRegistry.list_providers(include_unavailable=True)
for expected in ("ollama", "google", "cohere", "ollama_compat", "cohere_compat"):
    assert expected in providers, f"Missing provider/alias: {expected}"

for name in ("ollama", "google", "cohere"):
    if name == "google" and sys.version_info < (3, 10):
        print("SKIP google: requires Python 3.10+")
        continue
    verify_provider(name, MODELS[name])

# Verify AgentConfig.provider_call_options path from Phase 3.
configured = [
    p for p in ("cohere", "google", "ollama")
    if validate_model_and_api_key(p, MODELS[p], api_key=None).success
]
if configured:
    agent = Agent(
        AgentConfig(
            provider_name=configured[0],
            model=MODELS[configured[0]],
            no_tools=True,
            provider_call_options={"temperature": 0},
        )
    )
    _ = agent.run("Return one word.")
    assert agent.last_run_metadata is not None
    print(f"OK agent provider_call_options ({configured[0]})")
else:
    print("No native provider configured; set OLLAMA/GOOGLE/COHERE env and rerun.")
```

## Phase 4: CLI & DX Enhancements 🏗️ IN PROGRESS

**Status**: 🏗️ In Progress

**Focus**: CLI & DX improvements to match existing open source projects like [Open Code](http://opencode.ai/docs/), [Kilo CLI](https://kilo.ai/docs/getting-started), [Copilot CLI](https://docs.github.com/en/copilot/concepts/agents/copilot-cli/about-copilot-cli) and closed source projects like [Claude Code](https://code.claude.com/docs/en/overview), [Cline CLI](https://docs.cline.bot/getting-started/what-is-cline), [Codex CLI](https://developers.openai.com/codex/cli/), [Gemini CLI](https://geminicli.com/docs/).

### Goals
- Support DX features of existing agentic REPLs
- No significant performance overhead
- Autocomplete works
- TUI improvements

### Tasks

#### Env Validation
- [x] **`allos/cli/utils.py`**
    - `ValidationResult` Pydantic model for provider, model, and API key config validation
        - `success`: whether validation was successful
        - `model`: name of the model to use
        - `model_defaulted`: whether the default model name was used
        - `error`: error message if validation failed
        - `error_type`: type of validation error - `model`, or `api_key` focused
        - `field_validators` for `error_type` and `error` message
    - `determine_model`: returns `env_var` or `error_msg`, uses `ProviderRegistry.get_env_var_name` and `ProviderRegistry.check_provider_env`
    - `validate_model_and_api_key`: returns `ValidationResult`
    - `display_validation_error`: returns validation error to be displayed in Rich console panel
    - `display_provider_info`: Display provider/model info message

#### Interactive Mode Improvements
- [x] Welcome message must include information about streaming mode
- [x] Use `ValidationResult`, `display_provider_info`, `display_validation_error`, `validate_model_and_api_key` to validate provider, model and API key config
- [x] Update `start_interactive_session`
    - Handle `stream` argument to decide whether to stream or not
    - Refactor to use helpers from utils.py
- [x] Update `_print_welcome_message` to print info about streaming and updated slash commands
- [x] Update `_run_repl_loop`
    - Handle `stream` argument
    - Handle slash commands such as `/quit`, `/exit/`, `/help`, `/stream on`, `/stream off`
    - Streaming should be configured to be turned on or off mid-session
- [x] Add helpers such as `_execute_sync` for simple synchronous flow and `_execute_streaming` for streaming output

#### Core CLI Improvements
- [x] **`allos/cli/main.py`**
    - Pass stream flag to interactive mode if provided
    - Use Validation helpers in core functions such as run_agent and run_agent_stream
    - `--active-providers` has table headers: `Provider`, `Status`, `Details`

#### Command Autocomplete

**Status**: 🗓️ Not Started

**Focus**: command autocomplete, fuzzy matching, keyboard navigation, visual feedback

**Goal**:
- Instant feedback
- Smart matching of commands as user types
- Arrow keys and tab for selection
- Color-coded highlighting for matches
- Recently used commands appear first


- [ ] **Configuration `pyproject.toml`**
    - Install `prompt-toolkit` and `rapidfuzz`. Check `rapidfuzz` docs about Python 3.9 compatibility.
- [ ] **`allos/cli/commands.py`**
    - Centralized command registry to manage available commands
    - `ReplCommand` class
        - `name: str`: help, stream, exit, etc.
        - `description: str`
        - `aliases: Optional[List[str]] = None`: `e.g. ["quit"] for exit`
        - `usage: Optional[str] = None`: `e.g. "/stream [on|off]"`
        - `handler: Optional[Callable] = None`
        - `last_used: Optional[datetime] = None`
        - `use_count: int = 0`
    - `CommandRegistry` clas
        - Registry for REPL commands with usage tracking
        - `_commands: Dict[str, ReplCommand] = {}`
        - `register` classmethod: Register a new command
        - `get_all_commands` classmethod: Get all commands, optionally sorted by frecency
        - `mark_used` classmethod: Update usage statistics when a command is executed
        - `fuzzy_match` classmethod: Find commands matching query (command name + command description) using fuzzy matching and `rapidfuzz` library
    - **Autocomplete Integration `allos/cli/autocomplete.py`**
        - `SlashCommandCompleter` class (inherits from `Completer` of `prompt_toolkit.completion`): custom completer for slash commands with fuzzy matching
          - `__init__`: initialize registry
          - `get_completions`: generate completions for current input
        - `create_prompt_session`: Create a configured `prompt_toolkit` session (handle keyboard interrupts)
        -  Other functions as necessary
   -  **Modified REPL loop `allos/cli/interactive.py`**
        - `_run_repl_loop`
            - Create prompt session
            - Instead of console.input, use prompt_toolkit
            - Mark command used
            - Handle streaming enablement
            - Configure key bindings
    - **Performance Optimization**:
        - Cache frecency scores
        - Lazy load command descriptions
        - Profile autocomplete latency
    - **Color Scheme**

| State | Command Color | Description Color | Background |
|-------|---------------|-------------------|------------|
| Default | `cyan` | `dim` | `#333333` |
| Highlighted (current selection) | `cyan bold` | `cyan` | `#00aaaa` |
| Executed | `green` | N/A | N/A |

- **Example Command Structure**
```python
COMMANDS = [
    ReplCommand(
        name="help",
        description="Show available commands and their usage",
        usage="/help",
    ),
    ReplCommand(
        name="stream",
        description="Show current streaming status",
        usage="/stream",
    ),
    ReplCommand(
        name="stream on",
        description="Enable streaming mode for responses",
        usage="/stream on",
    ),
    ReplCommand(
        name="stream off",
        description="Disable streaming mode",
        usage="/stream off",
    ),
    ReplCommand(
        name="exit",
        description="Exit the interactive session",
        usage="/exit",
        aliases=["quit"],
    ),
    ReplCommand(
        name="clear",
        description="Clear the terminal screen",
        usage="/clear",
    ),
    ReplCommand(
        name="history",
        description="Show conversation history",
        usage="/history [n]",
    ),
    ReplCommand(
        name="save",
        description="Save current session to file",
        usage="/save <filename>",
    ),
]
```
#### Future Commands (see further phases)
- `/model <name>` - Switch LLM model
- `/provider <name>` - Switch provider
- `/tools` - List available tools
- `/config` - Show/modify configuration
- `/retry` - Retry last prompt
- `/edit` - Edit last prompt in $EDITOR
- `/context` related commands
- `/save` or `export` - only if conversation has content
- User defined shortcuts (stored in config)
- Hide `/stream off` when streaming is already off
- Press `F1` on highlighted command to see full docs
- Emacs support for navigation, detect terminal capabilities on startup, show appropriate help based on terminal
- Support for multi-line prompts
- Support for command chaining
- Additional keyboard shortcuts


#### Test related updates
- [x] **`tests/e2e/test_cli.py`**
    - Update `mock_agent_and_load_session`, `mock_anthropic_validation` to use `ValidationResult` as return_value
    - Update `test_run_command_max_tokens` to assert updated error/info messages
    - Update `test_run_command_session_management` and other related tests to use updated slash commands and assert updated values
- [x] **`tests/unit/cli/test_cli_utils.py`**
    - Update tests to assert correct error/info messages and validation values
    - Add `TestValidationResult` test class to test for `ValidationResult` Pydantic model validators
- [x] **`tests/unit/cli/test_interactive.py`**
    - Tests for new slash commands
- [x] **`tests/unit/test_agent.py`**
    - Tests to validate aggregate metadata (provider-specific)
- [ ] **`test`/unit/cli/test_commands.py`**
    - Unit test command registry, fuzzy matching
    - Unit test slash command autocomplete
- [ ] **`tests/e2e/test_interactive_autocomplete.py`**
    - Test autocomplete completes command
    - Test arrow navigation selects command
    - Test frecency affects order
    - and many more

#### Websocket Mode in OpenAI Responses API

**Status**: 🗓️ Planned

**Focus**: websocket transport for responses API

**Goal**:
- Websocket mode for both Agentic synchronous and asynchronous methods
- Warmup
- Standalone compaction + server-side compaction pass-through
- Reconnect/recovery behavior
- Provider state persistence
- Full CLI support

- [ ] **Configuration `pyproject.toml`**
    - Update `openai` library dependency version to `v2.22.0`
    - Add `websocket-client` dependency to `openai` dependency group
- [ ] Add internal helper module **`allos/providers/_openai_ws.py`** or in `allos/utils/`
    - Create a provider-internal class to manage one socket per provider instance
        - [ ] Connect to `wss://api.openai.com/v1/responses` via auth header with bearer key
        - [ ] Track `connected_at`, `last_response_id`, `last_sent_input_items`, `last_model`, `last_store`, `reconnect_count`
        - [ ] Enforce one in-flight response at a time on a socket
        - [ ] Auto-reconnect before 60 min timeout (e.g. proactive rotate at 55 min)
        - [ ] On hard close/timeout: reconnect and continue
- [ ] **`allos/agent/agent.py`** in `AgentConfig`'s `provider_call_options`:
    - `transport: Literal["http", "websocket"]`: default is `http`
    - `ws_fallback: bool`: SDK default is `False`, CLI default is `True`
    - `store: Optional[bool]`: unchanged default behavior when unset
    -  `ws_warmup: bool`: default is False - sends a warmup request with `generate=False` before first generated turn
    -  `ws_compact: Literal["off", "server", "standalone"]`: default is `off`
    -  `ws_compact_threshold_items: int`: default is `120` and is used only in standalone mode
    -  `ws_compact_model: Optional[str]`: default current provider model
- [ ] **`allos/providers/openai.py` In OpenAI provider call path**
    - If `transport="websocket"`:
        - If previous stat exists and new input has old input as exact prefix:
            - send only suffix items as `input`
            - send `previous_response_id=<last_response_id>`
        - Else:
            - send full `input`
            - omit `previous_response_id` (new chain)
    - Send event payload:
        - `type: "response.create"`
        - `model`, `input`, optional `tools`, optional `instructions`, optional `store`
        - `generate=False` when warmup is requested
    - Parse returned streaming events and update state with latest response id
    - On error 4xx/5xx continuation failure:
        - evict cached prior response state as docs require
        - follow `ws_fallback` policy
    - In `chat()`:
        - When `transport="websocket"`:
            - consume event stream to completion internally
            - accumulate test/tool calls exactly like current streaming handlers
            - build `ProviderResponse` + metadata
            - return sync response to `Agent.run()`
        - When `transport="http"`
            - Keep current behavior
    - In `stream_chat()`:
        - When `transport="websocket"`
            - map incoming WS events to existing `ProviderChunk` semantics:
                - text deltas -> `ProviderChunk(content=...)`
                - function call start/delta/done -> tool chunks
                - completed -> `ProviderChunk(final_metadata=...)`
                - error -> `ProviderChunk(error=...)`
            - preserve current HTTP streaming behavior for non-WS mode
    - Warmup support `generate=false`
        - If `ws_warmup=True`, before first generated turn in a chain:
            - send `response.create` with same model/instructions/tools and `generate=False`
            - cache returned response id
            - next generated turn uses `previous_response_id` for faster start
            - no model output should be expected from warmup turn
    - Compaction support
        - `ws_compact="server"`:
            - pass through `context_management` options in `response.create`
        - `ws_compact="standalone"`
            - when `len(input_times) >= ws_compact_threshold_items` call `/responses/compact` first
            - start a new chain (`previous_response_id` ommitted/null)
            - `input = compacted.output + new items`
        - `ws_compact="off"`: no compaction behavior beyond existing
    - Error handling
        - `previous_response_not_found`:
            - if `ws_fallback=True`: retur once via HTTP with full context
            - else: raise `ProviderError` with actionable message
            - reconnect socket
        - `websocket_connection_limit_reached`
            - reconnect socket
            - retry current request atleast once (configurable?)
            - if retry fails, apply fallback policy
        - socket close/read timeout
            - reconnect + retry once
        - malformed event payload
          - yield/raise normalized provider error
- [ ] Agent + context state persistence for `store=true`
  - Update `allos/agent/agent.py`
  - Update `allos/context/manager.py`
- [ ] Provider State hooks
    - If provider implements `export_runtime_state()` copy into `context.provider_state["openai"]`
        - do after successful run/stream completion
        - do before `save_session(...)`
    - If provider implements `restore_runtime_state(state)`, call during agent init/load
- [ ] Session persistence
    - persist OpenAI WS continuation state only if provider says `store=True`
    - if `store` is false/unset, strip response continuation identifiers on save/load
    - maintain backward compatibility with existing session files

-  [ ] **New CLI options `allos/cli/main.py`**:
    - `--transport [http|websocket]`: default is `http`
    - `--ws-fallback/no-ws-fallback`: default is `True` in CLI
    - `--store [default|true|false]`: tri-state
    - `--ws-warmup`
    - `--ws-compact [off|server|standalone]`
    - `--ws-compact-threshold <int>`
    - Parse new flags
    - Build `provider_call_options1 from CLI flags and pass into `AgentConfig.provider_call_options`
    - Enforce compatibility:
        - if `provider != "openai"` and `transport=websocket`, fail early with clear message
        - `--active-providers` output includes OpenAI WS readiness hint (API key set + transport availability)
- [ ] **New interactive mode commands `allos/cli/interactive.py`**:
    - `/transport` (show current transport)
    - `/transport http`
    - `/transport websocket`
    - `/ws` (show ws settings)
    - `/ws fallback on|off`
    - `/ws compact off|server|standalone`
    - `/ws warmup on|off`
    - Show transport and ws settings in welcome/help
    - switching `/transport websocket` updates agent/provider call options in-session
    - keep existing `/stream` behavior independent
- [ ] **`allos/providers/metadata.py`**
    - Extend `ProviderSpecificOpenAI` with:
        - `transport: Optional[str]` (`http` or `websocket`)
        - `continuation_used: bool` (whether `previous_response_id` was sent)
        - `fallback_used: bool` (whether HTTP fallback path was used)
        - `reconnect_count: int`

##### Test related updates
- **`tests/unit/providers/test_openai_websocket_provider.py`**:
      - opens and reuses socket
      - incremental suffix sending with `previous_response_id`
      - warmup `generate=false` returns and caches response ID
      - standalone compact path starts a new chain
      - `previous_response_not_found`:
          - fallback false => error
          - fallback true => HTTP retry with full context
      - connection limit error triggers reconnect path
      - metadata includes transport/continuation/fallback/reconnect fields
- **`tests/unit/test_agent.py`**
    - captures provider runtime state after run/stream
    - restores runtime state on loaded session when allowed
    - does not restore continuation state when `store!=True`
- **`tests/unit/cli/test_cli_utils.py` and `tests/unit/cli/test_interactive.py`**
    - parse/validate new flags
    - non-openai + websocket transport fails
    - slash commands mutate settings correctly
- **`tests/e2e/test_cli.py`**
    - add websocket-model CLI invocations and option propagation checks
    - add interactive command flow tests for `/transport` and `/ws`
- **`tests/integration/test_openai_real.py`**
    - websocket simple chat
    - websocket tool-calling loop
    - websocket stream parity vs HTTP stream
    - fallback path test (if fault injection unavailable then can be mock-assisted)
    - independent gating with `requires_openai_ws`

##### Documentation
- Update `docs/providers/openai.md`
    - add transport modes `http` vs `websocket`
    - continuation semantics `previous_response_id`
    - warmup and compaction modes
    - reconnect and failure handling
- Update `docs/reference/agent-api.md`
    - document new `provider_call_options`
- Update `docs/reference/provider-api.md`
    - metadata field additions for OpenAI transport diagnostics
- Update `docs/reference/cli-reference.md`
    - document new flags and slash commands

##### Pending
- [ ] Consider how context management with this feature would affect future phases concerning context management


## Phase 5: Test Coverage Expansion



## Phase 6: Cloud-Based Providers

**Status**: 🗓️ Planned

**Focus**: Azure Openai, Amazon Bedrock

### Goals
- First class cloud providers addition
- Support chat, streaming, tool-calling, error normalization, and metadata
- Keep CLI DX consistent

### Tasks

#### Native Azure OpenAI Provider
- [ ] Update `pyproject.toml`
    - Optional dependency group `azure` with `azure-identity` for Microsoft Entra ID auth
    - Include `azure` in `all` extra
- [ ] **`allos/providers/azure_openai.py`**
    - Implement `AzureOpenAIProvider` so Azure uses Responses API conversion/parsing paths already used by native OpenAI
    - Add `check_env_config`
    - Support auth modes:
        1. Mode 1: `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT`
        2. Mode 2: Entra ID token provider via `azure.identity.DefaultAzureCredential` + `get_bearer_token_provider`
    - Enforce endpoint normalization to Azure v1 endpoint shape (`.../openai/v1`)
    - Treat `model` as deployment name (Azure requirement) with validation errors when deployment is missing/invalid
    - Add Azure-specific error normalization helpers for 401/403/404/429 + deployment-not-found guidance
- [ ] Update `allos/providers/metadata.py`
    - Add `ProviderSpecificAzureOpenAI`
        - `endpoint_host`
        - `deployment`
        - `auth_mode`
- [ ] Update `allos/providers/registry.py`
    - Native provider registration visibility and env checks
- [ ] Update `allos/cli/utils.py`
    - Model detection for `azure_openai` using `AZURE_OPENAI_DEPLOYMENT` fallback when `--model` not passed

##### Tests & Polish
- [ ] **Update configuration**:
    - `tests/conftest.py`
        - Add markers: `requires_azure_openai`
        - Add model provider defaults
        - Extend integration skip logic for Azure
    - `tests/e2e/conftest.py`
        - Add provider mock branches for Azure `openai.OpenAI`
        - Add metadata fixtures for Azure provider-specific fields
- [ ] **E2E Test Updates**:
    - `tests/e2e/test_cross_provider_workflows.py`: include Azure
    - `tests/e2e/test_streaming_workflows.py`: include Azure streaming
    - `tests/e2e/test_cli.py`: add validation and active-provider coverage
- **Integration Test Updates**:
    - `tests/integration/providers/test_azure_openai_real.py`: Real Azure API Integration test
    - `tests/integration/test_provider_switching.py`: include Azure
    - `tests/integration/test_session_provider_switch_real.py`: include Azure
- **Unit Test Updates**:
    - `tests/unit/providers/test_azure_openai_provider.py`: Unit tests for azure
    - `tests/unit/providers/test_metadata.py`: add Azure provider-specific schema tests
    - `tests/unit/test_providers.pt`: Missing deps + provider registration + env checks

##### Documentation
- [ ] Create `docs/providers/azure-openai.md`
- [ ] Update `docs/guides/providers.md` with cloud provider comparison and selection guidance
- [ ] Update `docs/installation.md`: extras for `azure`
- [ ] Update `docs/reference/metadata.md`: `ProviderSpecificAzureOpenAI`

##### Examples
- [ ] Create `examples/azure_openai_usage.py`
- [ ] Update `examples/provider_switching.py` to include Cloud provider switching
- [ ] Update `examples/universal_chat.py` to include cloud-provider targets when credentials exist

##### Experimental
- [ ] Optional fallback route from `azure_openai` to `openai` on Azure deployment outage (feature-flagged).

#### Native AWS Bedrock Provider
- [ ] Update `pyproject.toml`
    - Optional dependency group `bedrock` with `boto3`
    - Include `bedrock` in `all` extra
- [ ] **`allos/providers/bedrock.py`**
    - Initialize `boto3.client("bedrock-runtime", region_name=...)`
    - Add optional model discovery client `boto3.client("bedrock", region_name=...)` for model validation when permissions allow.
    - Implement `check_env_config` using AWS credential chain and region availability checks
    - Implement message conversion for Bedrock Converse API (`messages`, `system`, text blocks)
    - Implement tool conversion from `BaseTool` to Bedrock `toolConfig.tools[].toolSpec`
    - Implement parsing of `toolUse` blocks to `ToolCall`
    - Implement tool result mapping from `MessageRole.TOOL` to Bedrock `toolResult` content blocks
    - Implement `chat` via `converse`
    - Implement `stream_chat` via `converse_stream` including text deltas and tool-calling chunks
    - Implement provider-call-option mapping to Bedrock `inferenceConfig` (`temperature`, `topP`, `maxTokens`, `stopSequences`)
    - Add robust error normalization for botocore exceptions (`AccessDeniedException`, throttling, validation/model errors)
- [ ] Update `allos/providers/metadata.py`
    - Add `ProviderSpecificBedrock`
        - `region`
        - `stop_reason`
        - `latency_ms`

##### Tests & Polish
- [ ] **Update configuration**:
    - `tests/conftest.py`
        - Add markers: `requires_bedrock`
        - Add model provider defaults
        - Extend integration skip logic for AWS
    - `tests/e2e/conftest.py`
        - Add provider mock branches for Bedrock `boto3.client`
        - Add metadata fixtures for Bedrock provider-specific fields
- [ ] **E2E Test Updates**:
    - `tests/e2e/test_cross_provider_workflows.py`: include Bedrock
    - `tests/e2e/test_streaming_workflows.py`: include Bedrock streaming
    - `tests/e2e/test_cli.py`: add validation and active-provider coverage
- **Integration Test Updates**:
    - `tests/integration/providers/test_bedrock_real.py`: Real AWS Bedrock API Integration test
    - `tests/integration/test_provider_switching.py`: include Bedrock
    - `tests/integration/test_session_provider_switch_real.py`: include Bedrock
- **Unit Test Updates**:
    - `tests/unit/providers/test_bedrock_provider.py`: Unit tests for bedrock
    - `tests/unit/providers/test_metadata.py`: add Bedrock provider-specific schema tests
    - `tests/unit/test_providers.py`: Missing deps + provider registration + env checks

##### Documentation
- Create `docs/providers/bedrock.md`
- Update `docs/guides/providers.md` with cloud provider comparison and selection guidance
- Update `docs/installation.md` with extras for `bedrock`
- Update `docs/reference/metadata.md` with `ProviderSpecificBedrock`

##### Examples
- [ ] Create `examples/bedrock_usage.py`
- [ ] Update `examples/provider_switching.py` to include Cloud provider switching
- [ ] Update `examples/universal_chat.py` to include cloud-provider targets when credentials exist

##### Pending
- [ ] Bedrock multimodel content block parity: image/document/video (can be deferred)
- [ ] Advanced Bedrock config (follow up phases)
- [ ] Conditional provider selection, multiple authentication options (OIDC, tokens, Service Account JSON, Oauth) across providers, responses API, Anthropic Messages API reusability across providers (THIS IS A MAJOR UNDERTAKING)

### Success Criteria
✅ Can initialize and run `azure_openai`, `bedrock` providers.

✅ Chat and streaming work for all cloud providers.

✅ Tool calling works for Azure OpenAI and Bedrock native providers.

✅ Provider switching and session continuation work across all cloud providers.

✅ All tests pass (unit, e2e, integration).

### Deliverables
- Working Azure OpenAI native provider (Responses API semantics).
- Working Amazon Bedrock native provider (Converse/ConverseStream).
- Provider-specific metadata for Azure and Bedrock.
- Comprehensive tests, docs, and runnable examples.

### Example Usage After This Phase PENDING VERIFICATION
```python
import os

from allos import Agent, AgentConfig
from allos.cli.utils import validate_model_and_api_key

targets = [
    ("azure_openai", os.getenv("AZURE_OPENAI_DEPLOYMENT")),
    ("bedrock", os.getenv("BEDROCK_MODEL_ID")),
    ("github_copilot", os.getenv("ALLOS_GITHUB_MODEL", "openai/gpt-4.1-mini")),
]

for provider_name, model in targets:
    if not model:
        print(f"SKIP {provider_name}: model/deployment not configured")
        continue

    validation = validate_model_and_api_key(provider_name, model, api_key=None)
    if not validation.success:
        print(f"SKIP {provider_name}: {validation.error}")
        continue

    agent = Agent(
        AgentConfig(
            provider_name=provider_name,
            model=validation.model,
            no_tools=True,
            provider_call_options={"temperature": 0},
        )
    )

    response = agent.run("Return exactly: CLOUD_PROVIDER_OK")
    assert "CLOUD_PROVIDER_OK" in response
    assert agent.last_run_metadata is not None

    saw_stream_output = False
    for chunk in agent.stream_run("Return exactly: STREAM_PROVIDER_OK"):
        if chunk.content:
            saw_stream_output = True
    assert saw_stream_output, f"{provider_name}: no streaming output"

    print(f"OK {provider_name} -> {validation.model}")
```
