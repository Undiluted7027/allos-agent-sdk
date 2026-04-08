# Examples

This directory contains runnable examples for the Allos SDK.

## Quick Index

| Example | Focus | Requires API Key | Uses Tools | Notes |
|---|---|---:|---:|---|
| `basic_usage.py` | Direct provider chat (`ProviderRegistry`) | Yes | No | Smallest Python API example |
| `agent_usage.py` | End-to-end agent flow + session save/load | Yes | Yes | Multi-step workflow |
| `custom_tools.py` | Defining and using a custom tool | Yes | Yes | `@tool` registration |
| `tool_usage.py` | Direct `ToolRegistry` usage | No | Yes | No LLM call required |
| `provider_switching.py` | Switching providers via shared session context | Yes | Mixed | Multi-provider pipeline |
| `provider_discovery.py` | Inspect available providers + env readiness | No | No | Native vs alias inventory |
| `universal_chat.py` | Same prompt across multiple providers | Yes | No | Easy provider comparison |
| `fast_inference.py` | Metadata-based speed comparison | Yes | No | Throughput-focused |
| `local_models.py` | Native Ollama with tool calling | No (local) | Yes | Requires Ollama |
| `ollama_compat.py` | Ollama via OpenAI-compatible endpoint | No (local) | No | Simpler compat mode |
| `ollama_usage.py` | Comprehensive Ollama guide | No (local) | Mixed | Native vs compat + streaming |
| `local_vs_cloud.py` | Local vs cloud comparison | Mixed | Optional | Cost and privacy framing |
| `omnibus.py` | Multi-provider Python omnibus workflow | Yes | Yes | Advanced demo |
| `cli_workflow.sh` | Guided CLI walkthrough | Depends | Mixed | CLI-focused |
| `omnibus_cli.sh` | Multi-provider CLI workflow | Yes | Yes | CLI + session chaining |
| `alias_workflow.sh` | Env-driven alias smoke workflow (CLI) | Depends on alias | No | No-tools + stream in one script |
| `provider_call_options.py` | `provider_call_options` and runtime kwargs | Yes | No | `run`/`stream_run` options |
| `chat_completions_alias_smoke.py` | Smoke test any alias provider via env | Depends on alias | No | Great for quick alias checks |
| `cohere_usage.py` | Native Cohere provider guide | Yes (`COHERE_API_KEY`) | Mixed | Chat, stream, tools, agent, metadata |
| `cohere_compat.py` | Cohere via OpenAI-compatible endpoint | Yes (`COHERE_API_KEY`) | No | Compatibility mode (`cohere_compat`) |
| `google_direct_provider.py` | Advanced Google direct-provider config | Yes | No | Vertex/Gemini kwargs |
| `streaming_chunks.py` | Provider-direct streaming chunk anatomy | Yes | Optional | Content/tool/metadata chunks |
| `metadata_inspection.py` | Inspect normalized + provider-specific metadata | Yes | No | Metadata-focused |

## Recommended Start Paths

1. Agent basics:
   - `basic_usage.py`
   - `agent_usage.py`
2. Tools:
   - `tool_usage.py`
   - `custom_tools.py`
3. Providers:
   - `provider_discovery.py`
   - `universal_chat.py`
   - `provider_switching.py`
   - `alias_workflow.sh`
   - `chat_completions_alias_smoke.py`
   - `cohere_usage.py`
   - `cohere_compat.py`
4. Ollama/local:
   - `local_models.py`
   - `ollama_compat.py`
   - `ollama_usage.py`
5. Metadata/streaming/options:
   - `metadata_inspection.py`
   - `streaming_chunks.py`
   - `provider_call_options.py`

## Environment Notes

- Cloud providers typically use env vars such as:
  - `OPENAI_API_KEY`
  - `ANTHROPIC_API_KEY`
  - `COHERE_API_KEY`
  - `GROQ_API_KEY`
  - `MISTRAL_API_KEY`
  - `TOGETHER_API_KEY`
  - `GOOGLE_API_KEY` / `GEMINI_API_KEY`
- For local Ollama examples, run:
  - `ollama serve`
  - pull required models (for example `ollama pull llama3.1`)

## Cohere Modes

- `cohere` (native): Use `cohere_usage.py` for native Cohere chat, streaming, tool-calling, and metadata behavior.
- `cohere_compat` (compat endpoint): Use `cohere_compat.py` when you specifically want the OpenAI-compatible endpoint path.

## Alias Discovery

- To see current provider names in your install: `allos --list-providers`
- To inspect readiness from Python: run `provider_discovery.py`
- To smoke-test one alias quickly: run `chat_completions_alias_smoke.py`
- To smoke-test alias flows from CLI: run `alias_workflow.sh`
