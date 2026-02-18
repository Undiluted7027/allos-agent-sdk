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
| `universal_chat.py` | Same prompt across multiple providers | Yes | No | Easy provider comparison |
| `fast_inference.py` | Metadata-based speed comparison | Yes | No | Throughput-focused |
| `local_models.py` | Native Ollama with tool calling | No (local) | Yes | Requires Ollama |
| `ollama_compat.py` | Ollama via OpenAI-compatible endpoint | No (local) | No | Simpler compat mode |
| `ollama_usage.py` | Comprehensive Ollama guide | No (local) | Mixed | Native vs compat + streaming |
| `local_vs_cloud.py` | Local vs cloud comparison | Mixed | Optional | Cost and privacy framing |
| `omnibus.py` | Multi-provider Python omnibus workflow | Yes | Yes | Advanced demo |
| `cli_workflow.sh` | Guided CLI walkthrough | Depends | Mixed | CLI-focused |
| `omnibus_cli.sh` | Multi-provider CLI workflow | Yes | Yes | CLI + session chaining |
| `provider_call_options.py` | `provider_call_options` and runtime kwargs | Yes | No | `run`/`stream_run` options |
| `cohere_usage.py` | Native Cohere provider guide | Yes (`COHERE_API_KEY`) | Mixed | Chat, stream, tools, agent, metadata |
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
   - `universal_chat.py`
   - `provider_switching.py`
   - `cohere_usage.py`
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
  - `GROQ_API_KEY`
  - `MISTRAL_API_KEY`
  - `TOGETHER_API_KEY`
  - `GOOGLE_API_KEY` / `GEMINI_API_KEY`
- For local Ollama examples, run:
  - `ollama serve`
  - pull required models (for example `ollama pull llama3.1`)
