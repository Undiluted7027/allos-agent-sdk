#!/bin/bash

# examples/alias_workflow.sh
# Quick alias smoke workflow:
# 1) no-tools non-stream turn
# 2) no-tools stream turn
#
# Usage:
#   export ALLOS_ALIAS_PROVIDER="groq"
#   export ALLOS_ALIAS_MODEL="llama-3.1-8b-instant"
#   # optional:
#   # export ALLOS_ALIAS_PROMPT="Explain eventual consistency in one sentence."
#   # export ALLOS_ALIAS_STREAM_PROMPT="Give 3 short bullets on idempotency."
#   # export ALLOS_ALIAS_BASE_URL="https://custom-endpoint.example/v1"
#   # export ALLOS_ALIAS_API_KEY="sk-..."
#   bash examples/alias_workflow.sh

set -euo pipefail

DEFAULT_PROVIDER="groq"

default_model_for_provider() {
  case "$1" in
    groq) echo "llama-3.1-8b-instant" ;;
    together) echo "Qwen/Qwen2.5-7B-Instruct-Turbo" ;;
    mistral) echo "mistral-small-latest" ;;
    deepseek) echo "deepseek-chat" ;;
    openrouter) echo "openai/gpt-4o-mini" ;;
    portkey) echo "openai/gpt-4o-mini" ;;
    cohere_compat) echo "command-r7b-12-2024" ;;
    ollama_compat) echo "llama3.1:latest" ;;
    xai) echo "grok-3-mini" ;;
    *) echo "" ;;
  esac
}

PROVIDER="${ALLOS_ALIAS_PROVIDER:-$DEFAULT_PROVIDER}"
DEFAULT_MODEL="$(default_model_for_provider "$PROVIDER")"
MODEL="${ALLOS_ALIAS_MODEL:-$DEFAULT_MODEL}"
PROMPT="${ALLOS_ALIAS_PROMPT:-Explain the role of an API gateway in one short sentence.}"
STREAM_PROMPT="${ALLOS_ALIAS_STREAM_PROMPT:-Provide 3 short bullets on deterministic decoding.}"
BASE_URL="${ALLOS_ALIAS_BASE_URL:-}"
API_KEY="${ALLOS_ALIAS_API_KEY:-}"

if [[ -z "$MODEL" ]]; then
  echo "❌ No model set for provider '$PROVIDER'."
  echo "Set ALLOS_ALIAS_MODEL explicitly."
  exit 1
fi

echo "========================================================"
echo "🚀 ALLOS ALIAS WORKFLOW"
echo "========================================================"
echo "Provider: $PROVIDER"
echo "Model:    $MODEL"
echo

COMMON_ARGS=(--provider "$PROVIDER" --model "$MODEL" --no-tools)
if [[ -n "$BASE_URL" ]]; then
  COMMON_ARGS+=(--base-url "$BASE_URL")
fi
if [[ -n "$API_KEY" ]]; then
  COMMON_ARGS+=(--api-key "$API_KEY")
fi

echo "[1/2] Non-stream smoke test..."
allos "$PROMPT" "${COMMON_ARGS[@]}"

echo
echo "[2/2] Streaming smoke test..."
allos --stream "$STREAM_PROMPT" "${COMMON_ARGS[@]}"

echo
echo "✅ Alias workflow completed."
