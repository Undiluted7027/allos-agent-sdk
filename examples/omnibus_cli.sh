#!/bin/bash

# examples/omnibus_cli.sh
# Omnibus (CLI Version)
# Comprehensive test encompassing 5 providers (Groq, Mistral, Together AI, OpenAI, Anthropic).
# This script demonstrates the power of the Allos CLI by chaining together multiple
# providers in a single session, passing context between them, and utilizing their
# unique strengths (speed, planning, coding, execution, review).

set -e # Exit immediately on error

echo "========================================================"
echo "🚀 ALLOS CLI OMNIBUS TEST"
echo "========================================================"

SESSION_FILE="cli_omnibus.json"
rm -f "$SESSION_FILE" "cli_demo.py"

# --- 1. GROQ (The Ideator) ---
# Testing: --no-tools, alias provider
echo -e "\n[1/5] GROQ: Generating Idea (--no-tools)..."
allos "Give me ONE simple Python automation idea involving file manipulation. Just the idea." \
  --provider groq \
  --model "llama-3.1-8b-instant" \
  --no-tools \
  --session "$SESSION_FILE"

# --- 2. MISTRAL (The Planner) ---
# Testing: Provider switching, session loading
echo -e "\n[2/5] MISTRAL: Creating Plan..."
allos "Create a step-by-step implementation plan for that idea." \
  --provider mistral \
  --model "mistral-small-latest" \
  --session "$SESSION_FILE"

# --- 3. TOGETHER AI (The Coder) ---
# Testing: Explicit 'chat_completions' provider, manual base-url, manual api-key
echo -e "\n[3/5] TOGETHER: Writing Code (Explicit Adapter)..."
allos "Write the Python code for this plan. Do not execute it." \
  --provider chat_completions \
  --model "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo" \
  --base-url "https://api.together.xyz/v1" \
  --api-key "$TOGETHER_API_KEY" \
  --session "$SESSION_FILE" \

# --- 4. OPENAI (The Executor) ---
# Testing: Max tokens, Tools enabled (default), Auto-approve
echo -e "\n[4/5] OPENAI: Executing Code (--max-tokens, --auto-approve)..."
allos "Save that code to 'cli_demo.py' and execute it. Report the output." \
  --provider openai \
  --model "gpt-4o" \
  --max-tokens 2000 \
  --auto-approve \
  --session "$SESSION_FILE" \
  --tool "write_file" \
  --verbose

# --- 5. ANTHROPIC: The Reviewer (Testing Native Provider & Max Tokens) ---
# We switch to Claude to review the output. This tests the Anthropic provider
# correctly handling a context populated by OpenAI tools.
echo -e "\n[5/5] ANTHROPIC: Reviewing Output (Native Provider) ..."
allos "Analyze the output of the script execution. Did it work as expected? Be brief." \
  --provider anthropic \
  --model "claude-3-haiku-20240307" \
  --session "$SESSION_FILE" \
  --max-tokens 1000 \
  --verbose

# --- Verification ---
if [ -f "cli_demo.py" ]; then
    echo -e "\n✅ SUCCESS: 'cli_demo.py' exists."
    rm "cli_demo.py" "$SESSION_FILE"
    exit 0
else
    echo -e "\n❌ FAILURE: 'cli_demo.py' was not created."
    exit 1
fi
