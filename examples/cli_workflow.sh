#!/bin/bash

# examples/cli_workflow.sh
# A guided tour of Allos capabilities.

set -e

function header() {
    echo -e "\n\033[1;34m════════════════════════════════════════════════════════════════\033[0m"
    echo -e "\033[1;36m $1 \033[0m"
    echo -e "\033[1;34m════════════════════════════════════════════════════════════════\033[0m"
}

function run_demo() {
    echo -e "\033[0;33m$ allos $1\033[0m"
    # Execute
    allos $1
}

header "1. DIAGNOSTICS"
echo "Checking which providers are configured in your environment..."
allos --active-providers

echo "Checking all supported providers..."
allos --list-providers

echo "Checking all supported tools..."
allos --list-tools

header "2. INSTANT SWITCHING (Groq)"
if [ -n "$GROQ_API_KEY" ]; then
    echo "Demonstrating high-speed inference..."
    run_demo "'Explain the concept of 'Agentic Workflow' in one sentence.' --provider groq --model llama-3.1-8b-instant --no-tools"
else
    echo "Skipping Groq (Key not found)"
fi

header "3. THE AGENTIC LOOP (OpenAI)"
if [ -n "$OPENAI_API_KEY" ]; then
    echo "Creating a file using tool execution..."
    run_demo "'Create a file named demo_script.py that prints 'Hello Allos'. Verify it exists.' --provider openai --model gpt-4o --auto-approve"
else
    echo "Skipping OpenAI (Key not found)"
fi

header "4. CLEANUP (Mistral): Should say that it can't execute the deletion command"
if [ -n "$MISTRAL_API_KEY" ]; then
    echo "Using a different provider to clean up..."
    run_demo "'Delete the file demo_script.py.' --provider mistral --model mistral-small-latest --auto-approve"
fi

echo -e "\n\033[1;32m✅ Demo Complete!\033[0m"
