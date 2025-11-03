#!/bin/bash

# examples/cli_workflow.sh
#
# A shell script demonstrating a complete end-to-end workflow using the Allos CLI.
# This script is interactive and will prompt you to approve tool usage.
#
# To run this script:
# 1. Make sure your virtual environment is active.
# 2. Make sure your .env file has both OPENAI_API_KEY and ANTHROPIC_API_KEY.
# 3. From the project root, run: bash examples/cli_workflow.sh

# --- Helper Functions ---
set -e # Exit immediately if a command exits with a non-zero status.

# Function to print a styled header
step() {
    echo ""
    echo "----------------------------------------------------------------------"
    echo "🚀 STEP: $1"
    echo "----------------------------------------------------------------------"
}

# Function to explain and run a command
run_command() {
    local cmd=$1
    local desc=$2
    echo ""
    echo "   Description: $desc"
    echo "   Command    : allos $cmd"
    echo ""
    read -p "   Press Enter to execute..."
    # The `allos` command must be available in the shell's PATH
    allos $cmd
}

# --- Main Script ---
echo "############################################################"
echo "### Starting Allos CLI End-to-End Workflow Demo"
echo "############################################################"

# --- Setup a clean workspace ---
WORKSPACE="cli_workspace"
step "Setup: Creating a clean workspace in './$WORKSPACE'"
if [ -d "$WORKSPACE" ]; then
    rm -rf "$WORKSPACE"
fi
mkdir "$WORKSPACE"
cd "$WORKSPACE"
echo "✅ Done. Now operating inside './$WORKSPACE'"

# Ensure cleanup happens on script exit or interruption
trap 'cd .. && rm -rf "$WORKSPACE" && echo -e "\n✅ Workspace cleaned up."' EXIT

# --- Step 1: Discovery ---
step "Discovery: Listing available providers and tools"
run_command "--list-providers" "List all available LLM providers."
run_command "--list-tools" "List all available built-in tools."

# --- Step 2: Create a file with OpenAI ---
step "Task 1: Create a Python script using OpenAI"
run_command "'Create a file named app.py with a function that returns the string \\\"Hello from Allos!\\\"'" \
            "The agent will use the 'write_file' tool. Please approve with 'y'."

# --- Step 3: Verify the file was created ---
step "Verification: Check that the file was created"
if [ -f "app.py" ]; then
    echo "✅ Success! 'app.py' exists. Contents:"
    echo "--------------------"
    cat app.py
    echo "--------------------"
else
    echo "❌ Failure! 'app.py' was not created."
    exit 1
fi

# --- Step 4: Modify the file using Anthropic and a session ---
step "Task 2: Modify the script using Anthropic and a session file"
run_command "--provider anthropic --session my_session.json 'Read app.py and add a line to call the function and print its result.'" \
            "The agent will use 'read_file' and 'edit_file' (or 'write_file'). Please approve with 'y'. The session will be saved."

# --- Step 5: Verify the modification ---
step "Verification: Check that the file was modified and session was saved"
if grep -q "print(" app.py; then
    echo "✅ Success! 'app.py' was modified. Contents:"
    echo "--------------------"
    cat app.py
    echo "--------------------"
else
    echo "❌ Failure! 'app.py' was not modified."
    exit 1
fi
if [ -f "my_session.json" ]; then
    echo "✅ Success! 'my_session.json' was created."
else
    echo "❌ Failure! Session file was not created."
    exit 1
fi

# --- Step 6: Execute the final script using the loaded session ---
step "Task 3: Execute the final script using the loaded session"
run_command "--session my_session.json 'Execute the app.py script and show the output.'" \
            "The agent will use 'shell_exec'. Please approve with 'y'. It loads the previous context."

echo ""
echo "############################################################"
echo "### ✅ Demo Complete!"
echo "############################################################"
