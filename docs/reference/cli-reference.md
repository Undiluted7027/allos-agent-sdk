# CLI Reference

The `allos` command-line interface is the primary way to interact with the Allos Agent SDK for single tasks and interactive sessions.

## Synopsis

```bash
allos [OPTIONS] [PROMPT]...
```

The CLI is designed to be intuitive. You can either provide a prompt directly to the `allos` command or use one of the action flags like `--interactive`.

## Options

All options can be used before the main prompt.

#### `-h, --help`
Shows the main help message and exits.

#### `--list-providers`
Lists all available and registered LLM providers and exits.

#### `--list-tools`
Lists all available and registered tools, including their permission levels and descriptions, and exits.

#### `-i, --interactive`
Starts an interactive REPL session with the agent, allowing for a continuous, multi-turn conversation. If this flag is used, any `[PROMPT]` argument is ignored.

#### `-p, --provider <name>`
Specifies the LLM provider to use.
- **Choices:** `openai`, `anthropic`
- **Default:** `openai`

#### `-m, --model <model_name>`
Specifies the exact model name to use. If not provided, a sensible default will be chosen for the selected provider (e.g., `gpt-4o` for OpenAI).

#### `--tool <tool_name>`
Restricts the agent to using only the specified tool(s). This option can be used multiple times. If no tools are specified, the agent has access to all registered tools by default.
- **Example:** `allos --tool read_file --tool shell_exec "Read the content of setup.py"`

#### `-s, --session <filepath>`
Loads an agent session from a specified JSON file and saves the updated session back to the same file upon completion. If the file does not exist, a new one will be created upon saving.

#### `--auto-approve`
Automatically approves all tool execution requests that would normally require user confirmation (`ASK_USER` permission).
> [!WARNING]
> Use this option with extreme caution, especially with tools like `shell_exec`.

#### `-v, --verbose`
Enables verbose, DEBUG-level logging to the console. Useful for debugging agent behavior.

## Main Usage

### Running a Single Task

To run a single task, simply provide the prompt after the `allos` command.

```bash
# Use the default provider (OpenAI)
allos "Create a python script that prints 'Hello, World!'"

# Specify a different provider and model
allos --provider anthropic --model claude-3-5-sonnet-20240620 "Summarize the file 'main.py'"
```

### Starting an Interactive Session

Use the `-i` or `--interactive` flag to start a conversational session.

```bash
# Start a simple interactive session
allos --interactive

# Start an interactive session with Anthropic and a persistent session file
allos -i -p anthropic -s my_anthropic_session.json
```
Inside the interactive session, you can type `exit` or `quit` to end the session.

## Examples

#### Create a file
```bash
# The agent will ask for permission to run 'write_file'
allos "Create a file named 'app.py' with a simple Flask 'Hello World' app."
```

#### Use a session to continue a task
```bash
# First, create the file and save the context
allos -s project.json "Create a file 'test.py' with a function that adds two numbers."

# Next, load the session and continue the task
allos -s project.json "Now add a unit test for that function in the same file."
```
