# 5-Minute Quickstart

Get your first "Hello, World!" app created and running with the Allos agent in under 5 minutes, directly from your command line.

## Prerequisites

1.  **Install Allos** with all provider dependencies:
    ```bash
    uv pip install "allos-agent-sdk[all]" python-dotenv
    ```
2.  **Create a `.env` file** in your project directory with your API key. For this example, we'll use OpenAI.
    ```env
    # .env
    OPENAI_API_KEY="your_openai_api_key_here"
    ```

## Step 1: Create the File

Run the following command in your terminal. The agent will ask for your permission to write the file. Type `y` and press Enter.

```bash
allos "Create a Python file named 'app.py' that prints 'Hello from Allos!'"
```

**Expected Interaction:**
```text
Model not specified, defaulting to 'gpt-4o' for provider 'openai'.
╭────────────────────────── Input ──────────────────────────────────────────╮
│ User: Create a Python file named 'app.py' that prints 'Hello from Allos!' │
╰───────────────────────────────────────────────────────────────────────────╯
🧠 Thinking...
╭──────────────────── Tool Call Requested ─────────────────────╮
│ Tool: write_file                                             │
│ Arguments: {                                                 │
│   "path": "app.py",                                          │
│   "content": "print('Hello from Allos!')"                    │
│ }                                                            │
╰──────────────────────────────────────────────────────────────╯
❓ Allow tool 'write_file' to run? (y/n): y
╭────────────────── Tool Result: write_file ───────────────────╮
│ {                                                            │
│   "status": "success",                                       │
│   "message": "Successfully wrote 24 bytes to 'app.py'."      │
│ }                                                            │
╰──────────────────────────────────────────────────────────────╯
🧠 Thinking...
╭─────────────────────── Final Response ───────────────────────╮
│ Agent: The file `app.py` has been created successfully.      │
╰──────────────────────────────────────────────────────────────╯
```
You should now have a file named `app.py` in your directory.

## Step 2: Run the File

Now, let's ask the agent to execute the script it just created. It will ask for permission again.

```bash
allos "Execute the 'app.py' script using python."
```

**Expected Interaction:**
```text
╭────────────────────────── Input ───────────────────────────╮
│ User: Execute the 'app.py' script using python.            │
╰────────────────────────────────────────────────────────────╯
🧠 Thinking...
╭──────────────────── Tool Call Requested ─────────────────────╮
│ Tool: shell_exec                                             │
│ Arguments: {                                                 │
│   "command": "python app.py"                                 │
│ }                                                            │
╰──────────────────────────────────────────────────────────────╯
❓ Allow tool 'shell_exec' to run? (y/n): y
╭────────────────── Tool Result: shell_exec ───────────────────╮
│ {                                                            │
│   "status": "success",                                       │
│   "return_code": 0,                                          │
│   "stdout": "Hello from Allos!\n",                           │
│   "stderr": ""                                               │
│ }                                                            │
╰──────────────────────────────────────────────────────────────╯
🧠 Thinking...
╭─────────────────────── Final Response ────────────────────────────────────╮
│ Agent: The script executed successfully and printed the following output: │
│                                                                           │
│ ```                                                                       │
│ Hello from Allos!                                                         │
│ ```                                                                       │
╰───────────────────────────────────────────────────────────────────────────╯
```

Congratulations! You have successfully used the Allos agent to write and execute code.

For a more conversational experience, try `allos --interactive`.
