# 5-Minute Quickstart

Create and run your first "Hello, World!" application using the Allos agent in under 5 minutes, directly from your command line.

## Prerequisites

1.  **Install Allos** with all provider dependencies:
    ```bash
    uv pip install "allos-agent-sdk[all]" python-dotenv
    ```
2.  **Choose your provider**:

    **Option A: Cloud Provider (OpenAI, Anthropic, Groq, Google)**

    Create a `.env` file in your project directory with your API key:
    ```env
    # .env
    OPENAI_API_KEY="your_openai_api_key_here"
    # Or: ANTHROPIC_API_KEY="your_key"
    # Or: GROQ_API_KEY="gsk_..."
    # Or: GEMINI_API_KEY="..."
    ```

    **Option B: Local Models (Ollama) - No API Key Required!**

    Install and start Ollama:
    ```bash
    # Install from https://ollama.com
    # Then pull a model and start the server
    ollama pull llama3.1
    ollama serve
    ```

## Step 1: Check Your Setup

Run the diagnostics command to see which providers are ready to use based on your environment variables.

```bash
allos --active-providers
```

You should see `[Ready]` next to the providers you have configured.

## Step 2: Create the Application File

Run the following command in your terminal. The agent will ask for your permission to write the file. **Type `y` and press Enter.**

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
│ Arguments: { ... }                                           │
╰──────────────────────────────────────────────────────────────╯
❓ Allow tool 'write_file' to run? (y/n): y
...
╭─────────────────────── Final Response ───────────────────────╮
│ Agent: The file `app.py` has been created successfully.      │
╰──────────────────────────────────────────────────────────────╯
```
You should now have a file named `app.py` in your directory.

## Step 3: Run the Application

Now, let's ask the agent to execute the script it just created. It will ask for permission again. **Type `y` and press Enter.**

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
│ Arguments: { "command": "python app.py" }                    │
╰──────────────────────────────────────────────────────────────╯
❓ Allow tool 'shell_exec' to run? (y/n): y
...
╭─────────────────────── Final Response ────────────────────────────────────╮
│ Agent: The script executed successfully and printed the following output: │
│ ```                                                                       │
│ Hello from Allos!                                                         │
│ ```                                                                       │
╰───────────────────────────────────────────────────────────────────────────╯
```

## Step 4: Try a Different Provider

### Option A: Fast Cloud Provider (Groq)

Allos makes it instant to switch providers. If you have a Groq key, try this for blazing fast speed:

```bash
allos "Explain how this python script works" \
  --provider groq \
  --model llama-3.1-8b-instant \
  --no-tools
```

> [!NOTE]
We used `--no-tools` here because smaller models often work better in pure chat mode.

### Option B: Local Model (Ollama) - Private & Free

Run completely local models with zero API costs and full privacy:

```bash
# Using native Ollama provider
allos "Explain how this python script works" \
  --provider ollama \
  --model llama3.1 \
  --no-tools
```

**Benefits:**
- ✅ **Free** - No API costs
- ✅ **Private** - Data never leaves your machine
- ✅ **Offline** - Works without internet
- ✅ **Tool Calling** - Full agent capabilities with compatible models

To see all available local models:
```bash
allos --list-ollama-models
```

For more Ollama examples, see [examples/ollama_usage.py](../../examples/ollama_usage.py).

---

Congratulations! You have successfully used the Allos agent.

For a more conversational experience, try `allos --interactive`.
