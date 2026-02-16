# Google Provider

The Google provider enables you to use Google's Gemini models through either the **Gemini API** (simple, API key-based) or **Vertex AI** (enterprise Google Cloud integration).

> [!IMPORTANT]
> The Google provider requires **Python 3.10 or higher** due to dependencies on the Google GenAI SDK and [google-auth library](https://github.com/googleapis/google-auth-library-python).

## Two Ways to Use Google's Gemini Models

| Mode | Description | Best For | Authentication |
|------|-------------|----------|----------------|
| **Gemini API** | Consumer API with simple API key | Quick start, development, smaller projects | API Key (`GOOGLE_API_KEY` or `GEMINI_API_KEY`) |
| **Vertex AI** | Enterprise Google Cloud service | Production, enterprise, Google Cloud users | Service Accounts, ADC, IAM |

## Prerequisites

- Python 3.10 or higher
- One of the following:
  - **Gemini API**: API key from [Google AI Studio](https://aistudio.google.com)
  - **Vertex AI**: Google Cloud project with Vertex AI API enabled

## Installation

```bash
# Install with Google provider support (requires Python 3.10+)
uv pip install "allos-agent-sdk[google]"

# Or if using all providers
uv pip install "allos-agent-sdk[all]"
```

---

## Gemini API Mode (Recommended for Getting Started)

The simplest way to use Google's Gemini models with an API key.

### Configuration

Set your API key as an environment variable:

```bash
export GOOGLE_API_KEY="your-api-key-here"
# Or use the alternative variable name
export GEMINI_API_KEY="your-api-key-here"
```

### CLI Usage

```bash
# Basic usage with Gemini API
allos "Explain quantum computing" --provider google --model gemini-2.0-flash

# With streaming
allos --stream "Write a story about AI" --provider google --model gemini-1.5-pro
```

### Python API

```python
from allos import Agent, AgentConfig

config = AgentConfig(
    provider_name="google",
    model="gemini-2.0-flash",
)

agent = Agent(config)
response = agent.run("What is the capital of France?")
print(response)
```

### Using the Provider Directly

```python
from allos.providers import ProviderRegistry, Message, MessageRole

# Get the Google provider (Gemini API mode)
provider = ProviderRegistry.get_provider(
    "google",
    model="gemini-2.0-flash",
    api_key="your-api-key"  # Optional if GOOGLE_API_KEY is set
)

messages = [
    Message(role=MessageRole.USER, content="What is machine learning?")
]

response = provider.chat(messages)
print(response.content)
```

---

## Vertex AI Mode (For Enterprise/Google Cloud Users)

Vertex AI is Google Cloud's enterprise ML platform with advanced features, IAM integration, and better compliance support.

### Prerequisites

1. Google Cloud project with Vertex AI API enabled
2. One of the following authentication methods configured:
   - Service Account JSON file
   - Application Default Credentials (ADC)
   - Service Account impersonation

### Authentication Methods

#### Method 1: Service Account File

```bash
# Set environment variable
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"

# Or in Python
from allos.providers import ProviderRegistry

provider = ProviderRegistry.get_provider(
    "google",
    model="gemini-2.0-flash",
    vertexai=True,
    project="your-gcp-project-id",
    credentials_path="/path/to/service-account.json"
)
```

#### Method 2: Application Default Credentials (ADC)

Best for local development and Cloud Shell:

```bash
# Authenticate with gcloud
gcloud auth application-default login

# Set your project
export GOOGLE_CLOUD_PROJECT="your-project-id"
```

```python
from allos.providers import ProviderRegistry

provider = ProviderRegistry.get_provider(
    "google",
    model="gemini-2.0-flash",
    vertexai=True,
    project="your-gcp-project-id",  # Or use GOOGLE_CLOUD_PROJECT env var
)
```

#### Method 3: Service Account Impersonation

For fine-grained access control:

```python
from allos.providers import ProviderRegistry

provider = ProviderRegistry.get_provider(
    "google",
    model="gemini-2.0-flash",
    vertexai=True,
    project="your-gcp-project-id",
    impersonate_service_account="sa@your-project.iam.gserviceaccount.com"
)
```

#### Method 4: Explicit Credentials Object

```python
from google.oauth2 import service_account
from allos.providers import ProviderRegistry

credentials = service_account.Credentials.from_service_account_file(
    "/path/to/service-account.json",
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)

provider = ProviderRegistry.get_provider(
    "google",
    model="gemini-2.0-flash",
    vertexai=True,
    project="your-gcp-project-id",
    credentials=credentials
)
```

#### Method 5: Service Account JSON Content

```python
import json
from allos.providers import ProviderRegistry

with open("/path/to/service-account.json") as f:
    sa_json = json.load(f)

provider = ProviderRegistry.get_provider(
    "google",
    model="gemini-2.0-flash",
    vertexai=True,
    credentials_json=sa_json  # Can be dict or JSON string
)
```

### Configuration

```python
from allos import Agent, AgentConfig

# AgentConfig supports core SDK fields only.
# For Google-specific options (vertexai/project/location/credentials),
# use ProviderRegistry.get_provider(...) directly.
config = AgentConfig(
    provider_name="google",
    model="gemini-2.0-flash",
)

agent = Agent(config)
response = agent.run("Summarize this architecture.")
```

> [!IMPORTANT] Agent vs Direct Provider
> - Use `Agent + AgentConfig` for standard agent workflows.
> - `AgentConfig` does **not** accept provider-specific kwargs (for example: `vertexai`, `project`, `location`, `credentials_path`, `credentials_json`, `impersonate_service_account`).
> - For advanced Google/Vertex AI configuration, initialize the provider directly with `ProviderRegistry.get_provider(...)`.

```python
from allos.providers import ProviderRegistry, Message, MessageRole

provider = ProviderRegistry.get_provider(
    "google",
    model="gemini-2.0-flash",
    vertexai=True,
    project="your-gcp-project-id",
    location="us-central1",
    credentials_path="/path/to/service-account.json",
)

messages = [Message(role=MessageRole.USER, content="Explain Kubernetes autoscaling.")]
response = provider.chat(messages)
print(response.content)
```

### Regional Endpoints

Vertex AI supports multiple regions:

```python
provider = ProviderRegistry.get_provider(
    "google",
    model="gemini-2.0-flash",
    vertexai=True,
    project="your-project",
    location="europe-west1"  # Or us-central1, asia-southeast1, etc.
)
```

---

## Supported Models

| Model | Context Window | Best For |
|-------|---------------|----------|
| `gemini-3-flash-preview` | 1,048,576 tokens | Speed, scale |
| `gemini-3-pro-preview` | 1,048,576 tokens | Multimodality, agentic |
| `gemini-2.5-flash` | 1,048,576 tokens | Low-latency, cost-effective |
| `gemini-2.5-pro` | 1,048,576 tokens | Complex problems |
| `gemini-2.5-flash-lite` | 1,048,576 tokens | Fastest model |

For the latest model availability, refer to:
- [Gemini API Models](https://ai.google.dev/gemini-api/docs/models)
- [Vertex AI Models](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models)

---

## Advanced Configuration

### Custom System Instructions

```python
from allos.providers import ProviderRegistry, Message, MessageRole

provider = ProviderRegistry.get_provider("google", model="gemini-2.0-flash")

messages = [
    Message(role=MessageRole.SYSTEM, content="You are a helpful coding assistant."),
    Message(role=MessageRole.USER, content="Write a Python function to sort a list")
]

response = provider.chat(messages)
```

### Streaming Responses

```python
from allos import Agent, AgentConfig

config = AgentConfig(
    provider_name="google",
    model="gemini-2.0-flash",
)

agent = Agent(config)

# Streaming is handled automatically
for chunk in agent.stream_run("Tell me a story"):
    if chunk.content:
        print(chunk.content, end="", flush=True)
```

### Tool Calling

The Google provider fully supports function/tool calling:

```python
from allos import Agent, AgentConfig

config = AgentConfig(
    provider_name="google",
    model="gemini-2.0-flash",
    tool_names=["read_file", "write_file", "shell_exec"]
)

agent = Agent(config)
response = agent.run("Read the file 'data.json' and tell me what it contains")
```

### What are Thought Signatures?

Gemini 3.x models use "thought signatures" - encrypted representations of the model's internal
reasoning process. These are required during function calling to maintain reasoning context
across multiple steps.

#### How Allos Handles Them

The Allos SDK **automatically manages thought signatures** for you:

✅ Extracted from model responses
✅ Stored in conversation history
✅ Sent back to the API in subsequent requests
✅ Validated by the Google API

**You don't need to do anything!** Just use the Agent or Provider normally.

#### When You Might See Errors

If you see an error like:
```
Function call is missing a thought_signature in functionCall parts
```

**Causes**:
1. Using Gemini 3.x models with an outdated version of the Allos SDK
2. Manually constructing messages without preserving thought signatures
3. Using a custom tool execution loop that doesn't preserve message metadata

**Solutions**:
1. Update to the latest Allos SDK version: `uv pip install --upgrade allos-agent-sdk`
2. Use the `Agent` class instead of manual provider calls
3. If using providers directly, ensure you preserve `thought_signatures` in Message objects

#### Model Behavior Differences

| Feature | Gemini 3.x | Gemini 2.5.x | Gemini 2.0.x |
|---------|------------|--------------|--------------|
| Thought Signatures | **Required** for function calling | Optional (recommended) | Not used |
| Validation | Strict (400 error if missing) | No validation | N/A |
| Context Preservation | Mandatory | Improves quality | N/A |

**Recommendation**: Use Gemini 2.0-flash or 2.5-flash unless you specifically need Gemini 3.x features.

---

## Authentication Precedence

The Google provider checks for authentication in this order:

1. **Explicit `credentials` parameter** (Python Credentials object)
2. **`credentials_path` parameter** (path to service account JSON)
3. **`GOOGLE_APPLICATION_CREDENTIALS`** environment variable
4. **`credentials_json` parameter** (JSON string or dict)
5. **Service account impersonation** (`impersonate_service_account`)
6. **Application Default Credentials (ADC)** (fallback)

For **Gemini API mode**, only checks:
1. **`api_key` parameter**
2. **`GOOGLE_API_KEY`** environment variable
3. **`GEMINI_API_KEY`** environment variable

---

## Troubleshooting

### Python Version Error

**Error**: `ImportError: Google provider requires Python 3.10 or higher`

**Solution**: Upgrade to Python 3.10+:
```bash
# Check your Python version
python --version

# If using pyenv
pyenv install 3.10.16
pyenv global 3.10.16

# If using uv
uv python install 3.10
```

### Authentication Errors (Gemini API)

**Error**: `Authentication or configuration error: Invalid API key`

**Solution**:
1. Verify your API key at [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Check environment variable is set: `echo $GOOGLE_API_KEY`
3. Ensure no spaces or quotes in the key

### Authentication Errors (Vertex AI)

**Error**: `Application Default Credentials (ADC) not found`

**Solution**:
```bash
# Option 1: Use gcloud CLI
gcloud auth application-default login

# Option 2: Set service account
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/sa.json"

# Option 3: Use explicit credentials in code (see examples above)
```

### Project ID Missing

**Error**: `Vertex AI requires a project ID`

**Solution**:
```bash
# Set via environment variable
export GOOGLE_CLOUD_PROJECT="your-project-id"

# Or provide in code
provider = ProviderRegistry.get_provider(
    "google",
    model="gemini-2.0-flash",
    vertexai=True,
    project="your-project-id"
)
```

### Permission Errors

**Error**: `Failed to impersonate service account: Permission denied`

**Solution**:
- Ensure your source credentials have the `roles/iam.serviceAccountTokenCreator` role
- Check IAM permissions in Google Cloud Console

### Model Not Available

**Error**: `Model 'gemini-x.x' is not available`

**Solution**:
1. Check model name spelling
2. Verify model is available in your region (Vertex AI only)
3. Ensure Vertex AI API is enabled in your project
4. Check [model availability documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models)

---

## Checking Provider Status

```bash
# Check if Google provider is available
allos --active-providers

# Expected output (if Python 3.10+ and dependencies installed):
# Available Providers:
#   ✓ google (GOOGLE_API_KEY or Vertex AI configured)
```

---

## Environment Variables Reference

### Gemini API Mode
- `GOOGLE_API_KEY` - API key for Gemini API (preferred)
- `GEMINI_API_KEY` - Alternative API key variable

### Vertex AI Mode
- `GOOGLE_CLOUD_PROJECT` - GCP project ID
- `GOOGLE_CLOUD_LOCATION` - Region (default: `us-central1`)
- `GOOGLE_APPLICATION_CREDENTIALS` - Path to service account JSON file

---

## Best Practices

1. **Use Gemini API for Development**: Simpler setup, faster iteration
2. **Use Vertex AI for Production**: Better compliance, IAM integration, enterprise support
3. **Choose the Right Model**:
   - `gemini-2.0-flash` or `2.5-flash`: Fast, cost-effective
   - `gemini-1.5-pro`: Highest quality, largest context
4. **Handle Rate Limits**: Implement exponential backoff for production workloads
5. **Use Streaming**: For better user experience with long responses
6. **Secure API Keys**: Never commit API keys to version control

---

## Additional Resources

- [Google AI Studio](https://aistudio.google.com/) - Get Gemini API keys
- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [Gemini API Documentation](https://ai.google.dev/gemini-api/docs)
- [Google Auth Library](https://google-auth.readthedocs.io/)
- [Model Pricing](https://cloud.google.com/vertex-ai/generative-ai/pricing)

---

## See Also

- [OpenAI Provider](./openai.md)
- [Anthropic Provider](./anthropic.md)
- [Ollama Provider](./ollama.md) - For local models
- [Chat Completions Provider](./chat-completions.md) - For OpenAI-compatible APIs
