# API Reference: OpenAI & Anthropic Streaming

This document summarizes the streaming behaviors, event structures, and tool calling mechanisms for OpenAI and Anthropic APIs, verified against their Python SDKs.

## 1. OpenAI Responses API (`/v1/responses`)

> **Note:** This API is distinct from the standard Chat Completions API and uses a different signature in the SDK.

### Streaming Interface
To enable streaming, use the `stream=True` parameter. The SDK method is `client.responses.create`.

**Signature:**
```python
client.responses.create(
    model="gpt-4o",
    input="User prompt here",  # Note: Uses 'input' instead of 'messages'
    tools=[...],
    stream=True
)
```

### Event Structure
The stream yields event objects (e.g., `ResponseStreamEvent`).
*   **Events:** The stream provides lifecycle events.
*   **Tool Calls:** Tool calls are part of the event stream.
*   **Delta:** Events containing updates have a `delta` property.

### Tool Calling
Tool definitions are passed via the `tools` parameter. The structure is strict (validation errors occur if `name` is missing in the wrong place).

### Token Usage
Usage metadata is typically available in the final event or a dedicated usage event, similar to Chat Completions.

**Manual Verification Script:**
```python
import inspect
import os

from openai import OpenAI

# Initialize client
# Note: Ensure OPENAI_API_KEY is set
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

try:
    print("Calling client.responses.create with 'input' parameter...")

    stream = client.responses.create(
        model="gpt-4o",
        input="What is the weather in San Francisco?",
        tools=[
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get the weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "unit": {"type": "string", "enum": ["c", "f"]},
                    },
                    "required": ["location"],
                },
            }
        ],
        stream=True,
    )

    print("Stream created. Iterating events...")
    for event in stream:
        print(f"Event Type: {type(event)}")
        # Try to inspect event
        try:
            # Check for common event attributes in this new API
            if hasattr(event, "type"):
                print(f"  Type: {event.type}")

            # If it's a tool call event
            if hasattr(event, "delta") and hasattr(event.delta, "tool_calls"):
                print(f"  Tool Call Delta: {event.delta.tool_calls}")

            # Dump full event if possible
            if hasattr(event, "model_dump_json"):
                print(f"  JSON: {event.model_dump_json()}")
            else:
                print(f"  Raw: {event}")

        except Exception as inner_e:
            print(f"Error inspecting event: {inner_e}")

except Exception as e:
    print(f"An error occurred: {e}")
```

**Output:**
```shell
python verify_openai_responses.py
Calling client.responses.create with 'input' parameter...
Stream created. Iterating events...
Event Type: <class 'openai.types.responses.response_created_event.ResponseCreatedEvent'>
  Type: response.created
  JSON: {"response":{"id":"resp_03e355cd3f8c866c00693100aa17ec81979a0c79eb9be10c2d","created_at":1764819114.0,"error":null,"incomplete_details":null,"instructions":null,"metadata":{},"model":"gpt-4o-2024-08-06","object":"response","output":[],"parallel_tool_calls":true,"temperature":1.0,"tool_choice":"auto","tools":[{"name":"get_weather","parameters":{"type":"object","properties":{"location":{"type":"string"},"unit":{"type":"string","enum":["c","f"]}},"required":["location","unit"],"additionalProperties":false},"strict":true,"type":"function","description":"Get the weather for a location"}],"top_p":1.0,"background":false,"conversation":null,"max_output_tokens":null,"max_tool_calls":null,"previous_response_id":null,"prompt":null,"prompt_cache_key":null,"reasoning":{"effort":null,"generate_summary":null,"summary":null},"safety_identifier":null,"service_tier":"auto","status":"in_progress","text":{"format":{"type":"text"},"verbosity":"medium"},"top_logprobs":0,"truncation":"disabled","usage":null,"user":null,"prompt_cache_retention":null,"store":true},"sequence_number":0,"type":"response.created"}
Event Type: <class 'openai.types.responses.response_in_progress_event.ResponseInProgressEvent'>
  Type: response.in_progress
  JSON: {"response":{"id":"resp_03e355cd3f8c866c00693100aa17ec81979a0c79eb9be10c2d","created_at":1764819114.0,"error":null,"incomplete_details":null,"instructions":null,"metadata":{},"model":"gpt-4o-2024-08-06","object":"response","output":[],"parallel_tool_calls":true,"temperature":1.0,"tool_choice":"auto","tools":[{"name":"get_weather","parameters":{"type":"object","properties":{"location":{"type":"string"},"unit":{"type":"string","enum":["c","f"]}},"required":["location","unit"],"additionalProperties":false},"strict":true,"type":"function","description":"Get the weather for a location"}],"top_p":1.0,"background":false,"conversation":null,"max_output_tokens":null,"max_tool_calls":null,"previous_response_id":null,"prompt":null,"prompt_cache_key":null,"reasoning":{"effort":null,"generate_summary":null,"summary":null},"safety_identifier":null,"service_tier":"auto","status":"in_progress","text":{"format":{"type":"text"},"verbosity":"medium"},"top_logprobs":0,"truncation":"disabled","usage":null,"user":null,"prompt_cache_retention":null,"store":true},"sequence_number":1,"type":"response.in_progress"}
Event Type: <class 'openai.types.responses.response_output_item_added_event.ResponseOutputItemAddedEvent'>
  Type: response.output_item.added
  JSON: {"item":{"arguments":"","call_id":"call_yaZU0crtwWLnlBU5qLR9VtBV","name":"get_weather","type":"function_call","id":"fc_03e355cd3f8c866c00693100aab7748197bf30451656f4aea2","status":"in_progress"},"output_index":0,"sequence_number":2,"type":"response.output_item.added"}
Event Type: <class 'openai.types.responses.response_function_call_arguments_delta_event.ResponseFunctionCallArgumentsDeltaEvent'>
  Type: response.function_call_arguments.delta
  JSON: {"delta":"{\"","item_id":"fc_03e355cd3f8c866c00693100aab7748197bf30451656f4aea2","output_index":0,"sequence_number":3,"type":"response.function_call_arguments.delta","obfuscation":"LrGckESS8Y8j8r"}
Event Type: <class 'openai.types.responses.response_function_call_arguments_delta_event.ResponseFunctionCallArgumentsDeltaEvent'>
  Type: response.function_call_arguments.delta
  JSON: {"delta":"location","item_id":"fc_03e355cd3f8c866c00693100aab7748197bf30451656f4aea2","output_index":0,"sequence_number":4,"type":"response.function_call_arguments.delta","obfuscation":"3fzhpg3V"}
Event Type: <class 'openai.types.responses.response_function_call_arguments_delta_event.ResponseFunctionCallArgumentsDeltaEvent'>
  Type: response.function_call_arguments.delta
  JSON: {"delta":"\":\"","item_id":"fc_03e355cd3f8c866c00693100aab7748197bf30451656f4aea2","output_index":0,"sequence_number":5,"type":"response.function_call_arguments.delta","obfuscation":"poR9AWUrHnPyu"}
Event Type: <class 'openai.types.responses.response_function_call_arguments_delta_event.ResponseFunctionCallArgumentsDeltaEvent'>
  Type: response.function_call_arguments.delta
  JSON: {"delta":"San","item_id":"fc_03e355cd3f8c866c00693100aab7748197bf30451656f4aea2","output_index":0,"sequence_number":6,"type":"response.function_call_arguments.delta","obfuscation":"Xvwv8sZhOSqm8"}
Event Type: <class 'openai.types.responses.response_function_call_arguments_delta_event.ResponseFunctionCallArgumentsDeltaEvent'>
  Type: response.function_call_arguments.delta
  JSON: {"delta":" Francisco","item_id":"fc_03e355cd3f8c866c00693100aab7748197bf30451656f4aea2","output_index":0,"sequence_number":7,"type":"response.function_call_arguments.delta","obfuscation":"pcZQzl"}
Event Type: <class 'openai.types.responses.response_function_call_arguments_delta_event.ResponseFunctionCallArgumentsDeltaEvent'>
  Type: response.function_call_arguments.delta
  JSON: {"delta":"\",\"","item_id":"fc_03e355cd3f8c866c00693100aab7748197bf30451656f4aea2","output_index":0,"sequence_number":8,"type":"response.function_call_arguments.delta","obfuscation":"HiPW1MWIwqb42"}
Event Type: <class 'openai.types.responses.response_function_call_arguments_delta_event.ResponseFunctionCallArgumentsDeltaEvent'>
  Type: response.function_call_arguments.delta
  JSON: {"delta":"unit","item_id":"fc_03e355cd3f8c866c00693100aab7748197bf30451656f4aea2","output_index":0,"sequence_number":9,"type":"response.function_call_arguments.delta","obfuscation":"IHvPJ1tGQWWg"}
Event Type: <class 'openai.types.responses.response_function_call_arguments_delta_event.ResponseFunctionCallArgumentsDeltaEvent'>
  Type: response.function_call_arguments.delta
  JSON: {"delta":"\":\"","item_id":"fc_03e355cd3f8c866c00693100aab7748197bf30451656f4aea2","output_index":0,"sequence_number":10,"type":"response.function_call_arguments.delta","obfuscation":"LNaxS4wdxjWzs"}
Event Type: <class 'openai.types.responses.response_function_call_arguments_delta_event.ResponseFunctionCallArgumentsDeltaEvent'>
  Type: response.function_call_arguments.delta
  JSON: {"delta":"f","item_id":"fc_03e355cd3f8c866c00693100aab7748197bf30451656f4aea2","output_index":0,"sequence_number":11,"type":"response.function_call_arguments.delta","obfuscation":"M40TQ0pkevti9QA"}
Event Type: <class 'openai.types.responses.response_function_call_arguments_delta_event.ResponseFunctionCallArgumentsDeltaEvent'>
  Type: response.function_call_arguments.delta
  JSON: {"delta":"\"}","item_id":"fc_03e355cd3f8c866c00693100aab7748197bf30451656f4aea2","output_index":0,"sequence_number":12,"type":"response.function_call_arguments.delta","obfuscation":"KYKM2bpuA51PgS"}
Event Type: <class 'openai.types.responses.response_function_call_arguments_done_event.ResponseFunctionCallArgumentsDoneEvent'>
  Type: response.function_call_arguments.done
  JSON: {"arguments":"{\"location\":\"San Francisco\",\"unit\":\"f\"}","item_id":"fc_03e355cd3f8c866c00693100aab7748197bf30451656f4aea2","name":null,"output_index":0,"sequence_number":13,"type":"response.function_call_arguments.done"}
Event Type: <class 'openai.types.responses.response_output_item_done_event.ResponseOutputItemDoneEvent'>
  Type: response.output_item.done
  JSON: {"item":{"arguments":"{\"location\":\"San Francisco\",\"unit\":\"f\"}","call_id":"call_yaZU0crtwWLnlBU5qLR9VtBV","name":"get_weather","type":"function_call","id":"fc_03e355cd3f8c866c00693100aab7748197bf30451656f4aea2","status":"completed"},"output_index":0,"sequence_number":14,"type":"response.output_item.done"}
Event Type: <class 'openai.types.responses.response_completed_event.ResponseCompletedEvent'>
  Type: response.completed
  JSON: {"response":{"id":"resp_03e355cd3f8c866c00693100aa17ec81979a0c79eb9be10c2d","created_at":1764819114.0,"error":null,"incomplete_details":null,"instructions":null,"metadata":{},"model":"gpt-4o-2024-08-06","object":"response","output":[{"arguments":"{\"location\":\"San Francisco\",\"unit\":\"f\"}","call_id":"call_yaZU0crtwWLnlBU5qLR9VtBV","name":"get_weather","type":"function_call","id":"fc_03e355cd3f8c866c00693100aab7748197bf30451656f4aea2","status":"completed"}],"parallel_tool_calls":true,"temperature":1.0,"tool_choice":"auto","tools":[{"name":"get_weather","parameters":{"type":"object","properties":{"location":{"type":"string"},"unit":{"type":"string","enum":["c","f"]}},"required":["location","unit"],"additionalProperties":false},"strict":true,"type":"function","description":"Get the weather for a location"}],"top_p":1.0,"background":false,"conversation":null,"max_output_tokens":null,"max_tool_calls":null,"previous_response_id":null,"prompt":null,"prompt_cache_key":null,"reasoning":{"effort":null,"generate_summary":null,"summary":null},"safety_identifier":null,"service_tier":"default","status":"completed","text":{"format":{"type":"text"},"verbosity":"medium"},"top_logprobs":0,"truncation":"disabled","usage":{"input_tokens":56,"input_tokens_details":{"cached_tokens":0},"output_tokens":20,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":76},"user":null,"prompt_cache_retention":null,"store":true},"sequence_number":15,"type":"response.completed"}
```

---

## 2. OpenAI Chat Completions API (`/v1/chat/completions`)

### Delta Structure (`tool_calls`)
In streaming mode, `tool_calls` are delivered incrementally within the `delta` object.

**Structure:**
```json
{
  "choices": [
    {
      "delta": {
        "tool_calls": [
          {
            "index": 0,
            "id": "call_xyz...",       # Present in first chunk
            "function": {
              "name": "get_weather",   # Present in early chunks
              "arguments": "{\"loc"    # Streamed incrementally
            }
          }
        ]
      }
    }
  ]
}
```
*   **Assembly:** You must accumulate `arguments` by `index`. The `id` and `name` are usually sent once (or in the first few chunks).

### Usage Options
To receive token usage in streaming, set `stream_options={"include_usage": True}`.

**Behavior:**
*   The usage data appears in the **last chunk** of the stream.
*   This chunk will have `choices: []` (empty choices) and a populated `usage` field.

**Example:**
```python
stream = client.chat.completions.create(
    ...,
    stream=True,
    stream_options={"include_usage": True}
)
```

**Manual Script for Verification:**
```python
import json
import os

from openai import OpenAI

# Initialize client
# Note: Ensure OPENAI_API_KEY is set
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

print("--- Testing OpenAI Chat Completions API Streaming (SDK) ---")

try:
    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "What is the weather in Paris?"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                            "unit": {"type": "string", "enum": ["c", "f"]},
                        },
                        "required": ["location"],
                    },
                },
            }
        ],
        tool_choice="auto",
        stream=True,
        stream_options={"include_usage": True},
    )

    print("Stream created. Iterating chunks...")

    tool_calls_buffer = {}

    for chunk in stream:
        # Check for usage
        if chunk.usage:
            print(f"Usage received: {chunk.usage}")

        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta

        if delta.tool_calls:
            print(f"Tool Call Delta: {delta.tool_calls}")
            for tc in delta.tool_calls:
                index = tc.index
                if index not in tool_calls_buffer:
                    tool_calls_buffer[index] = {"id": "", "name": "", "arguments": ""}

                if tc.id:
                    tool_calls_buffer[index]["id"] += tc.id
                if tc.function:
                    if tc.function.name:
                        tool_calls_buffer[index]["name"] += tc.function.name
                    if tc.function.arguments:
                        tool_calls_buffer[index]["arguments"] += tc.function.arguments

    print("\nReconstructed Tool Calls:")
    print(json.dumps(tool_calls_buffer, indent=2))

except Exception as e:
    print(f"An error occurred: {e}")
```

**Output:**
```shell
python verify_openai_chat.py
--- Testing OpenAI Chat Completions API Streaming (SDK) ---
Stream created. Iterating chunks...
Tool Call Delta: [ChoiceDeltaToolCall(index=0, id='call_hjq9uBSN660V2MXXR3sGzNYe', function=ChoiceDeltaToolCallFunction(arguments='', name='get_weather'), type='function')]
Tool Call Delta: [ChoiceDeltaToolCall(index=0, id=None, function=ChoiceDeltaToolCallFunction(arguments='{"', name=None), type=None)]
Tool Call Delta: [ChoiceDeltaToolCall(index=0, id=None, function=ChoiceDeltaToolCallFunction(arguments='location', name=None), type=None)]
Tool Call Delta: [ChoiceDeltaToolCall(index=0, id=None, function=ChoiceDeltaToolCallFunction(arguments='":"', name=None), type=None)]
Tool Call Delta: [ChoiceDeltaToolCall(index=0, id=None, function=ChoiceDeltaToolCallFunction(arguments='Paris', name=None), type=None)]
Tool Call Delta: [ChoiceDeltaToolCall(index=0, id=None, function=ChoiceDeltaToolCallFunction(arguments='"}', name=None), type=None)]
Usage received: CompletionUsage(completion_tokens=14, prompt_tokens=61, total_tokens=75, completion_tokens_details=CompletionTokensDetails(accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=0, rejected_prediction_tokens=0), prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0))

Reconstructed Tool Calls:
{
  "0": {
    "id": "call_hjq9uBSN660V2MXXR3sGzNYe",
    "name": "get_weather",
    "arguments": "{\"location\":\"Paris\"}"
  }
}
```

---

## 3. Anthropic Messages API (`/v1/messages`)

### Streaming Helper vs. Raw
The Anthropic Python SDK **does** provide a streaming helper.

**Helper Method:**
```python
with client.messages.stream(
    model="claude-3-5-sonnet-20240620",
    messages=[...],
    tools=[...]
) as stream:
    for event in stream:
        # Handle events

    message = stream.get_final_message()
```
*   **Benefit:** Handles event accumulation and provides a final constructed message object.

### Event Types (Raw)
If iterating over raw events (`client.messages.create(stream=True)`), the key event types are:

*   `message_start`: Initial message metadata.
*   `content_block_start`: Start of a content block (text or tool_use).
*   `content_block_delta`: Incremental updates.
    *   **Text:** `{"type": "text_delta", "text": "..."}`
    *   **Tool Use:** `{"type": "input_json_delta", "partial_json": "..."}`
*   `content_block_stop`: End of a block.
*   `message_delta`: Top-level updates (e.g., stop_reason).
*   `message_stop`: End of stream.

### Tool Partial Assembly
*   **Helper:** The SDK helper automatically assembles tool inputs.
*   **Raw:** You receive `input_json_delta` events containing partial JSON strings. You must concatenate these strings and parse the JSON when the block stops.

**Example Raw Event:**
```json
{
  "type": "content_block_delta",
  "index": 0,
  "delta": {
    "type": "input_json_delta",
    "partial_json": "{\"location\": \"Lon"
  }
}
```

```python
# Manual script
import os

from anthropic import Anthropic

# Note: Ensure ANTHROPIC_API_KEY is set

# Log to file to avoid potential stdout buffering issues in some environments
log_file = open("anthropic_verification.log", "w")


def log(msg):
    log_file.write(str(msg) + "\n")
    log_file.flush()
    print(msg)  # Also print to stdout


log("--- Testing Anthropic Messages API Streaming (SDK) ---")

try:
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    log("Client initialized.")

    # Test 1: Using the stream helper
    log("\nTest 1: Using client.messages.stream() helper")
    try:
        with client.messages.stream(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "What is the weather in London?"}],
            tools=[
                {
                    "name": "get_weather",
                    "description": "Get the weather for a location",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                            "unit": {"type": "string", "enum": ["c", "f"]},
                        },
                        "required": ["location"],
                    },
                }
            ],
        ) as stream:
            log("Stream context entered.")
            for event in stream:
                log(f"Helper Event: {type(event)}")
                if hasattr(event, "type"):
                    log(f"  Type: {event.type}")
                if hasattr(event, "delta"):
                    log(f"  Delta: {event.delta}")
                if hasattr(event, "input_json"):
                    log(f"  Input JSON (partial): {event.input_json}")

            log("Stream loop finished.")
            message = stream.get_final_message()
            log(f"Final Message: {message.model_dump_json(indent=2)}")

    except Exception as e:
        log(f"Error in Test 1: {e}")

    # Test 2: Using raw streaming (stream=True)
    log("\nTest 2: Using raw client.messages.create(stream=True)")
    try:
        stream_raw = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": "What is the weather in London?"}],
            tools=[
                {
                    "name": "get_weather",
                    "description": "Get the weather for a location",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                            "unit": {"type": "string", "enum": ["c", "f"]},
                        },
                        "required": ["location"],
                    },
                }
            ],
            stream=True,
        )

        log("Raw stream created.")
        for event in stream_raw:
            log(f"Raw Event: {event.type}")
            if event.type == "content_block_delta":
                log(f"  Delta: {event.delta}")

    except Exception as e:
        log(f"Error in Test 2: {e}")

except Exception as e:
    log(f"Fatal error: {e}")

log_file.close()
```

**Output:**

```shell
python verify_anthropic.py
--- Testing Anthropic Messages API Streaming (SDK) ---
Client initialized.

Test 1: Using client.messages.stream() helper
Stream context entered.
Helper Event: <class 'anthropic.types.raw_message_start_event.RawMessageStartEvent'>
  Type: message_start
Helper Event: <class 'anthropic.types.raw_content_block_start_event.RawContentBlockStartEvent'>
  Type: content_block_start
Helper Event: <class 'anthropic.types.raw_content_block_delta_event.RawContentBlockDeltaEvent'>
  Type: content_block_delta
  Delta: TextDelta(text='I', type='text_delta')
Helper Event: <class 'anthropic.lib.streaming._types.TextEvent'>
  Type: text
Helper Event: <class 'anthropic.types.raw_content_block_delta_event.RawContentBlockDeltaEvent'>
  Type: content_block_delta
  Delta: TextDelta(text="'ll", type='text_delta')
Helper Event: <class 'anthropic.lib.streaming._types.TextEvent'>
  Type: text
Helper Event: <class 'anthropic.types.raw_content_block_delta_event.RawContentBlockDeltaEvent'>
  Type: content_block_delta
  Delta: TextDelta(text=' check the current', type='text_delta')
Helper Event: <class 'anthropic.lib.streaming._types.TextEvent'>
  Type: text
Helper Event: <class 'anthropic.types.raw_content_block_delta_event.RawContentBlockDeltaEvent'>
  Type: content_block_delta
  Delta: TextDelta(text=' weather in London for', type='text_delta')
Helper Event: <class 'anthropic.lib.streaming._types.TextEvent'>
  Type: text
Helper Event: <class 'anthropic.types.raw_content_block_delta_event.RawContentBlockDeltaEvent'>
  Type: content_block_delta
  Delta: TextDelta(text=' you.', type='text_delta')
Helper Event: <class 'anthropic.lib.streaming._types.TextEvent'>
  Type: text
Helper Event: <class 'anthropic.lib.streaming._types.ContentBlockStopEvent'>
  Type: content_block_stop
Helper Event: <class 'anthropic.types.raw_content_block_start_event.RawContentBlockStartEvent'>
  Type: content_block_start
Helper Event: <class 'anthropic.types.raw_content_block_delta_event.RawContentBlockDeltaEvent'>
  Type: content_block_delta
  Delta: InputJSONDelta(partial_json='', type='input_json_delta')
Helper Event: <class 'anthropic.lib.streaming._types.InputJsonEvent'>
  Type: input_json
Helper Event: <class 'anthropic.types.raw_content_block_delta_event.RawContentBlockDeltaEvent'>
  Type: content_block_delta
  Delta: InputJSONDelta(partial_json='{"locat', type='input_json_delta')
Helper Event: <class 'anthropic.lib.streaming._types.InputJsonEvent'>
  Type: input_json
Helper Event: <class 'anthropic.types.raw_content_block_delta_event.RawContentBlockDeltaEvent'>
  Type: content_block_delta
  Delta: InputJSONDelta(partial_json='ion": "Lo', type='input_json_delta')
Helper Event: <class 'anthropic.lib.streaming._types.InputJsonEvent'>
  Type: input_json
Helper Event: <class 'anthropic.types.raw_content_block_delta_event.RawContentBlockDeltaEvent'>
  Type: content_block_delta
  Delta: InputJSONDelta(partial_json='ndon"', type='input_json_delta')
Helper Event: <class 'anthropic.lib.streaming._types.InputJsonEvent'>
  Type: input_json
Helper Event: <class 'anthropic.types.raw_content_block_delta_event.RawContentBlockDeltaEvent'>
  Type: content_block_delta
  Delta: InputJSONDelta(partial_json=', "u', type='input_json_delta')
Helper Event: <class 'anthropic.lib.streaming._types.InputJsonEvent'>
  Type: input_json
Helper Event: <class 'anthropic.types.raw_content_block_delta_event.RawContentBlockDeltaEvent'>
  Type: content_block_delta
  Delta: InputJSONDelta(partial_json='nit": "', type='input_json_delta')
Helper Event: <class 'anthropic.lib.streaming._types.InputJsonEvent'>
  Type: input_json
Helper Event: <class 'anthropic.types.raw_content_block_delta_event.RawContentBlockDeltaEvent'>
  Type: content_block_delta
  Delta: InputJSONDelta(partial_json='c"}', type='input_json_delta')
Helper Event: <class 'anthropic.lib.streaming._types.InputJsonEvent'>
  Type: input_json
Helper Event: <class 'anthropic.lib.streaming._types.ContentBlockStopEvent'>
  Type: content_block_stop
Helper Event: <class 'anthropic.types.raw_message_delta_event.RawMessageDeltaEvent'>
  Type: message_delta
  Delta: Delta(stop_reason='tool_use', stop_sequence=None)
Helper Event: <class 'anthropic.lib.streaming._types.MessageStopEvent'>
  Type: message_stop
Stream loop finished.
Final Message: {
  "id": "msg_01AphVR11Hm3n4fPsAFwTU3T",
  "content": [
    {
      "citations": null,
      "text": "I'll check the current weather in London for you.",
      "type": "text"
    },
    {
      "id": "toolu_017EwM8KEVwg48Q64drUSKfD",
      "input": {
        "location": "London",
        "unit": "c"
      },
      "name": "get_weather",
      "type": "tool_use"
    }
  ],
  "model": "claude-3-5-haiku-20241022",
  "role": "assistant",
  "stop_reason": "tool_use",
  "stop_sequence": null,
  "type": "message",
  "usage": {
    "cache_creation": {
      "ephemeral_1h_input_tokens": 0,
      "ephemeral_5m_input_tokens": 0
    },
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
    "input_tokens": 351,
    "output_tokens": 82,
    "server_tool_use": null,
    "service_tier": "standard"
  }
}

Test 2: Using raw client.messages.create(stream=True)
Raw stream created.
Raw Event: message_start
Raw Event: content_block_start
Raw Event: content_block_delta
  Delta: TextDelta(text='I', type='text_delta')
Raw Event: content_block_delta
  Delta: TextDelta(text="'ll", type='text_delta')
Raw Event: content_block_delta
  Delta: TextDelta(text=' help', type='text_delta')
Raw Event: content_block_delta
  Delta: TextDelta(text=' you get', type='text_delta')
Raw Event: content_block_delta
  Delta: TextDelta(text=' the weather information', type='text_delta')
Raw Event: content_block_delta
  Delta: TextDelta(text=' for London. I', type='text_delta')
Raw Event: content_block_delta
  Delta: TextDelta(text="'ll retrieve", type='text_delta')
Raw Event: content_block_delta
  Delta: TextDelta(text=' the weather details', type='text_delta')
Raw Event: content_block_delta
  Delta: TextDelta(text=' using', type='text_delta')
Raw Event: content_block_delta
  Delta: TextDelta(text=' the default', type='text_delta')
Raw Event: content_block_delta
  Delta: TextDelta(text=' temperature', type='text_delta')
Raw Event: content_block_delta
  Delta: TextDelta(text=' unit.', type='text_delta')
Raw Event: content_block_stop
Raw Event: content_block_start
Raw Event: content_block_delta
  Delta: InputJSONDelta(partial_json='', type='input_json_delta')
Raw Event: content_block_delta
  Delta: InputJSONDelta(partial_json='{"', type='input_json_delta')
Raw Event: content_block_delta
  Delta: InputJSONDelta(partial_json='locat', type='input_json_delta')
Raw Event: content_block_delta
  Delta: InputJSONDelta(partial_json='ion": "Lo', type='input_json_delta')
Raw Event: content_block_delta
  Delta: InputJSONDelta(partial_json='ndon"}', type='input_json_delta')
Raw Event: content_block_stop
Raw Event: message_delta
Raw Event: message_stop
```
