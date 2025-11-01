# Allos Agent SDK - Initial Architecture Design

## Core Architectural Principles

1. **Provider Agnostic**: Unified interface for all LLM providers
2. **Tool-First Design**: Tools are first-class citizens
3. **Stateless Agent**: All state lives in `Context` objects
4. **Declarative Tools**: Tools self-describe their capabilities
5. **Fail-Safe Defaults**: Safe by default, powerful when needed

---

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         CLI Layer                           │
│              (allos/cli - User Interface)                   │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                      Agent Core                             │
│     (allos/agent - Orchestration & Agentic Loop)            │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   Planner    │  │   Executor   │  │   Reflector  │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└─────────┬────────────────────┬───────────────────┬─────────-┘
          │                    │                   │
┌─────────▼─────────┐ ┌───────-▼────────┐ ┌───────▼─────────┐
│  Provider Layer   │ │   Tool Layer    │ │  Context Layer  │
│                   │ │                 │ │                 │
│ ┌───────────────┐ │ │ ┌────────────┐  │ │ ┌─────────────┐ │
│ │   OpenAI      │ │ │ │ FileSystem │  │ │ │  History    │ │
│ │   Anthropic   │ │ │ │   Shell    │  │ │ │  Compactor  │ │
│ │   Ollama      │ │ │ │   Web      │  │ │ │  Cache      │ │
│ │   Google      │ │ │ │   Custom   │  │ │ └─────────────┘ │
│ └───────────────┘ │ │ └────────────┘  │ └─────────────────┘
└───────────────────┘ └────────────────-┘
```

---

## Core Abstractions

### 1. Provider Layer

**Purpose**: Abstract LLM providers behind a unified interface

```python
# allos/providers/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Iterator
from dataclasses import dataclass
from enum import Enum

class MessageRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

@dataclass
class Message:
    """Unified message format across all providers"""
    role: MessageRole
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None  # For tool responses

@dataclass
class ToolCall:
    """Represents a tool invocation request"""
    id: str
    name: str
    arguments: Dict[str, Any]

@dataclass
class ProviderResponse:
    """Unified response format"""
    content: Optional[str]
    tool_calls: Optional[List[ToolCall]]
    finish_reason: str
    usage: Dict[str, int]  # tokens used
    raw_response: Any  # Original provider response

class BaseProvider(ABC):
    """Abstract base class for all LLM providers"""
    
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.extra_params = kwargs
    
    @abstractmethod
    def chat(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ProviderResponse:
        """
        Send a chat completion request
        
        Args:
            messages: Conversation history
            tools: Available tools in OpenAI function format
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            
        Returns:
            ProviderResponse with content or tool calls
        """
        pass
    
    @abstractmethod
    def supports_tool_calling(self) -> bool:
        """Whether this provider supports native tool calling"""
        pass
    
    @abstractmethod
    def get_token_count(self, text: str) -> int:
        """Count tokens in text (provider-specific)"""
        pass
    
    @property
    @abstractmethod
    def context_window(self) -> int:
        """Maximum context window size"""
        pass
    
    @property
    def name(self) -> str:
        """Provider name"""
        return self.__class__.__name__.replace("Provider", "").lower()
```

**OpenAI Implementation Example:**

```python
# allos/providers/openai.py
from openai import OpenAI
from .base import BaseProvider, Message, ProviderResponse, ToolCall, MessageRole
import tiktoken

class OpenAIProvider(BaseProvider):
    """OpenAI provider implementation"""
    
    def __init__(self, model: str = "gpt-4", **kwargs):
        super().__init__(model, **kwargs)
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self._tokenizer = tiktoken.encoding_for_model(model)
    
    def chat(self, messages, tools=None, temperature=0.7, max_tokens=None, **kwargs):
        # Convert our Message format to OpenAI format
        openai_messages = [self._convert_message(msg) for msg in messages]
        
        params = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": temperature,
        }
        
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"
        
        if max_tokens:
            params["max_tokens"] = max_tokens
        
        response = self.client.chat.completions.create(**params)
        
        # Convert OpenAI response to our format
        return self._convert_response(response)
    
    def _convert_message(self, msg: Message) -> dict:
        """Convert our Message to OpenAI format"""
        result = {"role": msg.role.value, "content": msg.content}
        
        if msg.tool_calls:
            result["tool_calls"] = msg.tool_calls
        if msg.tool_call_id:
            result["tool_call_id"] = msg.tool_call_id
        if msg.name:
            result["name"] = msg.name
            
        return result
    
    def _convert_response(self, response) -> ProviderResponse:
        """Convert OpenAI response to our format"""
        choice = response.choices[0]
        message = choice.message
        
        tool_calls = None
        if hasattr(message, 'tool_calls') and message.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments)
                )
                for tc in message.tool_calls
            ]
        
        return ProviderResponse(
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            raw_response=response
        )
    
    def supports_tool_calling(self) -> bool:
        return True
    
    def get_token_count(self, text: str) -> int:
        return len(self._tokenizer.encode(text))
    
    @property
    def context_window(self) -> int:
        # Model-specific context windows
        windows = {
            "gpt-4": 8192,
            "gpt-4-turbo": 128000,
            "gpt-4o": 128000,
            "gpt-3.5-turbo": 16385,
        }
        return windows.get(self.model, 8192)
```

**Provider Registry:**

```python
# allos/providers/registry.py
from typing import Dict, Type
from .base import BaseProvider

class ProviderRegistry:
    """Registry for managing available providers"""
    
    _providers: Dict[str, Type[BaseProvider]] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register a provider"""
        def wrapper(provider_class: Type[BaseProvider]):
            cls._providers[name.lower()] = provider_class
            return provider_class
        return wrapper
    
    @classmethod
    def get_provider(cls, name: str, **kwargs) -> BaseProvider:
        """Get a provider instance by name"""
        name = name.lower()
        if name not in cls._providers:
            raise ValueError(
                f"Unknown provider: {name}. "
                f"Available: {list(cls._providers.keys())}"
            )
        return cls._providers[name](**kwargs)
    
    @classmethod
    def list_providers(cls) -> list[str]:
        """List all registered providers"""
        return list(cls._providers.keys())

# Usage in provider files:
# @ProviderRegistry.register("openai")
# class OpenAIProvider(BaseProvider):
#     ...
```

---

### 2. Tool Layer

**Purpose**: Declarative, self-describing tools that agents can use

```python
# allos/tools/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import json

class ToolPermission(Enum):
    """Permission levels for tools"""
    ALWAYS_ALLOW = "always_allow"      # No confirmation needed
    ASK = "ask"                         # Ask user first
    NEVER = "never"                     # Blocked

@dataclass
class ToolParameter:
    """Describes a tool parameter"""
    name: str
    type: str  # "string", "integer", "boolean", "object", "array"
    description: str
    required: bool = True
    enum: Optional[List[Any]] = None
    default: Optional[Any] = None

class BaseTool(ABC):
    """Abstract base class for all tools"""
    
    # Subclasses should set these
    name: str
    description: str
    parameters: List[ToolParameter]
    
    # Default permission (can be overridden per instance)
    default_permission: ToolPermission = ToolPermission.ASK
    
    def __init__(self, permission: Optional[ToolPermission] = None):
        self.permission = permission or self.default_permission
        self._validate_class_attributes()
    
    def _validate_class_attributes(self):
        """Ensure subclass defined required attributes"""
        required = ["name", "description", "parameters"]
        for attr in required:
            if not hasattr(self, attr):
                raise AttributeError(
                    f"{self.__class__.__name__} must define '{attr}'"
                )
    
    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the tool with given parameters
        
        Returns:
            Dict with 'success' (bool) and either 'result' or 'error'
        """
        pass
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert tool to OpenAI function calling format"""
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                properties[param.name]["enum"] = param.enum
            if param.required:
                required.append(param.name)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                }
            }
        }
    
    def validate_arguments(self, arguments: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate tool arguments"""
        for param in self.parameters:
            if param.required and param.name not in arguments:
                return False, f"Missing required parameter: {param.name}"
        return True, None
    
    def __str__(self) -> str:
        return f"{self.name}: {self.description}"
```

**Example Tool Implementation:**

```python
# allos/tools/filesystem/read.py
from ..base import BaseTool, ToolParameter, ToolPermission
from pathlib import Path

class FileReadTool(BaseTool):
    """Tool for reading file contents"""
    
    name = "read_file"
    description = "Read the contents of a file"
    parameters = [
        ToolParameter(
            name="path",
            type="string",
            description="Path to the file to read",
            required=True
        ),
        ToolParameter(
            name="start_line",
            type="integer",
            description="Starting line number (1-indexed, optional)",
            required=False
        ),
        ToolParameter(
            name="end_line",
            type="integer",
            description="Ending line number (inclusive, optional)",
            required=False
        ),
    ]
    default_permission = ToolPermission.ALWAYS_ALLOW  # Reading is safe
    
    def execute(self, path: str, start_line: Optional[int] = None, 
                end_line: Optional[int] = None) -> Dict[str, Any]:
        try:
            file_path = Path(path).resolve()
            
            # Security: Prevent reading outside allowed directories
            # (This would be configurable in production)
            if not self._is_path_allowed(file_path):
                return {
                    "success": False,
                    "error": f"Access denied: {path}"
                }
            
            with open(file_path, 'r', encoding='utf-8') as f:
                if start_line or end_line:
                    lines = f.readlines()
                    start = (start_line or 1) - 1
                    end = end_line if end_line else len(lines)
                    content = ''.join(lines[start:end])
                else:
                    content = f.read()
            
            return {
                "success": True,
                "result": {
                    "path": str(file_path),
                    "content": content,
                    "lines": len(content.splitlines())
                }
            }
            
        except FileNotFoundError:
            return {"success": False, "error": f"File not found: {path}"}
        except PermissionError:
            return {"success": False, "error": f"Permission denied: {path}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _is_path_allowed(self, path: Path) -> bool:
        """Check if path is within allowed directories"""
        # MVP: Allow current directory and subdirectories
        # TODO: Make this configurable
        try:
            cwd = Path.cwd()
            path.relative_to(cwd)
            return True
        except ValueError:
            return False
```

**Tool Registry:**

```python
# allos/tools/registry.py
from typing import Dict, Type, List
from .base import BaseTool, ToolPermission

class ToolRegistry:
    """Registry for managing available tools"""
    
    _tools: Dict[str, Type[BaseTool]] = {}
    
    @classmethod
    def register(cls, tool_class: Type[BaseTool]) -> Type[BaseTool]:
        """Register a tool class"""
        tool_name = tool_class.name
        if tool_name in cls._tools:
            raise ValueError(f"Tool already registered: {tool_name}")
        cls._tools[tool_name] = tool_class
        return tool_class
    
    @classmethod
    def get_tool(cls, name: str, **kwargs) -> BaseTool:
        """Get a tool instance by name"""
        if name not in cls._tools:
            raise ValueError(f"Unknown tool: {name}")
        return cls._tools[name](**kwargs)
    
    @classmethod
    def list_tools(cls) -> List[str]:
        """List all registered tool names"""
        return list(cls._tools.keys())
    
    @classmethod
    def get_all_tools(cls, permissions: Dict[str, ToolPermission] = None) -> List[BaseTool]:
        """Get instances of all registered tools"""
        permissions = permissions or {}
        tools = []
        for name, tool_class in cls._tools.items():
            permission = permissions.get(name)
            tools.append(tool_class(permission=permission))
        return tools

# Auto-register decorator
def tool(cls: Type[BaseTool]) -> Type[BaseTool]:
    """Decorator to auto-register tools"""
    return ToolRegistry.register(cls)

# Usage:
# @tool
# class FileReadTool(BaseTool):
#     ...
```

---

### 3. Context Layer

**Purpose**: Manage conversation history, context windows, and caching

```python
# allos/context/manager.py
from typing import List, Optional
from dataclasses import dataclass, field
from ..providers.base import Message, MessageRole, ToolCall
import json

@dataclass
class ConversationContext:
    """Manages conversation state and history"""
    
    messages: List[Message] = field(default_factory=list)
    max_tokens: Optional[int] = None  # If None, use provider's window
    system_prompt: Optional[str] = None
    
    # Metadata
    total_tokens_used: int = 0
    total_cost: float = 0.0
    turn_count: int = 0
    
    def add_system_message(self, content: str):
        """Add or update system message"""
        self.system_prompt = content
        # System message is always first if present
        if self.messages and self.messages[0].role == MessageRole.SYSTEM:
            self.messages[0] = Message(role=MessageRole.SYSTEM, content=content)
        else:
            self.messages.insert(0, Message(role=MessageRole.SYSTEM, content=content))
    
    def add_user_message(self, content: str):
        """Add a user message"""
        self.messages.append(Message(role=MessageRole.USER, content=content))
        self.turn_count += 1
    
    def add_assistant_message(self, content: Optional[str], 
                            tool_calls: Optional[List[ToolCall]] = None):
        """Add an assistant message"""
        # Convert ToolCall objects to dict format for Message
        tool_calls_dict = None
        if tool_calls:
            tool_calls_dict = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments)
                    }
                }
                for tc in tool_calls
            ]
        
        self.messages.append(Message(
            role=MessageRole.ASSISTANT,
            content=content or "",
            tool_calls=tool_calls_dict
        ))
    
    def add_tool_result(self, tool_call_id: str, tool_name: str, result: str):
        """Add a tool execution result"""
        self.messages.append(Message(
            role=MessageRole.TOOL,
            content=result,
            tool_call_id=tool_call_id,
            name=tool_name
        ))
    
    def get_token_count(self, provider) -> int:
        """Estimate total tokens in context"""
        total = 0
        for msg in self.messages:
            total += provider.get_token_count(msg.content or "")
            # Add overhead for role, tool calls, etc.
            total += 10  # Rough estimate
        return total
    
    def needs_compaction(self, provider, buffer: int = 1000) -> bool:
        """Check if context needs compaction"""
        if not self.max_tokens:
            self.max_tokens = provider.context_window
        
        current_tokens = self.get_token_count(provider)
        return current_tokens > (self.max_tokens - buffer)
    
    def to_dict(self) -> dict:
        """Serialize context for persistence"""
        return {
            "messages": [
                {
                    "role": msg.role.value,
                    "content": msg.content,
                    "tool_calls": msg.tool_calls,
                    "tool_call_id": msg.tool_call_id,
                    "name": msg.name,
                }
                for msg in self.messages
            ],
            "system_prompt": self.system_prompt,
            "metadata": {
                "total_tokens_used": self.total_tokens_used,
                "total_cost": self.total_cost,
                "turn_count": self.turn_count,
            }
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ConversationContext":
        """Deserialize context"""
        messages = [
            Message(
                role=MessageRole(msg["role"]),
                content=msg["content"],
                tool_calls=msg.get("tool_calls"),
                tool_call_id=msg.get("tool_call_id"),
                name=msg.get("name"),
            )
            for msg in data["messages"]
        ]
        
        ctx = cls(messages=messages, system_prompt=data.get("system_prompt"))
        if "metadata" in data:
            ctx.total_tokens_used = data["metadata"].get("total_tokens_used", 0)
            ctx.total_cost = data["metadata"].get("total_cost", 0.0)
            ctx.turn_count = data["metadata"].get("turn_count", 0)
        
        return ctx
```

**Context Compaction (Phase 2):**

```python
# allos/context/compactor.py
from .manager import ConversationContext
from ..providers.base import BaseProvider, Message, MessageRole

class ContextCompactor:
    """Strategies for reducing context size"""
    
    @staticmethod
    def truncate_oldest(context: ConversationContext, 
                       target_tokens: int,
                       provider: BaseProvider) -> ConversationContext:
        """Simple truncation: keep system + most recent messages"""
        # Always keep system message
        system_msg = None
        other_messages = []
        
        for msg in context.messages:
            if msg.role == MessageRole.SYSTEM:
                system_msg = msg
            else:
                other_messages.append(msg)
        
        # Keep messages from the end until we hit target
        kept_messages = []
        current_tokens = 0
        
        for msg in reversed(other_messages):
            msg_tokens = provider.get_token_count(msg.content or "")
            if current_tokens + msg_tokens > target_tokens:
                break
            kept_messages.insert(0, msg)
            current_tokens += msg_tokens
        
        # Reconstruct context
        new_messages = []
        if system_msg:
            new_messages.append(system_msg)
        new_messages.extend(kept_messages)
        
        context.messages = new_messages
        return context
    
    # Phase 2: Add smarter strategies
    # - Summarization of older messages
    # - Importance-based retention
    # - Tool result compression
```

---

### 4. Agent Core

**Purpose**: The main agentic loop - plan, execute, reflect

```python
# allos/agent/agent.py
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass
import json
from rich.console import Console

from ..providers.base import BaseProvider, ToolCall
from ..providers.registry import ProviderRegistry
from ..tools.base import BaseTool, ToolPermission
from ..tools.registry import ToolRegistry
from ..context.manager import ConversationContext
from ..utils.errors import AllosError, ToolExecutionError

console = Console()

@dataclass
class AgentConfig:
    """Configuration for an agent"""
    provider: str
    model: str
    tools: List[str]  # Tool names to enable
    system_prompt: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    max_iterations: int = 15  # Prevent infinite loops
    
    # Tool permissions
    tool_permissions: Dict[str, ToolPermission] = None
    auto_approve_safe_tools: bool = True
    
    # Callbacks
    on_tool_call: Optional[Callable] = None
    on_iteration: Optional[Callable] = None

class Agent:
    """Main agent orchestrator"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        
        # Initialize provider
        self.provider: BaseProvider = ProviderRegistry.get_provider(
            config.provider,
            model=config.model
        )
        
        # Initialize tools
        self.tools: Dict[str, BaseTool] = {}
        for tool_name in config.tools:
            permission = None
            if config.tool_permissions:
                permission = config.tool_permissions.get(tool_name)
            tool = ToolRegistry.get_tool(tool_name, permission=permission)
            self.tools[tool_name] = tool
        
        # Initialize context
        self.context = ConversationContext()
        if config.system_prompt:
            self.context.add_system_message(config.system_prompt)
    
    def run(self, task: str) -> str:
        """
        Main entry point: Run the agent on a task
        
        Args:
            task: The task description from the user
            
        Returns:
            Final response string
        """
        console.print(f"[bold blue]Task:[/bold blue] {task}")
        
        # Add user's task to context
        self.context.add_user_message(task)
        
        # Main agentic loop
        for iteration in range(self.config.max_iterations):
            if self.config.on_iteration:
                self.config.on_iteration(iteration, self.context)
            
            console.print(f"\n[dim]Iteration {iteration + 1}/{self.config.max_iterations}[/dim]")
            
            # Get response from LLM
            response = self._get_llm_response()
            
            # If no tool calls, we're done
            if not response.tool_calls:
                final_response = response.content or ""
                console.print(f"\n[bold green]✓ Complete[/bold green]")
                return final_response
            
            # Execute tool calls
            self._execute_tool_calls(response.tool_calls)
            
            # Add assistant's response to context
            self.context.add_assistant_message(
                content=response.content,
                tool_calls=response.tool_calls
            )
        
        # Hit max iterations
        console.print(f"\n[bold yellow]⚠ Max iterations reached[/bold yellow]")
        return self.context.messages[-1].content or "Task incomplete: max iterations reached"
    
    def _get_llm_response(self):
        """Get response from LLM"""
        # Convert tools to OpenAI format
        tool_schemas = [tool.to_openai_format() for tool in self.tools.values()]
        
        # Call provider
        try:
            response = self.provider.chat(
                messages=self.context.messages,
                tools=tool_schemas if tool_schemas else None,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            
            # Update token usage
            self.context.total_tokens_used += response.usage["total_tokens"]
            
            return response
            
        except Exception as e:
            raise AllosError(f"Provider error: {str(e)}") from e
    
    def _execute_tool_calls(self, tool_calls: List[ToolCall]):
        """Execute requested tool calls"""
        for tool_call in tool_calls:
            console.print(f"[cyan]  → {tool_call.name}[/cyan]")
            
            # Get the tool
            if tool_call.name not in self.tools:
                error_msg = f"Tool not available: {tool_call.name}"
                console.print(f"[red]    ✗ {error_msg}[/red]")
                self.context.add_tool_result(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    result=json.dumps({"success": False, "error": error_msg})
                )
                continue
            
            tool = self.tools[tool_call.name]
            
            # Check permission
            if not self._check_tool_permission(tool, tool_call):
                error_msg = "Tool execution denied by user"
                console.print(f"[red]    ✗ {error_msg}[/red]")
                self.context.add_tool_result(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    result=json.dumps({"success": False, "error": error_msg})
                )
                continue
            
            # Validate arguments
            valid, error = tool.validate_arguments(tool_call.arguments)
            if not valid:
                console.print(f"[red]    ✗ {error}[/red]")
                self.context.add_tool_result(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    result=json.dumps({"success": False, "error": error})
                )
                continue
            
            # Execute tool
            try:
                result = tool.execute(**tool_call.arguments)
                
                if result["success"]:
                    console.print(f"[green]    ✓ Success[/green]")
                else:
                    console.print(f"[yellow]    ! {result.get('error', 'Failed')}[/yellow]")
                
                # Callback
                if self.config.on_tool_call:
                    self.config.on_tool_call(tool_call.name, tool_call.arguments, result)
                
                # Add result to context
                self.context.add_tool_result(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    result=json.dumps(result)
                )
                
            except Exception as e:
                error_msg = f"Tool execution failed: {str(e)}"
                console.print(f"[red]    ✗ {error_msg}[/red]")
                self.context.add_tool_result(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    result=json.dumps({"success": False, "error": error_msg})
                )
    
    def _check_tool_permission(self, tool: BaseTool, tool_call: ToolCall) -> bool:
        """Check if tool execution is permitted"""
        if tool.permission == ToolPermission.ALWAYS_ALLOW:
            return True
        
        if tool.permission == ToolPermission.NEVER:
            return False
        
        if tool.permission == ToolPermission.ASK:
            # Show user what the tool will do
            console.print(f"\n[yellow]  Tool wants to: {tool.description}[/yellow]")
            console.print(f"[yellow]  Arguments: {json.dumps(tool_call.arguments, indent=2)}[/yellow]")
            
            response = console.input("[yellow]  Allow? (y/n): [/yellow]").strip().lower()
            return response == 'y'
        
        return False
    
    def save_session(self, path: str):
        """Save conversation to file"""
        import json
        with open(path, 'w') as f:
            json.dump(self.context.to_dict(), f, indent=2)
        console.print(f"[green]Session saved to {path}[/green]")
    
    @classmethod
    def load_session(cls, path: str, config: AgentConfig) -> "Agent":
        """Load conversation from file"""
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        
        agent = cls(config)
        agent.context = ConversationContext.from_dict(data)
        console.print(f"[green]Session loaded from {path}[/green]")
        return agent
```

---

### 5. CLI Layer

**Purpose**: User-friendly command-line interface

```python
# allos/cli/main.py
import click
from rich.console import Console
from rich.panel import Panel
from pathlib import Path

from ..agent.agent import Agent, AgentConfig
from ..tools.base import ToolPermission
from ..providers.registry import ProviderRegistry
from ..tools.registry import ToolRegistry

console = Console()

@click.group()
@click.version_option()
def cli():
    """Allos Agent SDK - LLM-agnostic agentic coding"""
    pass

@cli.command()
@click.argument('task', required=False)
@click.option('--provider', '-p', default='openai', help='LLM provider')
@click.option('--model', '-m', help='Model name')
@click.option('--tools', '-t', multiple=True, help='Tools to enable')
@click.option('--auto-approve', is_flag=True, default=False, 
              help='Auto-approve safe tools')
@click.option('--session', '-s', help='Session file to save/resume')
@click.option('--interactive', '-i', is_flag=True, help='Interactive mode')
def run(task, provider, model, tools, auto_approve, session, interactive):
    """Run the agent on a task"""
    
    # Default model per provider
    if not model:
        model_defaults = {
            'openai': 'gpt-4',
            'anthropic': 'claude-sonnet-4-5-20250929',
            'ollama': 'qwen2.5-coder',
        }
        model = model_defaults.get(provider, 'gpt-4')
    
    # Default tools
    if not tools:
        tools = ['read_file', 'write_file', 'edit_file', 'shell_exec']
    
    # Build config
    config = AgentConfig(
        provider=provider,
        model=model,
        tools=list(tools),
        auto_approve_safe_tools=auto_approve,
    )
    
    # Load or create agent
    if session and Path(session).exists():
        agent = Agent.load_session(session, config)
    else:
        agent = Agent(config)
    
    # Show welcome
    console.print(Panel.fit(
        f"[bold]Allos Agent[/bold]\n"
        f"Provider: {provider}\n"
        f"Model: {model}\n"
        f"Tools: {', '.join(tools)}",
        border_style="blue"
    ))
    
    # Interactive mode
    if interactive or not task:
        while True:
            try:
                task = console.input("\n[bold cyan]You:[/bold cyan] ")
                if task.lower() in ['exit', 'quit', 'q']:
                    break
                
                response = agent.run(task)
                console.print(f"\n[bold green]Agent:[/bold green] {response}")
                
                if session:
                    agent.save_session(session)
                    
            except KeyboardInterrupt:
                console.print("\n[yellow]Goodbye![/yellow]")
                break
    else:
        # Single task mode
        response = agent.run(task)
        console.print(f"\n[bold green]Result:[/bold green] {response}")
        
        if session:
            agent.save_session(session)

@cli.command()
def list_providers():
    """List available providers"""
    providers = ProviderRegistry.list_providers()
    console.print("\n[bold]Available Providers:[/bold]")
    for p in providers:
        console.print(f"  • {p}")

@cli.command()
def list_tools():
    """List available tools"""
    tools = ToolRegistry.list_tools()
    console.print("\n[bold]Available Tools:[/bold]")
    for t in tools:
        tool = ToolRegistry.get_tool(t)
        console.print(f"  • [cyan]{tool.name}[/cyan]: {tool.description}")

if __name__ == '__main__':
    cli()
```

---

## Usage Examples

```python
# 1. Basic Usage (Python API)
from allos import Agent, AgentConfig

agent = Agent(AgentConfig(
    provider="openai",
    model="gpt-4",
    tools=["read_file", "write_file", "shell_exec"]
))

result = agent.run("Fix the bug in main.py")
print(result)

# 2. CLI Usage
# allos "Create a FastAPI hello world app"
# allos --provider anthropic --model claude-sonnet-4.5 "Same task"
# allos --provider ollama --model qwen2.5-coder "Same task"

# 3. Custom Tool
from allos.tools import BaseTool, tool, ToolParameter

@tool
class DatabaseQueryTool(BaseTool):
    name = "query_database"
    description = "Execute SQL query"
    parameters = [
        ToolParameter("query", "string", "SQL query", required=True)
    ]
    
    def execute(self, query: str):
        # Your implementation
        return {"success": True, "result": "..."}

# 4. Session Management
agent.run("Start building a web scraper")
agent.save_session("scraper_session.json")

# Later...
agent = Agent.load_session("scraper_session.json", config)
agent.run("Continue where we left off")
```

---

## Key Architectural Benefits

✅ **Provider Independence**: Easy to swap LLMs  
✅ **Tool Extensibility**: Adding tools is trivial  
✅ **Type Safety**: Pydantic for configs, strong typing throughout  
✅ **Testable**: Each component can be unit tested  
✅ **Observable**: Callbacks for monitoring  
✅ **Fail-Safe**: Permissions, validation, error handling  