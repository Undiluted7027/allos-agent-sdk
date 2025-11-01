# Using Built-in Tools

The Allos SDK comes with a powerful set of built-in tools that enable an agent to interact with its local environment, such as reading and writing files or executing shell commands.

## The `ToolRegistry`

Similar to providers, all tools are managed by a central `ToolRegistry`. You can use it to inspect, list, and get instances of any available tool.

```python
from allos.tools import ToolRegistry

# List the names of all available tools
available_tools = ToolRegistry.list_tools()
print(available_tools)
# Expected Output: ['edit_file', 'list_directory', 'read_file', 'shell_exec', 'write_file']

# Get an instance of a specific tool
read_file_tool = ToolRegistry.get_tool("read_file")
```

## Tool Permissions

Security is a primary concern when allowing an AI agent to interact with a system. Every tool in Allos has a `permission` attribute that defines its default security level.

-   `ALWAYS_ALLOW`: Safe, read-only operations.
-   `ASK_USER`: Potentially destructive or sensitive operations that require user confirmation.
-   `ALWAYS_DENY`: Reserved for tools that should be disabled.

The Agent Core (coming in Phase 4) will use this system to automatically prompt the user for confirmation before executing sensitive tools.

## Filesystem Tools

These tools allow the agent to interact with files and directories within its designated workspace. All paths are validated to prevent directory traversal attacks.

### `read_file`
- **Description:** Reads the content of a file.
- **Permission:** `ALWAYS_ALLOW`
- **Parameters:**
    - `path` (string, required): The relative path to the file.
    - `start_line` (integer, optional): The 1-based starting line to read.
    - `end_line` (integer, optional): The 1-based ending line to read.
    - `inclusive` (boolean, optional): If the `end_line` is to be included in output or not.

### `write_file`
- **Description:** Writes content to a file, overwriting it if it exists or creating it if it doesn't.
- **Permission:** `ASK_USER`
- **Parameters:**
    - `path` (string, required): The relative path to the file.
    - `content` (string, required): The content to write.
    - `append_mode` (boolean, optional): Whether to overwrite file or append.

### `edit_file`
- **Description:** Performs a unique find-and-replace in a file. The operation fails if the `find_string` is not found or is found more than once.
- **Permission:** `ASK_USER`
- **Parameters:**
    - `path` (string, required): The relative path to the file.
    - `find_string` (string, required): The exact string to find (must be unique).
    - `replace_with` (string, required): The string to replace it with.

### `list_directory`
- **Description:** Lists the contents of a directory.
- **Permission:** `ALWAYS_ALLOW`
- **Parameters:**
    - `path` (string, optional): The relative path to the directory (defaults to current).
    - `recursive` (boolean, optional): Whether to list contents recursively. Defaults to `false`.
    - `show_hidden` (boolean, optional): Whether to include hidden files/directories. Defaults to `false`.

## Execution Tools

### `shell_exec`
- **Description:** Executes a shell command in a non-interactive, sandboxed environment.
- **Permission:** `ASK_USER`
- **Parameters:**
    - `command` (string, required): The command to execute.
    - `timeout` (integer, optional): Timeout in seconds (default: 60).

> [!WARNING] Security Note:
The `shell_exec` tool includes a blocklist for common dangerous commands (`sudo`, `rm`, `mv`, etc.) and uses safe parsing to prevent shell injection. However, the primary security layer is the `ASK_USER` permission, which will require explicit user approval for every command execution.
