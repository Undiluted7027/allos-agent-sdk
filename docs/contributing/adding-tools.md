# Guide: Adding a Built-in Tool

This guide is for contributors who want to add a new, general-purpose tool to the Allos SDK itself.

## 1. Create the Tool File

If the tool belongs to an existing category (like `filesystem` or `execution`), add a new file there. For a new category, create a new subdirectory.

Example: `allos/tools/filesystem/new_tool.py`

## 2. Implement the Tool Class

Follow the pattern for creating a custom tool:
- Inherit from `BaseTool`.
- Use the `@tool` decorator.
- Define `name`, `description`, `parameters`, and `permission`.
- Implement the `execute(**kwargs)` method.

```python
# allos/tools/filesystem/new_tool.py
from typing import Any, Dict
from allos.tools import BaseTool, tool, ToolParameter, ToolPermission

@tool
class NewTool(BaseTool):
    name: str = "new_tool"
    description: str = "This is a new tool."
    parameters: list[ToolParameter] = [
        ToolParameter(name="param1", type="string", required=True)
    ]
    permission: ToolPermission = ToolPermission.ASK_USER

    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        param1 = kwargs.get("param1")
        # ... tool logic ...
        return {"status": "success", "result": f"Processed {param1}"}
```

## 3. Register the Tool for Discovery

Open the `__init__.py` file in the tool's category directory (e.g., `allos/tools/filesystem/__init__.py`) and add an import for your new tool class. This is crucial for the tool to be automatically registered.

```python
# allos/tools/filesystem/__init__.py

# ... other tool imports
from .new_tool import NewTool

__all__ = [..., "NewTool"]
```

## 4. Write Comprehensive Tests

Add a new test class for your tool in the appropriate test file (e.g., `tests/unit/test_filesystem_tools.py`).

Your tests should cover:
- The successful execution path.
- All error conditions (e.g., missing arguments, invalid inputs).
- Any security checks (e.g., path validation).

## 5. Submit a Pull Request

Once your implementation is complete and all tests are passing, submit a pull request for review. Thank you for contributing!
