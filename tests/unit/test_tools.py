# tests/unit/test_tools.py

from typing import Any, Dict

import pytest

from allos.tools.base import BaseTool, ToolParameter, ToolPermission
from allos.tools.registry import ToolRegistry, _tool_registry, tool
from allos.utils.errors import ToolError, ToolNotFoundError

# --- Test Setup ---


# A dummy tool for testing registration and base functionality
@tool
class DummySearchTool(BaseTool):
    name = "web_search"
    description = "Performs a web search."
    parameters = [
        ToolParameter(
            name="query", type="string", description="The search query.", required=True
        )
    ]
    permission = ToolPermission.ALWAYS_ALLOW

    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        return {"status": "success", "results": f"Results for {kwargs.get('query')}"}


class TestToolBase:
    """Tests for the base tool data structures and methods."""

    def test_tool_parameter_dataclass(self):
        param = ToolParameter(name="test", type="string", description="A test param.")
        assert param.name == "test"
        assert not param.required

    def test_validate_arguments_success(self):
        tool_instance = DummySearchTool()
        # Should not raise an error
        tool_instance.validate_arguments({"query": "hello world"})

    def test_validate_arguments_missing_required(self):
        tool_instance = DummySearchTool()
        with pytest.raises(ToolError) as excinfo:
            tool_instance.validate_arguments({"other_arg": "value"})
        assert "Missing required arguments" in str(excinfo.value)
        assert "query" in str(excinfo.value)

    def test_to_provider_format_openai(self):
        tool_instance = DummySearchTool()
        openai_format = tool_instance.to_provider_format("openai")

        assert openai_format["type"] == "function"
        assert openai_format["name"] == "web_search"
        assert "query" in openai_format["parameters"]["properties"]
        assert openai_format["parameters"]["additionalProperties"] is False

    def test_to_provider_format_anthropic(self):
        tool_instance = DummySearchTool()
        anthropic_format = tool_instance.to_provider_format("anthropic")

        assert anthropic_format["name"] == "web_search"
        assert "input_schema" in anthropic_format
        assert "query" in anthropic_format["input_schema"]["properties"]

    def test_to_provider_format_unsupported_raises_error(self):
        """Test that requesting an unsupported provider format raises ValueError."""
        tool_instance = DummySearchTool()
        with pytest.raises(ValueError) as excinfo:
            tool_instance.to_provider_format("unsupported_provider")
        assert "Unsupported provider format requested" in str(excinfo.value)

    def test_repr_method(self):
        """Test the __repr__ string representation of the tool."""
        tool_instance = DummySearchTool()
        representation = repr(tool_instance)
        assert representation == "DummySearchTool(name='web_search')"


class TestToolRegistry:
    """Tests for the tool registry and factory."""

    def setup_method(self):
        """Clean the registry for each test method."""
        self._original_registry = _tool_registry.copy()
        _tool_registry.clear()

    def teardown_method(self):
        """Restore the original registry state."""
        _tool_registry.clear()
        _tool_registry.update(self._original_registry)

    def test_tool_registration(self):
        """Test that the @tool decorator correctly registers a class."""
        assert "my_tool" not in ToolRegistry.list_tools()

        @tool
        class MyTool(BaseTool):
            name = "my_tool"

            def execute(self, **kwargs: Any) -> Dict[str, Any]:
                return {}

        assert "my_tool" in ToolRegistry.list_tools()

    def test_registration_fails_for_non_tool_class(self):
        """Test that the @tool decorator raises TypeError for invalid classes."""
        with pytest.raises(TypeError) as excinfo:

            @tool  # pyright: ignore[reportArgumentType]
            class NotATool:
                name = "invalid"

        assert "Registered class must be a subclass of BaseTool" in str(excinfo.value)

    def test_get_tool_success(self):
        """Test successfully getting a tool instance."""
        # Manually register the tool class for this isolated test.
        # The `tool` decorator takes the class itself as the argument.
        tool(DummySearchTool)

        instance = ToolRegistry.get_tool("web_search")
        assert isinstance(instance, DummySearchTool)
        assert instance.name == "web_search"

    def test_get_tool_not_found(self):
        """Test that getting a non-existent tool raises an error."""
        with pytest.raises(ToolNotFoundError) as excinfo:
            ToolRegistry.get_tool("non_existent_tool")
        assert "Tool 'non_existent_tool' not found" in str(excinfo.value)

    def test_duplicate_registration_raises_error(self):
        """Test that registering a tool with the same name twice fails."""

        @tool
        class Tool1(BaseTool):
            name = "duplicate_name"

            def execute(self, **kwargs: Any) -> Dict[str, Any]:
                return {}

        with pytest.raises(ValueError) as excinfo:

            @tool
            class Tool2(BaseTool):
                name = "duplicate_name"

                def execute(self, **kwargs: Any) -> Dict[str, Any]:
                    return {}

        assert "Tool 'duplicate_name' is already registered" in str(excinfo.value)

    def test_get_all_tools(self):
        """Test that get_all_tools returns instances of all registered tools."""
        # Manually register the DummySearchTool
        tool(DummySearchTool)

        # Register another tool for the test
        @tool
        class AnotherTool(BaseTool):
            name = "another_tool"

            def execute(self, **kwargs: Any) -> Dict[str, Any]:
                return {}

        all_tools = ToolRegistry.get_all_tools()

        assert len(all_tools) == 2
        tool_names = {t.name for t in all_tools}
        assert "web_search" in tool_names
        assert "another_tool" in tool_names
        assert isinstance(all_tools[0], BaseTool)
        assert isinstance(all_tools[1], BaseTool)
