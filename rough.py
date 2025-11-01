from allos.tools.registry import ToolRegistry

# Get a tool
read_tool = ToolRegistry.get_tool("read_file")

# Execute it
result = read_tool.execute(path=".env.example")
print(result["content"])

# List all tools
print(ToolRegistry.list_tools())
