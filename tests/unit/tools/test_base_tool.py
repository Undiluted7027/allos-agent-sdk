# tests/unit/tools/test_base_tool.py
import pytest

from allos.tools.base import BaseTool, ToolParameter
from allos.utils.errors import ToolError


class ValidationTestTool(BaseTool):
    name = "validation_test"
    description = "A tool for testing validation."
    parameters = [
        ToolParameter(
            name="req_str", type="string", required=True, description="required string"
        ),
        ToolParameter(
            name="opt_int",
            type="integer",
            required=False,
            description="some integder option",
        ),
        ToolParameter(
            name="opt_bool",
            type="boolean",
            required=False,
            description="some boolean option",
        ),
    ]

    def execute(self, **kwargs):
        pass


@pytest.mark.parametrize(
    "args, should_pass",
    [
        ({"req_str": "hello", "opt_int": 123, "opt_bool": True}, True),
        ({"req_str": "hello"}, True),  # Optional args missing is OK
        # Coercion cases
        ({"req_str": "hello", "opt_int": "123"}, True),
        ({"req_str": "hello", "opt_bool": "true"}, True),
        ({"req_str": "hello", "opt_bool": "false"}, True),
        # Failure cases
        ({"opt_int": 123}, False),  # Missing required
        ({"req_str": 123}, False),  # Wrong type for string
        ({"req_str": "hello", "opt_int": "abc"}, False),  # Bad coercion
        ({"req_str": "hello", "opt_bool": "not_a_bool"}, False),
    ],
)
def test_argument_validation(args, should_pass):
    tool = ValidationTestTool()
    if should_pass:
        tool.validate_arguments(args)  # Should not raise
    else:
        with pytest.raises(ToolError):
            tool.validate_arguments(args)
