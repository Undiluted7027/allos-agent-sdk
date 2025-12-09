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


class TestBaseToolCoercion:
    def test_validate_and_coerce_ignores_extra_arguments(self):
        """
        Test that _validate_and_coerce_types ignores arguments not in the
        tool's parameter map, covering the 'if name not in param_map' branch.
        """
        tool = ValidationTestTool()
        arguments = {"req_str": "hello", "extra_arg": "this should be ignored"}
        # This method modifies the dictionary in place
        tool._validate_and_coerce_types(arguments)

        # The method should complete without error and the extra arg should remain
        assert "extra_arg" in arguments

    def test_coerce_type_for_number_success(self):
        """
        Test the successful coercion of a string to a float for type 'number',
        covering the 'if expected_type == "number"' try block.
        """
        tool = ValidationTestTool()  # Can use any tool instance
        value = "123.45"
        coerced = tool._coerce_type(value, "number")
        assert isinstance(coerced, float)
        assert coerced == 123.45

    def test_coerce_type_for_number_failure(self):
        """
        Test that a non-numeric string for type 'number' is not coerced,
        covering the except block.
        """
        tool = ValidationTestTool()
        value = "not-a-float"
        coerced = tool._coerce_type(value, "number")
        # It should fail coercion and return the original string
        assert isinstance(coerced, str)
        assert coerced == "not-a-float"
