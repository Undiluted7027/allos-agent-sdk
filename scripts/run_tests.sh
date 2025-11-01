#!/bin/bash
# scripts/run_tests.sh

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- Running Allos Agent SDK Tests with Coverage ---"

# The root directory of the project
PROJECT_ROOT=$(git rev-parse --show-toplevel)

# Run pytest from the project root
# --cov=allos: Measure coverage for the 'allos' package
# --cov-report term-missing: Show lines that are not covered in the terminal
# "$@": Pass any additional arguments from the command line to pytest
#      (e.g., ./scripts/run_tests.sh -k "test_specific_feature")
python -m pytest \
  --cov=allos \
  --cov-report term-missing \
  "$@"

echo ""
echo "--- Tests Passed! ---"
echo "To generate an HTML coverage report, run:"
echo "python -m pytest --cov=allos --cov-report=html"
