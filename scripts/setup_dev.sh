#!/bin/bash
# scripts/setup_dev.sh

# Exit immediately if a command exits with a non-zero status.
set -e

# Helper for colored output
bold=$(tput bold)
green=$(tput setaf 2)
reset=$(tput sgr0)

echo "${bold}--- Setting up Allos Agent SDK Development Environment ---${reset}"

# 1. Check for uv
if ! command -v uv &> /dev/null
then
    echo "❌ ${bold}uv is not installed.${reset}"
    echo "Please install it first by following the instructions at https://astral.sh/uv"
    exit 1
fi
echo "✅ uv is installed."

# 2. Create virtual environment
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "🐍 Creating virtual environment in '$VENV_DIR'..."
    uv venv --python $(which python3)
else
    echo "🐍 Virtual environment already exists."
fi

# 3. Install dependencies
echo "📦 Installing dependencies from pyproject.toml..."
uv pip install -e ".[dev]"
echo "${green}✅ Dependencies installed successfully.${reset}"

# 4. Install pre-commit hooks
echo "🎣 Installing pre-commit hooks..."
pre-commit install
echo "${green}✅ Pre-commit hooks installed.${reset}"

# 5. Setup .env file
if [ ! -f ".env" ]; then
    echo "🔑 Creating .env file from .env.example..."
    cp .env.example .env
    echo "${green}✅ .env file created. Please add your API keys to it.${reset}"
else
    echo "🔑 .env file already exists."
fi

echo ""
echo "${bold}${green}🚀 Development environment setup complete!${reset}"
echo ""
echo "${bold}Next steps:${reset}"
echo "1. Activate the virtual environment: ${green}source .venv/bin/activate${reset}"
echo "2. Add your API keys to the ${green}.env${reset} file."
echo "3. Run the tests to verify the setup: ${green}./scripts/run_tests.sh${reset}"
