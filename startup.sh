#!/usr/bin/env bash

# ============================================================
# DSC180A Final Project – Environment Setup Script
# ============================================================
# This script:
#   • Ensures conda is available
#   • Creates or recreates the dsc180a-eval environment
#   • Installs all packages from environment.yml
#   • Activates the environment
#   • Creates a .env file if missing
#   • Prints usage instructions
#
# Usage:
#   bash startup.sh
# ============================================================

set -e  # Stop on errors

PROJECT_NAME="dsc180a-eval"
ENV_FILE="environment.yml"
DOTENV_FILE=".env"

echo "=============================================="
echo "  DSC180A FINAL PROJECT — ENVIRONMENT SETUP"
echo "=============================================="

# ------------------------------------------------------------
# Check conda availability
# ------------------------------------------------------------
if ! command -v conda &> /dev/null
then
    echo "❌ Conda is not installed or not on PATH."
    echo "   Please install Miniconda or Anaconda first:"
    echo "   https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✔ Conda found."

# ------------------------------------------------------------
# Confirm environment.yml exists
# ------------------------------------------------------------
if [ ! -f "$ENV_FILE" ]; then
    echo "❌ ERROR: $ENV_FILE not found in the project root."
    exit 1
fi

echo "✔ Found environment.yml"

# ------------------------------------------------------------
# Re-create environment
# ------------------------------------------------------------
echo "----------------------------------------------"
echo "Creating conda environment: $PROJECT_NAME"
echo "----------------------------------------------"

# Remove old env if it exists
recreate_env=false
if conda env list | grep -q "$PROJECT_NAME"; then
    echo "⚠ Environment '$PROJECT_NAME' already exists."
    read -p "   Delete and recreate it? (y/n): " yn
    case $yn in
        [Yy]* ) 
            echo "   Removing old environment..."
            conda env remove -n "$PROJECT_NAME" --yes
            recreate_env=true
            ;;
        * ) 
            echo "   Keeping existing environment."
            ;;
    esac
else
    recreate_env=true
fi

if [ "$recreate_env" = true ]; then
    echo "🌱 Creating (or updating) environment..."
    conda env create -f "$ENV_FILE"
else
    echo "🌱 Skipping environment creation; using existing '$PROJECT_NAME'."
fi

echo "✔ Environment created successfully."

# ------------------------------------------------------------
# Activate environment
# ------------------------------------------------------------
echo "----------------------------------------------"
echo "Activating environment: $PROJECT_NAME"
echo "----------------------------------------------"

# Initialize conda shell
eval "$(conda shell.bash hook)"

conda activate "$PROJECT_NAME"

echo "✔ Environment activated."

# ------------------------------------------------------------
# Create .env if missing
# ------------------------------------------------------------
if [ ! -f "$DOTENV_FILE" ]; then
    echo "----------------------------------------------"
    echo "Creating .env file (empty placeholder)"
    echo "----------------------------------------------"

    cat <<EOF > .env
# OpenAI API Key
OPENAI_API_KEY=

# Optional: logging level
LOG_LEVEL=INFO
EOF

    echo "✔ .env created. Please edit it and add your OpenAI API key."
else
    echo "✔ .env already exists – no changes made."
fi

# ------------------------------------------------------------
# Final instructions
# ------------------------------------------------------------
echo ""
echo "======================================================"
echo " Setup Complete! "
echo "======================================================"
echo ""
echo "IMPORTANT:"
echo "To run the setup script correctly, always use:"
echo "    source startup.sh"
echo ""
echo "To activate the environment in later sessions:"
echo "    conda activate $PROJECT_NAME"
echo ""
echo "To run an evaluation:"
echo "    python -m src.experiments.run_eval lecture1"
echo "    python -m src.experiments.run_eval lecture2 yes   # force regenerate S0"
echo ""
echo "To launch the interactive dashboard:"
echo \"    streamlit run src/visualization/interactive_dashboard.py\"
echo ""
echo "======================================================"
echo "You're all set. 🚀"
echo "======================================================"
