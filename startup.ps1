<#
==============================================================================
 DSC180A Final Project – Windows Environment Setup Script
==============================================================================

This script:
  • Ensures Conda is installed and available
  • Creates or recreates the dsc180a-eval Conda environment
  • Installs dependencies from environment.yml
  • Creates a .env file if missing
  • Prints usage instructions

Usage:
  PowerShell (Run as normal):
      powershell -ExecutionPolicy Bypass -File startup.ps1

------------------------------------------------------------------------------
#>

Write-Host "=============================================="
Write-Host "   DSC180A FINAL PROJECT — WINDOWS SETUP"
Write-Host "=============================================="

# ------------------------------------------------------------
# Check if conda is available
# ------------------------------------------------------------
Write-Host "`n🔍 Checking Conda installation..."

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Conda is not installed or not on PATH."
    Write-Host "   Please install Miniconda or Anaconda first:"
    Write-Host "   https://docs.conda.io/en/latest/miniconda.html"
    exit 1
}

Write-Host "✔ Conda found."

# ------------------------------------------------------------
# Check environment.yml exists
# ------------------------------------------------------------
$envFile = "environment.yml"

if (-not (Test-Path $envFile)) {
    Write-Host "❌ ERROR: environment.yml not found in project root."
    exit 1
}

Write-Host "✔ Found environment.yml"

# ------------------------------------------------------------
# Set environment name
# ------------------------------------------------------------
$envName = "dsc180a-eval"

Write-Host "`n----------------------------------------------"
Write-Host "Conda environment: $envName"
Write-Host "----------------------------------------------"

# ------------------------------------------------------------
# Remove existing environment?
# ------------------------------------------------------------
$existingEnv = conda env list | Select-String $envName
$recreate = $false

if ($existingEnv) {
    Write-Host "⚠ Environment '$envName' already exists."
    $resp = Read-Host "   Delete and recreate it? (y/n)"

    if ($resp -eq "y") {
        Write-Host "   Removing old environment..."
        conda env remove -n $envName --yes
        $recreate = $true
    }
    else {
        Write-Host "   Keeping existing environment."
    }
}
else {
    $recreate = $true
}

# ------------------------------------------------------------
# Create environment if needed
# ------------------------------------------------------------
if ($recreate) {
    Write-Host "`n🌱 Creating environment from environment.yml..."
    conda env create -f $envFile
} else {
    Write-Host "🌱 Skipping creation; using existing '$envName'."
}

Write-Host "✔ Environment created successfully."

# ------------------------------------------------------------
# .env creation
# ------------------------------------------------------------
$dotenv = ".env"

if (-not (Test-Path $dotenv)) {
    Write-Host "`n----------------------------------------------"
    Write-Host "Creating .env file (empty placeholder)"
    Write-Host "----------------------------------------------"

@"
# OpenAI API Key
OPENAI_API_KEY=

# Logging level
LOG_LEVEL=INFO
"@ | Out-File -Encoding utf8 .env

    Write-Host "✔ .env created. Please edit it and add your OpenAI API key."
}
else {
    Write-Host "✔ .env already exists — no changes made."
}

# ------------------------------------------------------------
# Completion message
# ------------------------------------------------------------
Write-Host "`n======================================================"
Write-Host " Setup Complete! "
Write-Host "======================================================"
Write-Host ""
Write-Host "To activate the environment in PowerShell, run:"
Write-Host "    conda activate $envName"
Write-Host ""
Write-Host "To run an evaluation:"
Write-Host "    python -m src.experiments.run_eval lecture1"
Write-Host "    python -m src.experiments.run_eval lecture2 yes   # force regenerate S0"
Write-Host ""
Write-Host "To launch the interactive dashboard:"
Write-Host "    streamlit run src/visualization/interactive_dashboard.py"
Write-Host ""
Write-Host "======================================================"
Write-Host "You're all set! 🚀"
Write-Host "======================================================"
