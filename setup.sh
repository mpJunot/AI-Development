#!/bin/bash

set -e

if [[ "$OS" == "Windows_NT" ]]; then
    echo "Windows detected."
    echo "Please run the following commands in Command Prompt or PowerShell:"
    echo ""
    echo "python -m venv ai"
    echo "ai\\Scripts\\activate"
    echo "python -m pip install --upgrade pip"
    echo "pip install -r requirements.txt"
    echo ""
    echo "Or use Windows Subsystem for Linux (WSL) and re-run this script."
    exit 0
fi

if ! command -v python3 &> /dev/null
then
    echo "Python3 could not be found. Please install Python 3."
    exit 1
fi

echo "Creating virtual environment 'ai'..."
python3 -m venv ai

echo "Activating virtual environment..."
source ai/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing requirements from requirements.txt..."
pip install -r requirements.txt

echo "Setup complete. To activate the environment later, run:"
echo "source ai/bin/activate"
