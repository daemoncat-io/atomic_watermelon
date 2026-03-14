#!/bin/bash

set -e  # Exit on error

VENV_DIR="research_transformer_venv" 

# --- Remove old venv if it exists ---
if [ -d "$VENV_DIR" ]; then
  echo "🧹 Removing existing virtual environment '$VENV_DIR'... "
  rm -rf "$VENV_DIR"
  echo "✅ Virtual environment '$VENV_DIR' has been removed."
fi

# --- Create virtual environment ---
echo "🐍 Creating virtual environment .."
python3 -m venv "$VENV_DIR"

# --- Activate venv ---
echo "🚀 Activating virtual environment..."
source research_transformer_venv/bin/activate

# --- Check for requirements.txt ---
if [ ! -f requirements.txt ]; then
  echo "‼️‼️‼️ ERROR ‼️‼️‼️ 'requirements.txt' NOT FOUND ‼️‼️‼️"
  deactivate
  exit 1
fi

# --- Upgrade pip ---
echo "🦾 Upgrading pip..."
pip install --upgrade pip

# --- Install base requirements ---
echo "📦 Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# --- Done ---
echo "✅ Setup complete. Virtual environment '$VENV_DIR' is active."

# --- Keep shell alive in venv ---
exec $SHELL