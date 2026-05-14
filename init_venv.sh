#!/bin/bash

set -e

VENV_DIR="atomic_watermelon_venv"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"   # override with PYTHON_BIN=python3.11 ./setup.sh

# --- Verify interpreter exists and is 3.10+ ---
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "‼️ '$PYTHON_BIN' not found. Install it (e.g. 'brew install python@3.12') or set PYTHON_BIN."
  exit 1
fi

PY_VERSION=$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=$("$PYTHON_BIN" -c 'import sys; print(sys.version_info.major)')
PY_MINOR=$("$PYTHON_BIN" -c 'import sys; print(sys.version_info.minor)')

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
  echo "‼️ Need Python 3.10+ for PEP 604 union types. Found $PY_VERSION."
  exit 1
fi
echo "🐍 Using $PYTHON_BIN ($PY_VERSION)"

# --- Remove old venv if it exists ---
if [ -d "$VENV_DIR" ]; then
  echo "🧹 Removing existing virtual environment '$VENV_DIR'..."
  rm -rf "$VENV_DIR"
fi

# --- Create venv ---
echo "🐍 Creating virtual environment..."
"$PYTHON_BIN" -m venv "$VENV_DIR"

# --- Activate ---
echo "🚀 Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# --- Check for requirements.txt ---
if [ ! -f requirements.txt ]; then
  echo "‼️ 'requirements.txt' NOT FOUND"
  deactivate
  exit 1
fi

# --- Upgrade pip ---
echo "🦾 Upgrading pip..."
pip install --upgrade pip

# --- Install dependencies ---
echo "📦 Installing dependencies..."
pip install -r requirements.txt

echo "✅ Setup complete on Python $PY_VERSION."

exec $SHELL