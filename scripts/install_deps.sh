#!/usr/bin/env bash
# Create a virtualenv and install dependencies from requirements.txt
set -euo pipefail
PYTHON=${PYTHON:-python3}
VENV_DIR=.venv

echo "Using python: $(command -v $PYTHON)"

if [ ! -x "$(command -v $PYTHON)" ]; then
  echo "Python not found: $PYTHON" >&2
  exit 1
fi

# create venv if missing
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtualenv in $VENV_DIR"
  $PYTHON -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

pip install --upgrade pip
pip install -r "$(dirname "$0")/../requirements.txt"

echo "Dependencies installed into $VENV_DIR. Activate with: source $VENV_DIR/bin/activate"
