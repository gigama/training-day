#!/bin/bash
# Create venv and install dependencies for pytorch-training

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

echo "Creating virtual environment at $VENV_DIR..."
python3 -m venv "$VENV_DIR"

echo "Installing dependencies..."
"$VENV_DIR/bin/pip" install --upgrade pip --quiet
#"$VENV_DIR/bin/pip" install -r "$SCRIPT_DIR/requirements.txt" --quiet

source .venv/bin/activate
pip install pandas numpy
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo
echo "Setup complete."
echo
echo "Activate with:  source $VENV_DIR/bin/activate"
echo
echo "Usage:"
echo "  python3 training-day.py --input text-out/ --output training-data/ --type True"
echo "  python3 training-day.py --input text-out/ --output training-data/ --type False"
