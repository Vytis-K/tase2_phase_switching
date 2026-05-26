#!/bin/zsh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

if [[ -x "$SCRIPT_DIR/.venv/bin/python3" ]]; then
    PYTHON="$SCRIPT_DIR/.venv/bin/python3"
else
    PYTHON="$(command -v python3)"
fi

if [[ -z "$PYTHON" ]]; then
    echo "Could not find python3. Install Python 3.11 or activate the project environment first."
    read -r "?Press Return to close this window."
    exit 1
fi

exec "$PYTHON" "$SCRIPT_DIR/run_desktop_app.py" --upload-first "$@"
