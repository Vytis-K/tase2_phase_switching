#!/bin/zsh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

exec python3 "$SCRIPT_DIR/run_desktop_app.py"
