#!/usr/bin/env bash
set -e

# Ensure CBORG_API_KEY is available (prefer existing env var; otherwise prompt)
if [ -z "${CBORG_API_KEY:-}" ]; then
  read -s -p "Enter CBORG_API_KEY (input hidden): " CBORG_API_KEY
  echo
fi

# Required for Codex CLI to talk to CBorg/LBL
export OPENAI_API_KEY="$CBORG_API_KEY"
export OPENAI_BASE_URL="https://api.cborg.lbl.gov"

# Default model recommendation from the page
# If user already provided --model/-m, don't override.
if printf '%s\n' "$@" | grep -Eq '(^| )(--model|-m)(=| )'; then
  exec codex "$@"
else
  exec codex -m gpt-5-codex "$@"
fi