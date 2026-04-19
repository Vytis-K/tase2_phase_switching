#!/usr/bin/env bash
set -e

# Require CBORG_API_KEY to be set in your environment
# Get CBORG_API_KEY (prefer env var; otherwise prompt)
if [ -z "${CBORG_API_KEY:-}" ]; then
  read -s -p "Enter CBORG_API_KEY (input hidden): " CBORG_API_KEY
  echo
  export CBORG_API_KEY
fi

# Set authorization and base URL
export ANTHROPIC_AUTH_TOKEN="$CBORG_API_KEY"
export ANTHROPIC_BASE_URL="https://api.cborg.lbl.gov"

# Model selection (as shown on the page)
export ANTHROPIC_DEFAULT_HAIKU_MODEL="claude-haiku-4-5"
export ANTHROPIC_DEFAULT_SONNET_MODEL="claude-sonnet-4-6"
export ANTHROPIC_DEFAULT_OPUS_MODEL="claude-opus-4-6"

# Default conversation model
export ANTHROPIC_MODEL="claude-sonnet-4-6"

# Default subagent model
export CLAUDE_CODE_SUBAGENT_MODEL="claude-haiku-4-5"

# Recommended settings
export DISABLE_NON_ESSENTIAL_MODEL_CALLS=1
export DISABLE_TELEMETRY=1
export CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=8192

# Launch Claude Code (pass through any args you provide)
exec claude "$@"

