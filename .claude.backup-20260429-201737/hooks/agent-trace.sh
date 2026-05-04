#!/bin/bash
# Agent Trace Logger — PreToolUse hook for Agent
# Logs every subagent dispatch for pipeline observability.
# Exit 0 = always allow (pure logging, no blocking).

set -euo pipefail

INPUT=$(cat)
AGENT_TYPE=$(echo "$INPUT" | jq -r '.tool_input.subagent_type // "general"')
DESCRIPTION=$(echo "$INPUT" | jq -r '.tool_input.description // "unknown"')
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

LOG_FILE="docs/sessions/agent-trace.log"
mkdir -p "$(dirname "$LOG_FILE")"
printf '%s\t%s\tspawn\t%s\n' "$TIMESTAMP" "$AGENT_TYPE" "$DESCRIPTION" >> "$LOG_FILE"

exit 0
