#!/bin/bash
# read-route.sh — PreToolUse(Read)
# Thin-orchestrator: orchestrator does not read source >50 lines.
# Subagents (agent_id present in stdin) read freely.

set -euo pipefail

INPUT=$(cat)
AGENT_ID=$(echo "$INPUT" | jq -r '.agent_id // empty')

# Subagent context — allow freely
[ -n "$AGENT_ID" ] && exit 0

LIMIT=$(echo "$INPUT" | jq -r '.tool_input.limit // empty')
FILE=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

# Allow if explicitly bounded ≤50
if [ -n "$LIMIT" ] && [ "$LIMIT" -le 50 ] 2>/dev/null; then
  exit 0
fi

# Allow if file is small (≤50 lines)
LINES=0
if [ -f "$FILE" ]; then
  LINES=$(wc -l < "$FILE" 2>/dev/null || echo 0)
fi
if [ "$LINES" -le 50 ] 2>/dev/null; then
  exit 0
fi

{
  echo "[read-route] Orchestrator does not read source >50 lines."
  echo "[read-route] File: $FILE ($LINES lines). Dispatch analyst with a brief."
  echo "[read-route] Or pass 'limit' ≤50 if you only need a small range."
} >&2
exit 2
