#!/bin/bash
# wiki-gate-bash.sh — Pre/Post hook for Bash writes to docs/wiki/*
# Closes the matcher gap where Bash heredoc/redirect bypasses wiki-gate.sh.
# Applies to ALL contexts (main and subagent) — wiki gate is about user approval,
# not about orchestrator-vs-subagent boundary, so no agent_id exemption.

set -euo pipefail

INPUT=$(cat)
EVENT=$(echo "$INPUT" | jq -r '.hook_event_name // empty')
CMD=$(echo "$INPUT" | jq -r '.tool_input.command // empty')

[ -z "$CMD" ] && exit 0

# Detect commands that write to docs/wiki/ via shell redirect or tee.
WIKI_WRITE_PATTERNS=(
  '>[[:space:]]*[^|&;]*docs/wiki/'
  '>>[[:space:]]*[^|&;]*docs/wiki/'
  'tee[[:space:]]+(-[a-zA-Z]*[[:space:]]+)*[^|&;]*docs/wiki/'
)

WRITES_WIKI=0
for PATTERN in "${WIKI_WRITE_PATTERNS[@]}"; do
  if echo "$CMD" | grep -qE "$PATTERN"; then
    WRITES_WIKI=1
    break
  fi
done

[ $WRITES_WIKI -eq 0 ] && exit 0

FLAG=".claude/state/wiki-approved"

if [ "$EVENT" = "PreToolUse" ]; then
  if [ -f "$FLAG" ]; then
    exit 0
  fi
  {
    echo "[wiki-gate-bash] Wiki writes via Bash require user approval."
    echo "[wiki-gate-bash] User must invoke /approve-wiki to set the flag, then retry."
    echo "[wiki-gate-bash] Blocked: $CMD"
  } >&2
  exit 2
fi

if [ "$EVENT" = "PostToolUse" ]; then
  rm -f "$FLAG"
  exit 0
fi

exit 0
