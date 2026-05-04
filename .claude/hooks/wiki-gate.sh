#!/bin/bash
# wiki-gate.sh — Pre/Post hook for Edit|Write|MultiEdit
# PreToolUse: blocks writes to docs/wiki/* unless .claude/state/wiki-approved exists.
# PostToolUse: clears the flag after a successful wiki write (one-shot).
# Flag is set by the /approve-wiki command (user-invoked).

set -euo pipefail

INPUT=$(cat)
EVENT=$(echo "$INPUT" | jq -r '.hook_event_name // empty')
FILE=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

# Only act on writes to docs/wiki/
case "$FILE" in
  *docs/wiki/*) ;;
  *) exit 0 ;;
esac

FLAG=".claude/state/wiki-approved"

if [ "$EVENT" = "PreToolUse" ]; then
  if [ -f "$FLAG" ]; then
    exit 0
  fi
  {
    echo "[wiki-gate] Wiki writes require user approval."
    echo "[wiki-gate] User must invoke /approve-wiki to set the flag, then retry."
    echo "[wiki-gate] Target: $FILE"
  } >&2
  exit 2
fi

if [ "$EVENT" = "PostToolUse" ]; then
  rm -f "$FLAG"
  exit 0
fi

exit 0
