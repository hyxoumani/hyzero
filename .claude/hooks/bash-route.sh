#!/bin/bash
# bash-route.sh — PreToolUse(Bash)
# Thin-orchestrator: orchestrator does not read or edit source via Bash.
# Subagents run Bash freely in their own contexts.

set -euo pipefail

INPUT=$(cat)
AGENT_ID=$(echo "$INPUT" | jq -r '.agent_id // empty')

# Subagent context — allow freely
[ -n "$AGENT_ID" ] && exit 0

CMD=$(echo "$INPUT" | jq -r '.tool_input.command // empty')
[ -z "$CMD" ] && exit 0

# Patterns that spill or modify file content. Each is an extended regex.
BLOCKED_PATTERNS=(
  # Content viewers
  '(^|[^[:alnum:]])cat[[:space:]]+[^|<>&]'
  '(^|[^[:alnum:]])head[[:space:]]'
  '(^|[^[:alnum:]])tail[[:space:]]'
  '(^|[^[:alnum:]])less[[:space:]]'
  '(^|[^[:alnum:]])more[[:space:]]'
  # git content forms
  '(^|[^[:alnum:]])git[[:space:]]+diff([[:space:]]|$)'
  '(^|[^[:alnum:]])git[[:space:]]+show([[:space:]]|$)'
  '(^|[^[:alnum:]])git[[:space:]]+blame'
  '(^|[^[:alnum:]])git[[:space:]]+log[[:space:]]+.*-p'
  # grep with context flags (spills surrounding content)
  'grep[[:space:]]+(-[a-zA-Z]*)?-[ABC]'
  # In-place editors / content-emitters
  '(^|[^[:alnum:]])awk[[:space:]]'
  '(^|[^[:alnum:]])sed[[:space:]]'
  '(^|[^[:alnum:]])tee[[:space:]]'
)

for PATTERN in "${BLOCKED_PATTERNS[@]}"; do
  if echo "$CMD" | grep -qE "$PATTERN"; then
    {
      echo "[bash-route] Orchestrator does not read or edit source via Bash."
      echo "[bash-route] Dispatch analyst (read) or developer (edit). Subagents run Bash freely."
      echo "[bash-route] Blocked: $CMD"
    } >&2
    exit 2
  fi
done

exit 0
