#!/bin/bash
# Safety Net — PreToolUse hook for Bash
# Blocks destructive commands that are hard to reverse.
# Exit 0 = allow, Exit 2 = block.

set -euo pipefail

INPUT=$(cat)
CMD=$(echo "$INPUT" | jq -r '.tool_input.command // empty')

[ -z "$CMD" ] && exit 0

# Whitelist: pure echo/printf with no shell composition.
# Avoids false positives on informational strings like
#   echo "git push --force is dangerous"
# Compound forms (echo ... ; rm -rf /) still fall through to pattern checks.
case "$CMD" in
  echo|printf|echo\ *|printf\ *)
    case "$CMD" in
      *';'*|*'&&'*|*'||'*|*'|'*|*'`'*|*'$('*) ;;
      *) exit 0 ;;
    esac
    ;;
esac

# Patterns are matched as POSIX extended regex against the command string.
# Each pattern uses \b or explicit anchors to avoid matching as substrings.
BLOCKED_PATTERNS=(
  '(^|[^[:alnum:]])rm[[:space:]]+(-[a-zA-Z]*[rRfF][a-zA-Z]*[[:space:]]+)+/([[:space:]]|$)'
  '(^|[^[:alnum:]])rm[[:space:]]+(-[a-zA-Z]*[rRfF][a-zA-Z]*[[:space:]]+)+\.([[:space:]]|$)'
  '(^|[^[:alnum:]])rm[[:space:]]+(-[a-zA-Z]*[rRfF][a-zA-Z]*[[:space:]]+)+\*([[:space:]]|$)'
  'git[[:space:]]+push.*--force'
  'git[[:space:]]+push.*[[:space:]]-f([[:space:]]|$)'
  'git[[:space:]]+reset[[:space:]]+--hard'
  'git[[:space:]]+clean[[:space:]]+-[a-zA-Z]*f'
  'git[[:space:]]+checkout[[:space:]]+--[[:space:]]+\.([[:space:]]|$)'
  'git[[:space:]]+restore[[:space:]]+\.([[:space:]]|$)'
  '>[[:space:]]*/dev/sd[a-z]'
  'mkfs\.'
  'dd[[:space:]]+if='
  ':\(\)\{:\|:&\};:'
)

for PATTERN in "${BLOCKED_PATTERNS[@]}"; do
  if echo "$CMD" | grep -qE "$PATTERN"; then
    echo "[safety-net] Blocked destructive command: $CMD" >&2
    echo "[safety-net] If intentional, ask the user to run it manually." >&2
    exit 2
  fi
done

exit 0
