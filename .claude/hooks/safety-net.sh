#!/bin/bash
# Safety Net — PreToolUse hook for Bash
# Blocks destructive commands that are hard to reverse.
# Exit 0 = allow, Exit 2 = block.

set -euo pipefail

INPUT=$(cat)
CMD=$(echo "$INPUT" | jq -r '.tool_input.command // empty')

[ -z "$CMD" ] && exit 0

# Destructive patterns to block
BLOCKED_PATTERNS=(
  'rm -rf /'
  'rm -rf \.'
  'rm -rf \*'
  'git push.*--force'
  'git push.*-f '
  'git reset --hard'
  'git clean -fd'
  'git checkout -- \.'
  'git restore \.'
  '> /dev/sda'
  'mkfs\.'
  'dd if='
  ':(){:|:&};:'
)

for PATTERN in "${BLOCKED_PATTERNS[@]}"; do
  if echo "$CMD" | grep -qE "$PATTERN"; then
    echo "[safety-net] Blocked destructive command: $CMD" >&2
    echo "[safety-net] If intentional, ask the user to run it manually." >&2
    exit 2
  fi
done

exit 0
