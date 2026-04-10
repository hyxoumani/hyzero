#!/bin/bash
# Read-Only Guard — PreToolUse hook for Write|Edit|MultiEdit
# Blocks edits to files listed as read-only in CLAUDE.md.
# Exit 0 = allow, Exit 2 = block.

set -euo pipefail

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // .tool_input.path // empty')

[ -z "$FILE_PATH" ] && exit 0

# Extract read-only files from CLAUDE.md
# Looks for the "Read-only files" line and parses the value
READONLY_PATTERN=$(grep -i "read-only files" CLAUDE.md 2>/dev/null | sed 's/.*: *//' | tr -d '`_' || echo "")

[ -z "$READONLY_PATTERN" ] && exit 0

# Check each read-only pattern
for PATTERN in $READONLY_PATTERN; do
  if [[ "$FILE_PATH" == *"$PATTERN"* ]]; then
    echo "Blocked: $FILE_PATH is listed as read-only in CLAUDE.md" >&2
    exit 2
  fi
done

exit 0
