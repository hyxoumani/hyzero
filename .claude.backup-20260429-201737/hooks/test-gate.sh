#!/bin/bash
# Test Gate — Stop hook
# Runs the project's test command when Claude finishes a task.
# Exit 0 = pass (allow stop), Exit 2 = fail (Claude sees the failure and continues).
#
# Reads the test command from CLAUDE.md. Format: a line beginning with `# Test`
# inside a fenced ```bash block, with the command on the next non-empty line.

set -euo pipefail

TEST_CMD=""
if [ -f CLAUDE.md ]; then
  TEST_CMD=$(awk '
    /^```(bash|sh)?$/ { in_block = 1; next }
    /^```$/ { in_block = 0; next }
    in_block && /^#[[:space:]]*[Tt]est[[:space:]]*$/ { found = 1; next }
    in_block && found && !/^[[:space:]]*$/ && !/^#/ { print; exit }
  ' CLAUDE.md)
fi

if [ -z "$TEST_CMD" ] || [ "$TEST_CMD" = "_TBD_" ]; then
  echo "[test-gate] No test command found in CLAUDE.md — skipping" >&2
  exit 0
fi

TIMEOUT_CMD=""
if command -v timeout >/dev/null 2>&1; then
  TIMEOUT_CMD="timeout 300"
elif command -v gtimeout >/dev/null 2>&1; then
  TIMEOUT_CMD="gtimeout 300"
fi

if [ -n "$TIMEOUT_CMD" ]; then
  OUTPUT=$($TIMEOUT_CMD bash -c "$TEST_CMD" 2>&1) || {
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 124 ]; then
      echo "[test-gate] Tests timed out after 5 minutes" >&2
    else
      echo "[test-gate] Tests failed. Fix before completing." >&2
      echo "$OUTPUT" | tail -n 30 >&2
    fi
    exit 2
  }
else
  OUTPUT=$(bash -c "$TEST_CMD" 2>&1) || {
    echo "[test-gate] Tests failed. Fix before completing." >&2
    echo "$OUTPUT" | tail -n 30 >&2
    exit 2
  }
fi

exit 0
