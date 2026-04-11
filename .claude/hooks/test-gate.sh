#!/bin/bash
# Test Gate — Stop hook
# Runs the project's test command when Claude finishes a task.
# Exit 0 = pass (allow stop), Exit 2 = fail (Claude sees the failure and continues).

set -euo pipefail

# Extract test command from CLAUDE.md
# Supports two formats:
#   1. Inline: "# Test: cargo test" or "**Test**: cargo test"
#   2. Code block: "# Test\ncargo test" inside ```bash blocks
TEST_CMD=""

# Try inline format first (command after colon on same line)
TEST_CMD=$(grep -iE '^\s*[-*#]*\s*\*{0,2}test( command)?\*{0,2}\s*:' CLAUDE.md 2>/dev/null | head -1 | sed 's/.*: *//' | tr -d '`' || echo "")

# If not found, try code block format (# Test comment, command on next line)
if [ -z "$TEST_CMD" ]; then
  TEST_CMD=$(awk '/^```(bash|sh)?$/,/^```$/' CLAUDE.md 2>/dev/null | awk '/^#[[:space:]]*[Tt]est[[:space:]]*$/{getline; if ($0 !~ /^(#|```|[[:space:]]*$)/) print; exit}' || echo "")
fi

# If no test command found, warn and allow stop
if [ -z "$TEST_CMD" ] || [ "$TEST_CMD" = "_TBD_" ]; then
  echo "[test-gate] No test command found in CLAUDE.md — skipping" >&2
  exit 0
fi

# Run tests with timeout (portable: try timeout, gtimeout, then no-timeout fallback)
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
      echo "Test gate: tests timed out after 5 minutes" >&2
    else
      echo "Test gate: tests failed. Fix before completing." >&2
      echo "$OUTPUT" | tail -n 30 >&2
    fi
    exit 2
  }
else
  OUTPUT=$(bash -c "$TEST_CMD" 2>&1) || {
    echo "Test gate: tests failed. Fix before completing." >&2
    echo "$OUTPUT" | tail -n 30 >&2
    exit 2
  }
fi

exit 0
