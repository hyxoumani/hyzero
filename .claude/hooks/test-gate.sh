#!/bin/bash
# Test Gate — Stop hook
# Runs the project's test command when Claude finishes a task.
# Exit 0 = pass (allow stop), Exit 2 = fail (Claude sees the failure and continues).

set -euo pipefail

# Extract test command from CLAUDE.md
# Tries multiple formats: "- **Test**:", "# Test", "**Test command**:", "cargo test", etc.
TEST_CMD=$(grep -iE '^\s*[-*#]*\s*\*{0,2}test( command)?\*{0,2}\s*:' CLAUDE.md 2>/dev/null | head -1 | sed 's/.*: *//' | tr -d '`' || echo "")

# If no test command found, warn and allow stop
if [ -z "$TEST_CMD" ] || [ "$TEST_CMD" = "_TBD_" ]; then
  echo "[test-gate] No test command found in CLAUDE.md — skipping" >&2
  exit 0
fi

# Run tests with timeout
OUTPUT=$(timeout 300 bash -c "$TEST_CMD" 2>&1) || {
  EXIT_CODE=$?
  if [ $EXIT_CODE -eq 124 ]; then
    echo "Test gate: tests timed out after 5 minutes" >&2
  else
    echo "Test gate: tests failed. Fix before completing." >&2
    echo "$OUTPUT" | tail -n 30 >&2
  fi
  exit 2
}

exit 0
