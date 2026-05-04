#!/bin/bash
# Smoke test runner — runs eval checks directly without git worktrees.
# Usage: bash evals/run-smoke.sh [task-name]
#
# Run from the framework root (the directory containing hooks/, agents/, etc.)

set -euo pipefail

TASKS_DIR="evals/tasks"
SPECIFIC_TASK="${1:-}"
PASS=0
FAIL=0
ERRORS=""

run_task() {
  local TASK_FILE="$1"
  local TASK_NAME=$(basename "$TASK_FILE" .md)
  local CHECKS_PASSED=0
  local CHECKS_TOTAL=0
  local TASK_ERRORS=""

  echo "━━━ $TASK_NAME ━━━"

  # Extract verification checks — lines matching "- [ ] `command`"
  while IFS= read -r line; do
    CMD=$(echo "$line" | sed -n 's/.*`\(.*\)`.*/\1/p')
    [ -z "$CMD" ] && continue
    CHECKS_TOTAL=$((CHECKS_TOTAL + 1))

    # Replace .claude/ prefix since we're running from the framework dir itself
    CMD=$(echo "$CMD" | sed 's|\.claude/||g')

    if eval "$CMD" >/dev/null 2>&1; then
      echo "  ✓ $CMD"
      CHECKS_PASSED=$((CHECKS_PASSED + 1))
    else
      echo "  ✗ $CMD"
      TASK_ERRORS="$TASK_ERRORS\n  FAIL: $CMD"
    fi
  done < <(sed -n '/^## Verification/,/^## /p' "$TASK_FILE" | grep '^\- \[')

  if [ "$CHECKS_TOTAL" -eq 0 ]; then
    echo "  (no checks found)"
    return
  fi

  if [ "$CHECKS_PASSED" -eq "$CHECKS_TOTAL" ]; then
    echo "  PASS ($CHECKS_PASSED/$CHECKS_TOTAL)"
    PASS=$((PASS + 1))
  else
    echo "  FAIL ($CHECKS_PASSED/$CHECKS_TOTAL)"
    FAIL=$((FAIL + 1))
    ERRORS="$ERRORS\n[$TASK_NAME]$TASK_ERRORS"
  fi
}

# Run tasks
if [ -n "$SPECIFIC_TASK" ]; then
  TASK_FILE="$TASKS_DIR/$SPECIFIC_TASK.md"
  [ ! -f "$TASK_FILE" ] && echo "Task not found: $TASK_FILE" && exit 1
  run_task "$TASK_FILE"
else
  for TASK_FILE in "$TASKS_DIR"/*.md; do
    [ ! -f "$TASK_FILE" ] && continue
    run_task "$TASK_FILE"
    echo ""
  done
fi

# Summary
echo "━━━━━━━━━━━━━━━━━"
TOTAL=$((PASS + FAIL))
echo "Total: $TOTAL | Pass: $PASS | Fail: $FAIL"

if [ -n "$ERRORS" ]; then
  echo ""
  echo "Failures:"
  echo -e "$ERRORS"
fi

[ "$FAIL" -eq 0 ] && exit 0 || exit 1
