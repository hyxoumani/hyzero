#!/bin/bash
# Eval Runner — executes eval tasks and grades results
# Usage: bash .claude/hooks/eval-runner.sh [task-name] [--trials N]

set -euo pipefail

EVAL_DIR=".claude/evals"
TASKS_DIR="$EVAL_DIR/tasks"
RESULTS_DIR="$EVAL_DIR/results"
TRIALS=1
SPECIFIC_TASK=""
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
RESULTS_FILE="$RESULTS_DIR/$TIMESTAMP.tsv"

# Parse args
while [[ $# -gt 0 ]]; do
  case $1 in
    --trials) TRIALS="$2"; shift 2 ;;
    *) SPECIFIC_TASK="$1"; shift ;;
  esac
done

mkdir -p "$RESULTS_DIR"

# Header
echo -e "task\ttrial\tstatus\tduration_s\tchecks_passed\tchecks_total\tnotes" > "$RESULTS_FILE"

# Collect tasks
if [ -n "$SPECIFIC_TASK" ]; then
  TASK_FILES=("$TASKS_DIR/$SPECIFIC_TASK.md")
else
  TASK_FILES=("$TASKS_DIR"/*.md)
fi

TOTAL_PASS=0
TOTAL_FAIL=0
TOTAL_TASKS=0

for TASK_FILE in "${TASK_FILES[@]}"; do
  [ ! -f "$TASK_FILE" ] && continue

  TASK_NAME=$(basename "$TASK_FILE" .md)
  TOTAL_TASKS=$((TOTAL_TASKS + 1))
  TASK_PASSED=false

  echo "━━━ Eval: $TASK_NAME ━━━"

  for TRIAL in $(seq 1 "$TRIALS"); do
    echo "  Trial $TRIAL/$TRIALS..."
    START_TIME=$(date +%s)

    # Create isolated worktree
    WORKTREE_DIR=$(mktemp -d)
    BRANCH_NAME="eval/$TASK_NAME-$TIMESTAMP-t$TRIAL"
    git worktree add -q "$WORKTREE_DIR" -b "$BRANCH_NAME" HEAD 2>/dev/null || {
      echo "  SKIP: could not create worktree"
      echo -e "$TASK_NAME\t$TRIAL\tskip\t0\t0\t0\tworktree creation failed" >> "$RESULTS_FILE"
      continue
    }

    # Run setup if specified
    SETUP_CMD=$(grep '^setup:' "$TASK_FILE" | sed 's/^setup: *//' || echo "")
    if [ -n "$SETUP_CMD" ]; then
      (cd "$WORKTREE_DIR" && bash -c "$SETUP_CMD" 2>/dev/null) || true
    fi

    # Extract task description (everything between ## Task and ## Verification)
    TASK_DESC=$(sed -n '/^## Task/,/^## Verification/p' "$TASK_FILE" | head -n -1 | tail -n +2)

    # Extract verification commands
    CHECKS=()
    while IFS= read -r line; do
      # Match lines like: - [ ] `command here`
      CMD=$(echo "$line" | grep -oP '`\K[^`]+' || echo "")
      [ -n "$CMD" ] && CHECKS+=("$CMD")
    done < <(sed -n '/^## Verification/,/^## /p' "$TASK_FILE" | grep '^\- \[')

    # Execute: spawn headless Claude with the task
    TIMEOUT_VAL=$(grep '^timeout:' "$TASK_FILE" | sed 's/^timeout: *//' || echo "180")
    (cd "$WORKTREE_DIR" && timeout "$TIMEOUT_VAL" claude -p "$TASK_DESC" --allowedTools "Read,Write,Edit,Bash,Grep,Glob" 2>/dev/null) || true

    # Grade: run each verification check
    CHECKS_PASSED=0
    CHECKS_TOTAL=${#CHECKS[@]}
    FAIL_NOTES=""

    for CHECK in "${CHECKS[@]}"; do
      if (cd "$WORKTREE_DIR" && bash -c "$CHECK" >/dev/null 2>&1); then
        CHECKS_PASSED=$((CHECKS_PASSED + 1))
      else
        FAIL_NOTES="$FAIL_NOTES; FAIL: $CHECK"
      fi
    done

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    # Determine status
    if [ "$CHECKS_PASSED" -eq "$CHECKS_TOTAL" ] && [ "$CHECKS_TOTAL" -gt 0 ]; then
      STATUS="pass"
      TASK_PASSED=true
    else
      STATUS="fail"
    fi

    # Clean up trailing semicolons
    FAIL_NOTES=$(echo "$FAIL_NOTES" | sed 's/^; //')

    echo -e "$TASK_NAME\t$TRIAL\t$STATUS\t$DURATION\t$CHECKS_PASSED\t$CHECKS_TOTAL\t$FAIL_NOTES" >> "$RESULTS_FILE"
    echo "  $STATUS ($CHECKS_PASSED/$CHECKS_TOTAL checks, ${DURATION}s)"

    # Cleanup worktree
    git worktree remove --force "$WORKTREE_DIR" 2>/dev/null || rm -rf "$WORKTREE_DIR"
    git branch -D "$BRANCH_NAME" 2>/dev/null || true
  done

  if $TASK_PASSED; then
    TOTAL_PASS=$((TOTAL_PASS + 1))
  else
    TOTAL_FAIL=$((TOTAL_FAIL + 1))
  fi
done

# Summary
echo ""
echo "━━━ Results ━━━"
echo "Tasks: $TOTAL_TASKS | Pass: $TOTAL_PASS | Fail: $TOTAL_FAIL"
if [ "$TOTAL_TASKS" -gt 0 ]; then
  PASS_RATE=$((TOTAL_PASS * 100 / TOTAL_TASKS))
  echo "pass@1: ${PASS_RATE}%"
fi
if [ "$TRIALS" -gt 1 ]; then
  echo "(pass@$TRIALS uses any-pass-in-k-trials — see $RESULTS_FILE)"
fi
echo "Results: $RESULTS_FILE"
