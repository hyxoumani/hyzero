#!/bin/bash
# Regression tests for safety-net.sh
# Usage: bash .claude/hooks/test-safety-net.sh
# Exit 0 if all cases pass, 1 otherwise.

set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HOOK="$SCRIPT_DIR/safety-net.sh"
PASS=0
FAIL=0

run_case() {
  local DESC="$1" CMD="$2" EXPECTED="$3"
  local INPUT
  INPUT=$(printf '{"tool_input":{"command":%s}}' "$(printf '%s' "$CMD" | jq -Rs .)")
  echo "$INPUT" | bash "$HOOK" >/dev/null 2>&1
  local ACTUAL=$?
  if [ "$ACTUAL" = "$EXPECTED" ]; then
    echo "  PASS [$EXPECTED] $DESC"
    PASS=$((PASS+1))
  else
    echo "  FAIL [exp=$EXPECTED got=$ACTUAL] $DESC :: $CMD"
    FAIL=$((FAIL+1))
  fi
}

echo "=== block cases (exit 2) ==="
run_case "rm -rf /"             "rm -rf /"                        2
run_case "rm -rf ."             "rm -rf ."                        2
run_case "git push --force"     "git push --force origin main"    2
run_case "git push -f"          "git push -f origin main"         2
run_case "git reset --hard"     "git reset --hard HEAD~1"         2
run_case "git clean -fd"        "git clean -fd"                   2
run_case "mkfs"                 "mkfs.ext4 /dev/sda1"             2
run_case "dd if="               "dd if=/dev/zero of=/dev/sda"     2

echo "=== allow cases (exit 0) ==="
run_case "ls"                   "ls -la"                          0
run_case "cargo test"           "cargo test"                      0
run_case "git push (no force)"  "git push origin feature-branch"  0
run_case "git status"           "git status"                      0

echo "=== regression: echo/printf whitelist (exit 0) ==="
run_case "echo with --force string"  'echo "git push --force is dangerous"'   0
run_case "echo with rm-rf in string" 'echo "rm -rf / would delete everything"' 0
run_case "printf with reset string"  'printf "git reset --hard\n"'            0

echo "=== regression: whitelist must NOT cover compound (exit 2) ==="
run_case "echo ; rm -rf /"      'echo hi; rm -rf /'                2
run_case "echo && force-push"   'echo hi && git push --force origin main' 2
run_case "echo | reset hard"    'echo hi | git reset --hard'       2

echo
echo "Passed: $PASS  Failed: $FAIL"
[ "$FAIL" -eq 0 ]
