#!/bin/bash
# Commit Review Gate — PreToolUse hook for Bash(git commit*)
# Reviews the full staged diff before allowing a commit.
# Exit 0 = approve, Exit 2 = deny (Claude sees the failure and fixes).

set -euo pipefail

INPUT=$(cat)

# Only trigger on git commit commands
CMD=$(echo "$INPUT" | jq -r '.tool_input.command // empty')
[[ "$CMD" =~ ^git\ commit ]] || exit 0

# Get the staged diff
DIFF=$(git diff --cached 2>/dev/null)
[ -z "$DIFF" ] && exit 0

# Build review prompt
REVIEW_PROMPT="Review this git diff before commit. Respond with ONLY valid JSON:
{\"verdict\": \"approve\" or \"deny\", \"reason\": \"brief explanation\", \"issues\": [\"list of specific issues if deny\"]}

Focus ONLY on:
- Bugs, logic errors, off-by-one mistakes
- Security vulnerabilities (hardcoded secrets, injection, path traversal)
- Missing error handling for new code paths
- Obvious performance issues

Do NOT flag: style preferences, naming opinions, or minor nitpicks.

Diff:
$DIFF"

# Call headless Claude with Sonnet (60s timeout)
RESPONSE=$(timeout 60 claude -p "$REVIEW_PROMPT" --model sonnet --output-format json 2>/dev/null) || {
  echo "[commit-review-gate] Auto-approved (timeout/error) — consider manual review" >&2
  exit 0
}

VERDICT=$(echo "$RESPONSE" | jq -r '(.result // .).verdict // "approve"' 2>/dev/null) || VERDICT="approve"

if [ "$VERDICT" = "deny" ]; then
  REASON=$(echo "$RESPONSE" | jq -r '(.result // .).reason // "Review failed"' 2>/dev/null) || REASON="Review failed"
  ISSUES=$(echo "$RESPONSE" | jq -r '(.result // .).issues // [] | join("; ")' 2>/dev/null) || ISSUES=""
  echo "[commit-review-gate] Blocked commit: $REASON" >&2
  [ -n "$ISSUES" ] && echo "[commit-review-gate] Issues: $ISSUES" >&2
  exit 2
fi

exit 0
