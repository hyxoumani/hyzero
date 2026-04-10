#!/bin/bash
# Sonnet Review Gate — PreToolUse hook for Write|Edit|MultiEdit
# Sends proposed changes to a headless Claude Sonnet instance for review.
# Exit 0 = approve, Exit 2 = deny (reason fed back to Claude).

set -euo pipefail

INPUT=$(cat)

# Extract file path and content from hook input
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // .tool_input.path // empty')
CONTENT=$(echo "$INPUT" | jq -r '.tool_input.content // .tool_input.new_str // empty')

# Skip non-code files
if [[ "$FILE_PATH" =~ \.(lock|svg|png|jpg|jpeg|gif|ico|woff|woff2|map)$ ]]; then
  exit 0
fi

# Skip if no content (shouldn't happen, but be safe)
[ -z "$CONTENT" ] && exit 0

# Build review prompt
REVIEW_PROMPT="Review this code change to $FILE_PATH. Respond with ONLY valid JSON:
{\"verdict\": \"approve\" or \"deny\", \"reason\": \"brief explanation\", \"issues\": [\"list of specific issues if deny\"]}

Focus ONLY on:
- Bugs, logic errors, off-by-one mistakes
- Security vulnerabilities (hardcoded secrets, injection, path traversal)
- Missing error handling
- Obvious performance issues

Do NOT flag: style preferences, naming opinions, or minor nitpicks.

Proposed content:
$CONTENT"

# Call headless Claude with Sonnet (60s timeout)
RESPONSE=$(timeout 60 claude -p "$REVIEW_PROMPT" --model sonnet --output-format json 2>/dev/null) || {
  # Timeout or error — auto-approve but warn
  echo "[review-gate] Auto-approved $FILE_PATH (timeout/error) — consider manual review for large diffs" >&2
  exit 0
}

# Parse verdict — try .result.verdict first (wrapped), then .verdict (direct)
VERDICT=$(echo "$RESPONSE" | jq -r '(.result // .).verdict // "approve"' 2>/dev/null) || VERDICT="approve"

if [ "$VERDICT" = "deny" ]; then
  REASON=$(echo "$RESPONSE" | jq -r '(.result // .).reason // "Review failed"' 2>/dev/null) || REASON="Review failed"
  echo "[review-gate] Denied $FILE_PATH: $REASON" >&2
  exit 2
fi

exit 0
