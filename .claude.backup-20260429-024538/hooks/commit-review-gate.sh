#!/bin/bash
# Commit Review Gate — PreToolUse hook for Bash(git commit*)
# Reviews the full staged diff using a tool-augmented Sonnet instance
# that can read project files, conventions, and surrounding code.
# Exit 0 = approve, Exit 2 = deny (Claude sees the failure and fixes).

set -euo pipefail

INPUT=$(cat)

# Only trigger on git commit commands
CMD=$(echo "$INPUT" | jq -r '.tool_input.command // empty')
[[ "$CMD" =~ ^git\ commit ]] || exit 0

# Get the staged diff
DIFF=$(git diff --cached 2>/dev/null)
[ -z "$DIFF" ] && exit 0

# Build review prompt — Sonnet gets tools to explore the codebase itself
REVIEW_PROMPT="You are a code reviewer. A commit is about to be made. Here is the staged diff:

\`\`\`diff
$DIFF
\`\`\`

Use your tools to understand the context:
1. Read CLAUDE.md and .claude/rules/ for project conventions
2. Read the full files being changed to understand surrounding code
3. Grep for usages of any modified functions/types to check for breakage
4. Check git log for recent history on heavily-changed files if relevant

Then respond with ONLY valid JSON:
{\"verdict\": \"approve\" or \"deny\", \"reason\": \"brief explanation\", \"issues\": [\"list of specific issues if deny\"]}

Focus ONLY on:
- Bugs, logic errors, off-by-one mistakes
- Security vulnerabilities (hardcoded secrets, injection, path traversal)
- Breaking changes to callers of modified functions
- Violations of project conventions found in CLAUDE.md / .claude/rules/
- Missing error handling for new code paths

Do NOT flag: style preferences, naming opinions, or minor nitpicks."

# Call headless Claude with Sonnet + tools (120s timeout for tool usage)
RESPONSE=$(timeout 120 claude -p "$REVIEW_PROMPT" \
  --model sonnet \
  --allowedTools "Read,Grep,Glob,Bash(git log*),Bash(git blame*)" \
  --output-format json 2>/dev/null) || {
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
