#!/bin/bash
# Sonnet Review Gate — PreToolUse hook for Write|Edit|MultiEdit
# Sends proposed changes to a tool-augmented headless Sonnet instance
# that can read project files, conventions, and surrounding code.
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

# Build review prompt — Sonnet gets tools to explore the codebase itself
REVIEW_PROMPT="You are a code reviewer. A change is being made to \`$FILE_PATH\`. Here is the proposed content:

\`\`\`
$CONTENT
\`\`\`

Use your tools to understand the context:
1. Read CLAUDE.md and .claude/rules/ for project conventions
2. Read the current version of \`$FILE_PATH\` to understand what is changing
3. Grep for usages of any modified functions/types to check for breakage
4. Read related files (imports, callers) if needed

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
  # Timeout or error — auto-approve but warn
  echo "[review-gate] Auto-approved $FILE_PATH (timeout/error) — consider manual review for large diffs" >&2
  exit 0
}

# Parse verdict
VERDICT=$(echo "$RESPONSE" | jq -r '(.result // .).verdict // "approve"' 2>/dev/null) || VERDICT="approve"

if [ "$VERDICT" = "deny" ]; then
  REASON=$(echo "$RESPONSE" | jq -r '(.result // .).reason // "Review failed"' 2>/dev/null) || REASON="Review failed"
  echo "[review-gate] Denied $FILE_PATH: $REASON" >&2
  exit 2
fi

exit 0
