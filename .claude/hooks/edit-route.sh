#!/bin/bash
# edit-route.sh — PreToolUse(Edit|Write|MultiEdit)
# Thin-orchestrator: orchestrator does not edit source.
# Subagents (agent_id present in stdin) edit freely in their own contexts.

set -euo pipefail

INPUT=$(cat)
AGENT_ID=$(echo "$INPUT" | jq -r '.agent_id // empty')

# Subagent context — allow freely
[ -n "$AGENT_ID" ] && exit 0

TOOL=$(echo "$INPUT" | jq -r '.tool_name // empty')
FILE=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

{
  echo "[edit-route] Orchestrator does not edit source. Dispatch developer (worktree-isolated)."
  echo "[edit-route] Tool: $TOOL, target: $FILE"
} >&2
exit 2
