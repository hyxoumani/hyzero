#!/bin/bash
# Agent Output Logger — PostToolUse hook for Agent
# Logs agent completion and captures full output for trace analysis.
# Exit 0 = always allow (pure logging, no blocking).

set -euo pipefail

INPUT=$(cat)
AGENT_TYPE=$(echo "$INPUT" | jq -r '.tool_input.subagent_type // "general"')
DESCRIPTION=$(echo "$INPUT" | jq -r '.tool_input.description // "unknown"')
OUTPUT=$(echo "$INPUT" | jq -r '.tool_response // "no output captured"')
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
SAFE_TS=$(echo "$TIMESTAMP" | tr ':' '-')

LOG_FILE="docs/sessions/agent-trace.log"
OUTPUT_DIR="docs/sessions/agent-outputs"
mkdir -p "$OUTPUT_DIR"

# Append completion to trace timeline
echo "$TIMESTAMP	$AGENT_TYPE	complete	$DESCRIPTION" >> "$LOG_FILE"

# Write full output to individual file
cat > "$OUTPUT_DIR/${SAFE_TS}-${AGENT_TYPE}.md" <<EOF
# ${AGENT_TYPE}: ${DESCRIPTION}
**Timestamp**: ${TIMESTAMP}

## Output

${OUTPUT}
EOF

exit 0
