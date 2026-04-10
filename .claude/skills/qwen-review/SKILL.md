# Qwen Local Review Pipeline

Optional alternative to the Sonnet review hook. Uses a local Qwen model via Ollama
for code review. Useful for offline/airgapped environments or to avoid API token cost.

## Prerequisites

- Ollama running with `qwen2.5-coder:32b` pulled
- `jq` installed
- `curl` available

## Setup

```bash
ollama pull qwen2.5-coder:32b
```

## How it works

Two hooks replace the default Sonnet review hook:

1. **PreToolUse gate**: Every file write/edit is sent to Qwen for review. Qwen
   approves or denies with reasons. Denied changes are fed back to Claude as errors.
2. **Stop review**: When Claude finishes a task, the full git diff is sent to Qwen
   for cross-file review. Results written to `.claude/hooks/last-review.md`.

## Enabling

Copy the hook scripts and update `.claude/settings.json`:

Replace the Sonnet review hook entries with:
```json
{
  "hooks": {
    "PreToolUse": [{
      "matcher": "Write|Edit|MultiEdit",
      "hooks": [{
        "type": "command",
        "command": "bash .claude/hooks/qwen-review-gate.sh",
        "timeout": 65000
      }]
    }],
    "Stop": [{
      "hooks": [{
        "type": "command",
        "command": "bash .claude/hooks/qwen-stop-review.sh",
        "timeout": 95000
      }]
    }]
  }
}
```

## Hook scripts

The hook scripts (`qwen-review-gate.sh` and `qwen-stop-review.sh`) should be placed
in `.claude/hooks/` and made executable. They read JSON from stdin (Claude's hook
input), extract the proposed changes, send them to Ollama's API, and parse Qwen's
structured JSON response.

Key behaviors:
- Auto-approve on timeout (60s) so Ollama slowness doesn't block work
- Skip non-code files (lockfiles, images, SVGs)
- Only deny for real bugs and security issues, not style nitpicks
- All decisions logged to `.claude/hooks/review-log.jsonl`

## Coexistence

The Qwen pipeline and the reviewer agent serve different purposes:
- **Qwen hook**: Per-edit, line-level, catches bugs in real-time
- **Reviewer agent**: Post-implementation, system-level, checks architecture

Both can run simultaneously. The Qwen hook catches issues the reviewer would otherwise
flag, reducing review iterations.

## Disabling

Remove the Qwen hook entries from `.claude/settings.json` and re-enable the Sonnet
review hook entries. No other changes needed.
