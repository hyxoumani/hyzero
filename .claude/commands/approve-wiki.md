---
description: Authorize the next write to docs/wiki/. One-shot — flag is cleared after one successful write.
---

# /approve-wiki

User-invoked command. Sets the wiki-write authorization flag at `.claude/state/wiki-approved`. The very next write to any path under `docs/wiki/` will be permitted by `wiki-gate.sh`; subsequent writes require fresh approval.

## Action

```bash
mkdir -p .claude/state
touch .claude/state/wiki-approved
echo "[approve-wiki] Flag set. Next write to docs/wiki/ will be permitted."
```

## When to use

After `/verify` returns APPROVE and the orchestrator presents a summary asking whether to save findings to the wiki, type `/approve-wiki` to authorize the write. The orchestrator will then dispatch the analyst to author/update `docs/wiki/{topic}.md`.

## Why the flag is one-shot

The `wiki-gate.sh` PostToolUse hook clears the flag after a successful wiki write. Each wiki update requires its own user approval — no "approved once, free forever."
