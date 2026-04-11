---
name: hook-test-gate
type: smoke
difficulty: simple
timeout: 10
---

## Task

Verify the test-gate hook skips gracefully when no test command is configured.

## Verification

- [ ] `bash .claude/hooks/test-gate.sh 2>/dev/null; test $? -eq 0`

## Expected outcome

When CLAUDE.md has `_TBD_` for the test command, the hook exits 0 (skip).
