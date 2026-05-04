---
name: hook-readonly-guard
type: smoke
difficulty: simple
timeout: 10
---

## Task

Verify the readonly-guard hook allows edits when no read-only list is configured,
and blocks edits when a read-only list is present.

## Verification

- [ ] `echo '{"tool_input":{"file_path":"src/main.rs","content":"test"}}' | bash .claude/hooks/readonly-guard.sh 2>/dev/null; test $? -eq 0`
- [ ] `echo '{"tool_input":{"file_path":"Cargo.lock","content":"test"}}' | bash .claude/hooks/readonly-guard.sh 2>/dev/null; test $? -eq 0`
- [ ] `echo '{"tool_input":{"file_path":"","content":"test"}}' | bash .claude/hooks/readonly-guard.sh 2>/dev/null; test $? -eq 0`

## Expected outcome

Without a CLAUDE.md listing read-only files, the hook allows all edits (exit 0).
After bootstrap populates the read-only list, the blocking behavior activates.
This eval tests the unconfigured (pre-bootstrap) state.
