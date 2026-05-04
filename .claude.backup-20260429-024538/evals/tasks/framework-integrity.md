---
name: framework-integrity
type: smoke
difficulty: simple
timeout: 10
---

## Task

Verify all framework files exist and are properly structured.

## Verification

- [ ] `test -f .claude/framework.json`
- [ ] `test -f .claude/PRINCIPLES.md`
- [ ] `test -f .claude/CLAUDE.md.template`
- [ ] `test -f .claude/settings.json`
- [ ] `test -d .claude/agents`
- [ ] `test -d .claude/hooks`
- [ ] `test -d .claude/rules`
- [ ] `test -d .claude/skills`
- [ ] `test -d .claude/commands`
- [ ] `cat .claude/framework.json | jq . >/dev/null 2>&1`
- [ ] `cat .claude/settings.json | jq . >/dev/null 2>&1`
- [ ] `test -x .claude/hooks/readonly-guard.sh`
- [ ] `test -x .claude/hooks/safety-net.sh`
- [ ] `test -x .claude/hooks/commit-review-gate.sh`
- [ ] `test -x .claude/hooks/auto-format.sh`
- [ ] `test -x .claude/hooks/test-gate.sh`

## Expected outcome

All framework files exist. JSON files parse. Hook scripts are executable.
