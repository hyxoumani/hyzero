---
name: hook-safety-net
type: smoke
difficulty: simple
timeout: 10
---

## Task

Verify the safety-net hook blocks destructive commands and allows safe ones.

## Verification

- [ ] `echo '{"tool_input":{"command":"rm -rf /"}}' | bash .claude/hooks/safety-net.sh 2>/dev/null; test $? -eq 2`
- [ ] `echo '{"tool_input":{"command":"git push --force origin main"}}' | bash .claude/hooks/safety-net.sh 2>/dev/null; test $? -eq 2`
- [ ] `echo '{"tool_input":{"command":"git reset --hard HEAD~1"}}' | bash .claude/hooks/safety-net.sh 2>/dev/null; test $? -eq 2`
- [ ] `echo '{"tool_input":{"command":"ls -la"}}' | bash .claude/hooks/safety-net.sh 2>/dev/null; test $? -eq 0`
- [ ] `echo '{"tool_input":{"command":"cargo test"}}' | bash .claude/hooks/safety-net.sh 2>/dev/null; test $? -eq 0`
- [ ] `echo '{"tool_input":{"command":"git push origin feature-branch"}}' | bash .claude/hooks/safety-net.sh 2>/dev/null; test $? -eq 0`

## Expected outcome

Destructive patterns are blocked (exit 2). Normal commands pass through (exit 0).
