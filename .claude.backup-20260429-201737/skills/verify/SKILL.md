---
name: verify
description: Run project tests and review the current diff against engineering principles and project rules. Use after any code change before committing, and before declaring a task done.
---

# /verify

You are verifying that a code change is ready to ship. Run the project's tests, then review the diff. Report a single verdict.

## Step 1: Run tests

Read the test command from `CLAUDE.md`. Run it. If a test fails, run it 2 more times — pass 2/3 = flaky (note, continue). 0-1/3 = real failure.

If tests fail outright, stop here and report FAIL. Don't review a broken diff.

## Step 2: Review the diff

```bash
git diff           # unstaged
git diff --cached  # staged
```

Read full files (not just the diff) for any non-trivial change. Grep for callers of modified functions to check for breakage.

Apply this checklist:

- **Correctness**: Off-by-one, null handling, race conditions. Does the change match the brief?
- **Security**: Hardcoded secrets, injection vectors, unsafe deserialization, path traversal.
- **Conventions**: Read `.claude/rules/` for path-scoped rules matching the changed files. Violations?
- **Scope**: Did the change stay in scope, or did it refactor adjacent code?
- **Tests**: New code paths covered? Edge cases?

## Step 3: Verdict

Return one of:

- **APPROVE** — tests pass, no critical issues. List any minor warnings inline.
- **REQUEST_CHANGES** — fixable issues. List specific `file:line — issue` entries. Include the fix.
- **REJECT** — fundamental problem with approach. Explain what's wrong and what to do instead.

## Output format

```
## Verification

### Tests
PASS | FAIL | FLAKY ({n}/3)
{first 5 lines of any failure output}

### Review
APPROVE | REQUEST_CHANGES | REJECT

{If REQUEST_CHANGES or REJECT:}
- {file}:{line} — {issue}. Fix: {what to do}.

{If warnings:}
Warnings (not blocking):
- {file}:{line} — {minor issue}
```

Stay under 30 lines total. Bulky test output goes to `runs/{timestamp}/verify.log` if needed.

## Step 4: On APPROVE, chain to /compact

If the verdict is APPROVE and the change is non-trivial, invoke `/compact` next with a scoped brief: the branch/files just verified and any decisions surfaced during review. The orchestrator pulls only what that change needs into context — not the full session.

Skip for trivial diffs (typos, dependency bumps, formatting-only).
