---
name: verify
description: Independent diff review by analyst against project rules, with user-gated wiki update. Use after any subagent-completed change before declaring done.
---

# /verify

You orchestrate a verification flow. You do not read source or review the diff yourself — analyst does that. You: dispatch analyst, iterate with developer until APPROVE, then present to user, then gate the wiki write.

## Step 1: Frame the diff

Use state-query Bash to know what changed. Do NOT run `git diff` (content; the analyst handles that in its own context):

```bash
git status --porcelain
git log --oneline -5
```

## Step 2: Dispatch analyst for review

Brief:

> Review the current uncommitted diff against project rules in `.claude/rules/`.
>
> Apply checklist:
> - **Correctness**: off-by-one, null handling, races. Does the change match the brief?
> - **Security**: secrets, injection vectors, unsafe deserialization, path traversal.
> - **Conventions**: load path-scoped rules matching the changed files.
> - **Scope**: did the change stay in scope, or refactor adjacent code?
> - **Tests**: new code paths covered? Edge cases?
>
> Read the diff via `git diff` and `git diff --cached`. Read full files for
> non-trivial changes. Grep for callers of modified functions.
>
> Return a single verdict: APPROVE | REQUEST_CHANGES | REJECT.
> If not APPROVE, list `file:line — issue. Fix: …` entries.

## Step 3: Iterate internally until APPROVE

The user does not see iteration. Loop in-orchestrator:

- **REJECT or REQUEST_CHANGES:** dispatch developer with a fix brief built from the listed issues. After the fix lands, return to Step 1 and re-dispatch analyst review.
- **APPROVE:** continue to user gate (Step 4).
- **Bailout:** after 3 failed iterations on the same set of issues, stop and present the failure context to the user. The user decides whether to continue, abandon, or take a different approach.

## Step 4: User gate (only after APPROVE)

Present the analyst's final summary verbatim and ask:

```
Verdict: APPROVE.
Summary: <analyst's findings>
Save to docs/wiki/{topic}.md? (yes / iterate / discard)
```

Wait for the user's response. Your turn ends here.

## Step 5: Branch on user response

**"yes":** the user must invoke `/approve-wiki` to set the gate flag. Then dispatch analyst with brief:

> Write or update `docs/wiki/{topic}.md` with current truth from the recent diff.
> Page structure (see `.claude/skills/compact/SKILL.md` Step 3): one-paragraph summary,
> `## Key decisions`, `## Gotchas`, `## Related`.
> Skip if the change was trivial (typo, dependency bump, formatting only).

The `wiki-gate.sh` hook permits the write only if the flag is set. After the write, the hook clears the flag (one-shot). Receive analyst's confirmation; report to user.

**"iterate":** dispatch developer with the user's additional brief. Return to Step 1 after the iteration completes.

**"discard":** verdict stands; no wiki update.

## Output

≤30 lines. Include:
- Verdict (APPROVE only — REJECT/REQUEST_CHANGES are handled internally)
- Files reviewed
- Wiki status (updated / skipped / discarded / not applicable)
- Iteration count if non-trivial (e.g., "1 fix cycle")
- Open questions, if any
