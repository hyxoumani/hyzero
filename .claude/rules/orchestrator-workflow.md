# Orchestrator Workflow Rules

Non-negotiable rules for session coordination and hand-offs.

## Post-task commit discipline

- After context-keeper finishes updating docs (wiki, CLAUDE.md, task status), the
  orchestrator MUST commit those changes BEFORE reporting to the user.
- Uncommitted wiki changes = knowledge loss for future sessions.
- Commit pattern: `git add docs/ && git commit -m "docs: update wiki and task status after {task name}"`
- Verify with `git status` — should show clean working tree before sign-off.

## Orchestrator memory is mandatory

- After EVERY completed task (not just sessions), write findings to
  `agent-memory/orchestrator/{topic}.md` BEFORE responding to the user.
- This is step 2 of the post-task checklist, after merging.
- Content focus: version decisions, merge conflict patterns, agent failure modes, FFI findings.
- Do NOT defer memory logging — future sessions depend on it.

## Plan review for complex tasks

- For tasks with 6+ files or 3+ parallel subtasks, spawn a second researcher to review
  the plan for gap analysis.
- Specifically check:
  - Do any subtasks share `mod.rs`, `__init__.py`, `Cargo.toml`, or other re-export files?
  - If yes: they CANNOT be parallel, even if other files are disjoint.
  - Planner often misses this.
- Example: task 27 subtasks 2+3 both needed to edit `src/py/mod.rs` — should be sequential.

## Worktree merge conflicts are expected

- Worktrees branch from older commits, not current HEAD.
- When main has moved forward, merging worktree back to main WILL conflict.
- This is normal behavior, not a sign of failure.
- Budget time for conflict resolution in the task plan.
- After merge, always run `cargo check && cargo test` to verify.

## Don't leave stale worktrees

- At session end, check `git worktree list`.
- Remove any worktrees that aren't needed: `git worktree remove .claude/worktrees/{name}`
- Delete the corresponding branch: `git branch -D claude/{branch-name}`
- Verify clean with `git status` and `git worktree list`.
