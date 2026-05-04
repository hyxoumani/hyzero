---
name: plan-and-develop
description: Structured workflow for planning and implementing features. Use for changes touching 3+ files, unfamiliar code that needs investigation before changing, or when the user says "plan this" / "design this" / "how should we implement".
---

# /plan-and-develop

A four-phase workflow for medium-to-complex changes. For simple changes (1-2 files, clear), skip this and just implement.

## Phase 1: Research

Explore the affected code. Trace imports, read function bodies, find tests, understand data flow.

For large investigations, dispatch the `analyst` subagent so log/file reads stay out of the main context. For small ones, use Read/Grep directly.

Write findings to `docs/plans/{feature-name}/research.md`:

- What exists today (specific files, functions, data flow)
- What patterns the codebase follows (conventions, error handling)
- What can't change (public APIs, shared interfaces, external contracts)
- What could break (adjacent code, downstream consumers)

## Phase 2: Plan

Write `docs/plans/{feature-name}/plan.md`:

```markdown
# Plan: {feature}

## Approach
{2-3 sentence high-level strategy}

## Subtasks

### 1. {subtask}
- Files: {list}
- Changes: {specific}
- Tests: {what to add}
- Dependencies: {which subtasks first, or "none"}

## Testing strategy
{End-to-end verification approach}

## Rollback
{How to safely revert}
```

Quality gates for the plan:

- Subtasks have disjoint file lists (parallelizable).
- Each subtask has a concrete test.
- A developer could execute any subtask with only the plan + codebase.
- The plan is simpler than the first approach you thought of.

## Phase 3: Review the plan

For 6+ file changes: dispatch a second `analyst` to review the plan for gaps, missing edge cases, overengineering.

For 3-5 file changes: self-review. Re-read the plan, cut anything unnecessary.

## Phase 4: Implement

If subtasks are parallelizable: dispatch multiple `developer` Tasks in worktrees, each owning a disjoint file set.

If sequential: one `developer` at a time, in dependency order.

After all subtasks complete, invoke `/verify` on the integrated diff.

## Commit the plan

```bash
mkdir -p docs/plans/{feature-name}
git add docs/plans/{feature-name}/
git commit -m "docs: plan for {feature-name}"
```

Plans are documentation. They survive the implementation and let future readers understand the why.
