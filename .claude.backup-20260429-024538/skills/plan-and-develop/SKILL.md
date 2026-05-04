# Plan and Develop

A structured workflow for planning and implementing features. Usable as a standalone
skill in any Claude Code session or preloaded into the planner agent.

## When to use

- Any feature or change touching 3+ files
- Unfamiliar code where you need to understand before changing
- When the user says "plan this", "design this", or "how should we implement"

## The workflow

### Phase 1: Research

Deeply explore the affected code. Not surface-level — trace imports, read function
bodies, find tests, understand data flow.

Use dynamic shell output to get live project state:

```
!`git log --oneline -10`
!`cat package.json 2>/dev/null || cat Cargo.toml 2>/dev/null || cat pyproject.toml 2>/dev/null | head -30`
!`find . -name '*.test.*' -o -name '*_test.*' -o -name 'test_*' | head -20`
```

Write findings to `docs/plans/{feature-name}/research.md`.

Key questions to answer:
- What exists today? (specific files, functions, data flow)
- What patterns does the codebase follow? (conventions, style, error handling)
- What can't change? (public APIs, shared interfaces, external contracts)
- What could break? (adjacent code, existing tests, downstream consumers)
- Is there a reference implementation? (similar feature already in the codebase)

### Phase 2: Plan

Write `docs/plans/{feature-name}/plan.md` with:

1. **Approach**: 2-3 sentence high-level strategy.
2. **Subtasks**: Each with files to touch, changes to make, tests to write, dependencies.
3. **Testing strategy**: How to verify end-to-end.
4. **Rollback**: How to safely revert.

Quality gates for the plan:
- Every subtask has disjoint file lists (parallelizable).
- Every subtask has a concrete test.
- A developer could implement any subtask with ONLY the plan + codebase.
- The plan is simpler than the first approach you thought of.

### Phase 3: Review gate

For complex tasks (6+ files, multi-subsystem):
- Have a second pass review the plan for gaps, missing edge cases, and overengineering.
- Check: are there simpler approaches? Can any subtask be eliminated?

For medium tasks (3-5 files):
- Self-review: re-read the plan after writing it. Cut anything unnecessary.

For simple tasks:
- Skip the plan. Just implement with test verification.

### Phase 4: Implement

If running within the orchestrator, hand off to implementers.
If running standalone:
1. Switch out of plan mode (Shift+Tab).
2. Implement one subtask at a time, in dependency order.
3. Run tests after each subtask. Do not proceed if tests fail.
4. Commit after each subtask.

### Phase 5: Verify

After all subtasks complete:
- Run the full test suite.
- Review the complete diff against the plan.
- Check: did we add unnecessary complexity? Can anything be removed?

## Commit the plan

```bash
mkdir -p docs/plans/{feature-name}
git add docs/plans/{feature-name}/
git commit -m "docs: plan for {feature-name}"
```

Plans are documentation. They survive the implementation. Future maintainers
(and future agents) can read them to understand why decisions were made.
