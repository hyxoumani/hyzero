# Eval

Run eval tasks against the framework to measure agent quality. Each task is a
self-contained scenario with verification criteria. Results are graded and tracked.

## Quick start

```bash
# Run all evals
claude --agent orchestrator -p "Run /eval"

# Run a specific eval
claude --agent orchestrator -p "Run /eval fix-bug-off-by-one"

# Run evals k times for pass@k measurement
claude --agent orchestrator -p "Run /eval --trials 3"
```

## How it works

### 1. Task definition

Each eval task lives in `evals/tasks/{task-name}.md`:

```markdown
---
name: fix-bug-off-by-one
type: bug-fix
difficulty: simple
timeout: 120
setup: bash evals/tasks/fix-bug-off-by-one/setup.sh
---

## Task

The function `count_items` in `src/counter.rs` returns one fewer item than
expected when the input list is empty. Fix the bug.

## Verification

- [ ] `cargo test test_count_empty` passes
- [ ] `cargo test test_count_one` still passes
- [ ] `cargo test test_count_many` still passes
- [ ] No other files modified besides `src/counter.rs`

## Expected outcome

The empty-list case should return 0, not -1. The fix is a single-line change
to the boundary condition.
```

### 2. Task types

- **bug-fix**: Given a failing test, fix the code. Graded by: test passes, minimal diff.
- **feature**: Given a spec, implement it. Graded by: tests pass, matches spec, no overengineering.
- **refactor**: Given code and a goal, restructure without changing behavior. Graded by: tests pass, complexity reduced.
- **research**: Given a question about the codebase, produce a correct context summary. Graded by: factual accuracy of claims.

### 3. Eval runner

For each task:

1. **Setup**: Create a temporary worktree. Run the task's `setup` command to scaffold
   the test project (create files, introduce the bug, etc.).
2. **Execute**: Spawn the appropriate agent(s) with the task description. Use the same
   pipeline the framework normally uses (researcher → planner → implementer → tester).
   For simple tasks, skip orchestration and run a single implementer.
3. **Grade**: Run each verification check. A check is either:
   - **Command-based**: A shell command that exits 0 on pass (e.g., `cargo test test_name`)
   - **Judge-based**: A headless Claude call that evaluates the outcome against criteria
4. **Score**: Binary pass/fail per check. Task passes if ALL checks pass.
5. **Cleanup**: Remove the worktree.

### 4. Grading

Grade outcomes, not processes. The agent may take a completely different path than
expected — that's fine if verification passes.

Two grader types:

**Deterministic (preferred)**: Shell commands that exit 0/1.
```yaml
checks:
  - command: "cargo test test_count_empty"
  - command: "git diff --name-only | wc -l | xargs test 1 -eq"
```

**Judge-based (for subjective criteria)**: Headless Claude evaluates.
```yaml
checks:
  - judge: "Is the fix a minimal change? Does it avoid unnecessary refactoring?"
  - judge: "Does the commit message accurately describe the change?"
```

### 5. Metrics

After a run, the eval produces `evals/results/{timestamp}.tsv`:

```
task	trial	status	duration_s	checks_passed	checks_total	notes
fix-bug-off-by-one	1	pass	45	4	4
feature-add-endpoint	1	fail	120	2	5	missing error handling test
refactor-extract-module	1	pass	90	3	3
```

Aggregate metrics:
- **pass@1**: Fraction of tasks passed on first try (reliability)
- **pass@3**: Fraction passed in at least 1 of 3 trials (capability)
- **mean duration**: Average time per task
- **check coverage**: Average checks_passed / checks_total

### 6. Building eval tasks from real failures

The best evals come from actual mistakes. When an agent makes a mistake during
real work:

1. Note what went wrong (the reviewer or user caught it)
2. Create a minimal reproduction as an eval task
3. Write verification checks that would catch the mistake
4. Add it to `evals/tasks/`

Start with 5-10 tasks. Grow to 20-50 as real failures accumulate.

## The eval loop

```
SETUP ──► EXECUTE ──► GRADE ──► LOG
  │                              │
  └──────── next task ◄──────────┘
```

After all tasks complete, print a summary table and write results to
`evals/results/{timestamp}.tsv`.

## Running evals after framework changes

When you modify agents, hooks, rules, or skills:

1. Run the full eval suite: `/eval --trials 3`
2. Compare pass@1 and pass@3 against the previous run
3. If any task regressed, investigate before merging

This catches regressions in agent behavior caused by prompt changes, hook
modifications, or rule updates.
