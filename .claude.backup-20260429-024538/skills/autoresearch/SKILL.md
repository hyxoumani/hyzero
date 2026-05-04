# Autoresearch

Autonomous experiment loop. Make a change, measure it, keep or discard, repeat.
Runs unattended overnight. Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

Applicable to: ML training experiments, infrastructure benchmarks, code optimization,
configuration tuning, or any task with a measurable metric.

## Prerequisites

Before entering autoresearch mode, the following must be configured in CLAUDE.md:

- **Metric**: The measurement to optimize (e.g., `val_bpb`, `latency_p99_ms`, `throughput_rps`).
  Lower is better unless noted. User must specify this during bootstrap.
- **Metric command**: Shell command that outputs the metric value (e.g., `grep "^val_bpb:" run.log`).
- **Run command**: How to execute an experiment (e.g., `uv run train.py > run.log 2>&1`).
- **Time budget**: Max wall-clock time per experiment (default: 300 seconds).
- **In-scope files**: Files the implementer can modify.
- **Read-only files**: Files that must not be touched.

## Setup

1. **Agree on a run tag**: Based on today's date (e.g., `apr9`).
2. **Create a branch**: `git checkout -b autoresearch/{tag}`.
3. **Write `program.md`** with three registers:
   - **Instructions**: What to search for, what kinds of improvements to try
   - **Constraints**: What must not change (interfaces, test contracts, file boundaries)
   - **Stopping criteria**: When to wrap up and report (e.g., "after 50 experiments" or "when metric plateaus for 10 consecutive runs")
4. **Initialize results.tsv** with header row:
   ```
   commit	{metric_name}	memory_gb	status	description
   ```
5. **Run baseline**: Execute the run command as-is. Log the result.
6. **Confirm and go**.

## The experiment loop

LOOP FOREVER:

### 1. Propose

The orchestrator spawns a FRESH researcher to propose an experiment. The researcher:
- Reads the current code state
- Reads `results.tsv` for history of what worked and what didn't
- Proposes a specific, testable change with a hypothesis

### 2. Implement

The orchestrator spawns a FRESH implementer (in a worktree) to make the change.
- Only modify in-scope files.
- Git commit the change.

### 3. Run

The orchestrator spawns a FRESH tester to execute the experiment.
```bash
timeout {TIME_BUDGET + 120} {RUN_COMMAND}
```
- Redirect all output: `> run.log 2>&1`. Do NOT let output flood context.
- Extract results: `grep "^{metric}:" run.log`
- If grep is empty, the run crashed. `tail -n 50 run.log` for the traceback.

### 4. Evaluate

Compare to the current best metric value.
- **Improved** (metric better): Keep the commit. Advance the branch.
- **Equal or worse**: `git reset --hard` back to the previous commit.
- **Crashed**: Log as crash. Attempt a fix if it's trivial (typo, missing import).
  If the idea is fundamentally broken, skip it.

### 5. Log

The context-keeper appends to `results.tsv`:
```
{commit_short}	{metric_value}	{memory_gb}	{keep|discard|crash}	{description}
```

### 6. Repeat

Go back to step 1 with a fresh researcher. NEVER accumulate experiments in one context.

## Critical rules

- **NEVER STOP**: Do not pause to ask the user. They may be sleeping. Continue
  indefinitely until manually interrupted. If out of ideas, re-read the codebase,
  re-read results.tsv for patterns, try combining previous near-misses, try more
  radical changes. Think harder.
- **Fresh context per experiment**: Each agent spawned for each experiment is NEW.
  This prevents context degradation over long runs (100+ experiments overnight).
- **Timeout**: If an experiment exceeds TIME_BUDGET + 2 minutes, kill it. Log as crash.
- **VRAM/Memory**: Some increase is acceptable for meaningful metric gains. Don't blow
  it up dramatically.
- **Simplicity**: A small improvement that adds complexity is usually not worth it.
  An equal result with simpler code IS worth keeping.
- **results.tsv is NOT committed**: Leave it untracked by git. It's a local log.

## Recovery

If the branch gets into a bad state:
```bash
git log --oneline -20  # Find the last good commit
git reset --hard {commit}  # Reset to it
```
Only do this if you've exhausted fix attempts. Prefer forward progress.

## Expected throughput

With a 5-minute time budget: ~12 experiments/hour, ~100 overnight (8 hours).
With a 1-minute budget: ~40/hour for faster iteration on small changes.
