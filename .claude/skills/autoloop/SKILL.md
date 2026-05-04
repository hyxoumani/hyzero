---
name: autoloop
description: Autonomous feature-development loop against a user-specified baseline metric. Inspired by Karpathy's autoresearch (https://github.com/karpathy/autoresearch). Iterates propose → implement → verify → benchmark → keep/discard. Runs without asking permission until budget exhausted or user interrupts.
---

# /autoloop

Autonomous improvement loop. Each iteration tries one feature on its own branch; improvements survive, regressions are discarded. Same "don't ask permission, just iterate" principle as autoresearch — adapted to this template's thin-orchestrator workflow.

## Required parameters

The user must specify at invocation:

- **`--bench <command>`** — shell command that runs the benchmark (prints score to stdout, or writes to a known file)
- **`--extract <method>`** — how to extract the score from `--bench` output: a shell pipeline applied to the output, e.g. `grep '^Final Elo:' | awk '{print $3}'`, or `jq -r '.score' results.json`
- **`--direction <higher|lower>`** — `higher` if higher score is better, `lower` if lower
- **`--iters <N>`** — max iteration count
- **`--wall <duration>`** — max wall-clock budget (e.g., `2h`, `30m`, `8h`)

Optional:
- **`--allow <glob>`** — restrict file modifications to a glob (e.g., `src/mcts/**`)
- **`--ideas <file>`** — path to a list of feature ideas, one per line; when exhausted, fall back to analyst proposals

If any required parameter is missing, ask the user ONCE for all of them. After that, do not ask permission until the loop terminates.

## Setup

```bash
TAG=$(date +%Y%m%d-%H%M%S)
RUN_DIR="runs/auto-$TAG"
mkdir -p "$RUN_DIR"
printf 'iter\tbranch\tcommit\tscore\tstatus\tdescription\n' > "$RUN_DIR/results.tsv"
git checkout main
```

## Step 1: Establish baseline

Run `--bench` on `main`. Extract the score with `--extract`. Append the row:

```
0\tmain\t<commit-sha>\t<score>\tbaseline\tbaseline measurement
```

`<score>` becomes the current best. All future iterations compare against the current best (advances on `keep`).

## Step 2: The loop

For iteration N = 1, 2, … until any stop condition fires:

### a. Propose a feature

If `--ideas` is provided and not exhausted, take the next idea.

Otherwise dispatch `analyst`:
> Read recent rows of `<RUN_DIR>/results.tsv`. Propose one feature improvement for the metric (`<direction>`-is-better). Name the change, files to touch, expected impact. Avoid features already tried.

### b. Create branch

```bash
git checkout -b auto/iter-<N>
```

### c. Implement

Dispatch `developer`:
> Implement: <description from step a>. {If `--allow` is set: "Restrict modifications to `<glob>`."} Run project tests; paste failures. Worktree-isolated. Read-only files per `CLAUDE.md` still apply.

### d. Verify (non-interactive)

Dispatch `analyst` with the rule-review brief, but skip the user gate:
> Review the uncommitted diff against rules in `.claude/rules/`. Return APPROVE | REJECT only. No user presentation needed; the orchestrator decides.

- **REJECT** → status `discard-verify`; skip to step g.
- **APPROVE** → continue.

### e. Benchmark

Run `--bench`. Extract score with `--extract`. If the command exits non-zero or extraction fails, status = `crash`; skip to step g.

### f. Compare

Against current best (using `--direction`):
- **Improved** → status = `keep`. Update current best to this score.
- **Equal or regressed** → status = `discard`.

### g. Record + cleanup

Append to `results.tsv`:
```
<N>\tauto/iter-<N>\t<commit-sha>\t<score>\t<status>\t<description>
```

Write a per-iteration summary to `<RUN_DIR>/iter-<N>.md` (≤20 lines: idea, files changed, verify verdict, score, status).

```bash
git checkout main
```

For non-keep statuses, also:
```bash
git branch -D auto/iter-<N>
```

Kept branches survive for user review post-run.

## Step 3: Stuck handling

After 5 consecutive iterations with status ∈ {`discard`, `discard-verify`, `crash`}, dispatch `analyst`:
> The last 5 iterations did not improve the metric. Propose a more radical change — different subsystem, fundamentally different approach, or a known technique from the literature. Be specific.

Use the proposal for the next iteration. Reset the consecutive-fail counter.

## Step 4: Stop conditions

Stop when ANY of:
- Iteration count reaches `--iters`
- Elapsed wall-clock reaches `--wall`
- User interrupts (next user prompt arrives)

## Step 5: Final report

Output ≤30 lines:
- Total iterations by status (keep / discard / discard-verify / crash)
- Top 3 candidate branches by score delta vs baseline
- Best candidate's one-line description
- Paths: `<RUN_DIR>/results.tsv` and per-iteration `<RUN_DIR>/iter-N.md` files

## Constraints during the loop

- **Do not ask the user for permission** during the loop. Run until a stop condition fires.
- **Do not write to `docs/wiki/`**. Wiki writes need `/approve-wiki` from the user, which is not available in autonomous mode. Findings stay in `<RUN_DIR>/`.
- **Do not merge to main** under any circumstances. The user decides what to merge post-run.
- **One branch per iteration**. Discarded branches deleted; kept branches survive.
- **Honor read-only files** declared in `CLAUDE.md`.
- If analyst's proposal duplicates a prior iteration, dispatch again with `Avoid: <list of tried ideas>`.
- After the loop ends, if the user wants the run's findings drained to the wiki, they invoke `/compact` interactively.
