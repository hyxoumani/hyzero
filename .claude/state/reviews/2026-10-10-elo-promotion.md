# Review 2026-10-10 — Elo-promotion feature series

**Range:** `7b53e5d..924f6be` (git log --oneline for the exact 7 commits)
**HEAD at review:** `bde4f9be00d1c59b648a4f3c8e59d63c9121d99c`
**Scope:** Elo-promotion / per-opponent Elo ladder feature (`src/selfplay/elo.rs`, `src/selfplay/pool.rs`, `src/selfplay/evaluation.rs`, `src/bin/selfplay.rs`)
**Verdict:** No high-severity defects. 4 lower-severity findings.

The Elo math, archive enumeration, per-game sign conventions, Elo-gate direction, win-rate aggregation, and cooldown arithmetic all check out.

## Findings

### 1. (CONFIRMED, MEDIUM) Pool enumeration excludes champion's own archive — misleading WARN after first promotion

**Files:** `src/selfplay/pool.rs:33-35`, `src/selfplay/evaluation.rs:270-281`
**Scenario:** Fresh run promotes challenger → v1, `best_v001.pt` lands. Next cycle calls `latest_archive_versions(dir, 1, 3)`, which excludes v=1 and returns empty. Code emits `[eval] WARN: pool empty despite champion_version=1 > 0; using win-rate fallback` even though no archive was deleted.
**Why it matters:** The code comment at `evaluation.rs:272-274` and `docs/wiki/elo-ladder-eval.md:55-57` claim the Elo gate takes over "once any best_v{NNN}.pt exists" — actually it needs one archive OTHER than the champion's own, i.e. the SECOND promotion. Behavior is safe (fallback runs) but operators will chase a bogus "archive deletion" warning.

### 2. (CONFIRMED, LOW-MEDIUM) PGN game number collides across opponents within a cycle

**Files:** `src/selfplay/evaluation.rs:422-428`, `src/selfplay/evaluation.rs:460-466`
**Scenario:** With `pool_size=3, games_per_side=4` a cycle plays 24 games, but `write_pgn_game(..., game_idx+1, ...)` and `..., gps+game_idx+1, ...` restart at 1 for every opponent. Tags `Eval Cycle N Game 1`..`Game 8` appear three times each with different Black/White labels. Any PGN consumer keying on (cycle, game#) collapses or confuses games.
**Note:** Bootstrap path (lines 294-299, 320-326) is fine because it has a single opponent.

### 3. (CONFIRMED, LOW) Env-var parse failures silently revert to defaults

**File:** `src/bin/selfplay.rs:144-157`
**Scenario:** The four new env vars (`HYZERO_ELO_K_FACTOR`, `HYZERO_POOL_SIZE`, `HYZERO_PROMOTION_ELO_DELTA`, `HYZERO_OPPONENT_INITIAL_ELO`) all use the `.ok().and_then(|v| v.parse().ok()).unwrap_or(default)` pattern. A typo like `HYZERO_POOL_SIZE=five` silently reverts to default 3 with no log line.
**Why it matters:** Operator cannot distinguish "config applied" from "config ignored" — exactly the failure mode we care about for a slow self-play loop.

### 4. (PLAUSIBLE, LOW) Inconsistent mutex-poisoning handling on `opp_handle`

**File:** `src/selfplay/evaluation.rs:397` (vs. sibling at `538-541`)
**Scenario:** `opp_handle.lock().unwrap()` panics on mutex poisoning while the sibling path uses `.lock().ok().and_then(...)`. If a PyO3 panic ever poisons this mutex (e.g. a GIL/`call_method1` failure that aborts across the `Python::attach` closure), every subsequent eval cycle crashes the task rather than skipping that opponent — losing all future Elo evaluation for the run.
**Caveat:** Unlikely in practice (single owner), but inconsistent with the file's own error-handling convention.

## Checked and ruled out

- Elo update formula and sign conventions (`elo.rs:16-26`, Black-side `-game_outcome` at `evaluation.rs:328` and `468`)
- Draw handling (score=0.5)
- `latest_archive_versions` boundary cases (empty dir, non-numeric names, k>n, excludes self)
- `candidate_elo` NaN/overflow (bounded by K·game_count)
- `win_rate` division-by-zero (guarded at `evaluation.rs:499`)
- Baseline `candidate_elo` extraction (defaults to 1500.0 both per-line and in `EVAL_CYCLES=0` branch)
- Env-var test `set_var` races (serial Mutex)
- Commit 924f6be is formatting-only as documented

## Convention established by this run

- Review-state lives at `.claude/state/review-state.json`.
- Per-review findings live at `.claude/state/reviews/{YYYY-MM-DD}-{slug}.md`.
- Next scheduled run should read `last_reviewed_commit` and review `{last_reviewed_commit}..HEAD`, skipping pure-log / pure-docs / pure-formatting commits.
