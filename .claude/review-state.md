# Review State

Tracks what code has been bug-reviewed by the scheduled review routine.
Each entry records the commit reviewed, the scope, and a one-line summary
of findings. Future runs diff from `last_reviewed_commit` forward.

## Last reviewed

- **commit:** `bde4f9b` (branch `claude/modest-rubin-xf4tqq`, baseline `main` at `df794b3`)
- **date:** 2026-06-17
- **scope:** elo-promotion feature, range `7b53e5d..df794b3` on `main`
- **files:** `src/selfplay/elo.rs`, `src/selfplay/pool.rs`, `src/selfplay/evaluation.rs`, `src/bin/selfplay.rs`, `scripts/run_baseline.sh`

## Findings summary

3 high, 4 medium, 2 low. See full notification body or this commit's message for the bug list.

### HIGH

1. `src/selfplay/evaluation.rs:526` — cooldown gate is a no-op (default 0 always passes; with cooldown>0 it goes true forever after cycle 1 because counter is monotonic across non-promoting cycles).
2. `src/selfplay/evaluation.rs:264,446` — `candidate_elo` resets to opponent_initial_elo (1500) every cycle, so the promotion gate tests single-cycle gain rather than cumulative ladder progress.
3. `src/selfplay/evaluation.rs:539` — poisoned-mutex `.ok()` silently skips champion path; `promote()` may run without archiving `best_v{NNN}.pt`, breaking next cycle's pool enumeration.

### MED

4. `src/selfplay/evaluation.rs:354-375` — pool nonempty but `opp_handle` unset causes early `continue` that bypasses bootstrap; blocks ALL future promotions in misconfigured runs.
5. `src/bin/selfplay.rs:351-396` — opponent inference server has no shutdown path; leaks CUDA model on eval-task error.
6. `src/selfplay/evaluation.rs:170` — outcome bucketing thresholds at ±0.5 silently treat NaN as draw; truncated games (outcome 0.4) misclassified.
7. `src/selfplay/pool.rs:34-43` — glob entries silently dropped on parse failure; `best_v7.pt`/`best_v007.pt` collisions not logged.

### LOW

8. Env-var parse failures swallowed by `unwrap_or(default)` — invalid input silently uses default; negative K-factor accepted without validation, reversing sign of update.
9. Tests cover happy paths only — no coverage for cooldown gate, cross-cycle elo persistence, opponent-handle-unset path, or `pool.rs` version collisions; integration test is `#[ignore]`'d.

## Open design questions

- Per-cycle `candidate_elo` reset (finding #2): is this intentional? Wiki docs (`docs/wiki/elo-ladder-eval.md`, `docs/wiki/champion-pool-promotion.md`) imply a ladder-cumulative rating, but the code implements per-cycle reset. Worth confirming with the author.

## How this file is used

Each review-routine run reads `last_reviewed_commit` and diffs from there to current HEAD. If the range is empty or contains only docs/logs, the run exits silently. On new findings, the run notifies the user and updates this file.
