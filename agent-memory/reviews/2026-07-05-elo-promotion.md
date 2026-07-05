# Review 2026-07-05 — elo-promotion feature

- **HEAD reviewed:** `bde4f9be00d1c59b648a4f3c8e59d63c9121d99c`
- **Scope:** `aff97fb..HEAD` (13 commits)
- **Focus:** bugs
- **Verdict:** no high-severity correctness bugs; 6 low-to-medium findings.

## Findings (most severe first)

1. **Hardcoded `1500.0` baseline in scoring formula** — `scripts/run_baseline.sh:251`. `score += (last_candidate_elo - 1500.0) * elo_score_weight`, but `HYZERO_OPPONENT_INITIAL_ELO` is user-configurable (default 1500). Overriding it (e.g. `=1600`) silently inflates all baseline scores by `(1600-1500)*0.05 = +5`, corrupting cross-run score comparability.

2. **`Mutex::lock().unwrap()` on poisonable mutex** — `src/selfplay/evaluation.rs:397`. Inside the pool loop, `opp_handle.lock().unwrap()` panics on a poisoned lock and aborts the entire eval task, while every other error path in the loop uses `continue 'pool_loop`. One poisoned lock → eval task dies for the rest of the run; self-play continues but never promotes.

3. **PGN Event tag collision across pool opponents** — `src/selfplay/evaluation.rs:184-190`, invoked at `:422` and `:460`. `game_num = game_idx + 1` restarts at 1 for each opponent inside the pool loop, so `pool_size=3, gps=4` writes three games all tagged `Event "Eval Cycle N Game 1"`. Downstream tools grouping by (Cycle, Game) will collide.

4. **`>=` vs `>` inconsistency at the promotion gate** — `src/selfplay/evaluation.rs:530-534`. Bootstrap uses `win_rate >= promotion_threshold`; pool uses `candidate_elo > opponent_initial_elo + promotion_elo_delta`. Looks like copy-paste asymmetry rather than intent.

5. **New Elo env vars have no shell-level defaults in `run_baseline.sh`** — `scripts/run_baseline.sh:6-15`. `HYZERO_POOL_SIZE`, `HYZERO_PROMOTION_ELO_DELTA`, `HYZERO_ELO_K_FACTOR`, `HYZERO_OPPONENT_INITIAL_ELO` are not aliased and not passed on the `target/release/selfplay` invocation line (`:134-147`). Breaks the script's own pattern; users overriding them expect them in the invocation echo.

6. **Silent env-var parse failures** — `src/bin/selfplay.rs:144-159`. `HYZERO_POOL_SIZE=3.14` falls back to default 3 with no warning. Codebase-wide pattern, not new — noted for completeness.

## Coverage — hardened

Elo math direction and asymmetry (elo.rs); sign flip on Black-side (`challenger_perspective = -outcome.game_outcome` at :328 and :468 vs. counter/score updates at :302-306, :329-333, :430-439, :469-478); pool sort/exclude/truncate; `candidate_elo` reset-per-cycle vs. gate; `total_games_since_last_promotion` cooldown counter and its reset on promotion; race between `latest_checkpoint_path` writer and `promote()` reader (pre-existing, not introduced here); GIL+mutex ordering in the `load_weights` call; awk extraction of `candidate_elo=` from ladder_match lines under empty and missing-field cases. No TODO/FIXME markers in the new code.
