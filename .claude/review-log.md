# Scheduled Bug Review Log

Log of the scheduled bug-focused code reviews on this branch. The most recent entry's
`head_sha` is the baseline for the NEXT scheduled run — review `head_sha..HEAD` next time.

---

## 2026-10-11 — Elo-promotion feature

- head_sha: bde4f9be00d1c59b648a4f3c8e59d63c9121d99c
- scope: initial baseline review; covered commits 7b53e5d..9450e38 (elo-promotion feature arc) plus formatting/doc follow-ups
- verdict: **0 HIGH, 2 MEDIUM, 3 LOW** — code logically sound, Elo math verified

### Findings

**MEDIUM**

1. **Cooldown branch has no test coverage.** `src/selfplay/evaluation.rs:526-528`. Every test
   sets `promotion_cooldown_games: 0`, so the `total_games_since_last_promotion <
promotion_cooldown_games` path is never exercised. A future refactor inverting the
   comparison or dropping the counter increments at lines 307/334/447/486 would not fail CI.
   The elo-promotion plan explicitly called for a cooldown integration test.

2. **No non-ignored integration test for the Elo-pool branch of `run()`.** `src/selfplay/evaluation.rs:745-755,892-991`.
   `eval_task_runs_per_opponent_ladder_helper_form` only drives the pure math helper.
   The only end-to-end test touching the real `pool.iter()` loop
   (`opponent_load_weights_changes_root_setup_output`) is `#[ignore]`-gated, so a sign flip
   or missed counter increment inside the loop would not fail CI.

**LOW**

3. **`HYZERO_POOL_SIZE=0` silently falls through to bootstrap gate.** `src/selfplay/pool.rs:44` + `src/bin/selfplay.rs:148-151`.
   With `pool_size=0`, `pool.is_empty()` is always true and gating silently reverts to
   `win_rate >= promotion_threshold` instead of Elo. No startup bounds check. Same silent
   fallback for negative `HYZERO_PROMOTION_ELO_DELTA` (would allow instant promotion).

4. **Env-var parsing silently coerces invalid values to defaults.** `src/bin/selfplay.rs:144-159`.
   All four Elo-related vars use `.ok().and_then(|v| v.parse().ok()).unwrap_or(default)`,
   so `HYZERO_ELO_K_FACTOR=abc` runs with default 32.0 with no warning. Hard to notice
   in long baselines.

5. **Misleading "one cycle = {n} games" NOTE and redundant OR in cooldown check.**
   `src/bin/selfplay.rs:183-192`: formula `2 * pool_size * games_per_side` overstates games per
   cycle when fewer than `pool_size` archives exist, and is wrong for the bootstrap path
   (plays only `2 * games_per_side`). `src/selfplay/evaluation.rs:526-528`: `|| promotion_cooldown_games == 0`
   is redundant with the preceding `>=` since `usize` is non-negative — harmless but a smell.

### Verified NOT bugs

- Cumulative `candidate_elo` across pool members: intentional per the plan ("opponents pinned at
  opponent_initial_elo"); reset at each cycle (`src/selfplay/evaluation.rs:264`).
- Opponent-server concurrency: `Mutex<Py<PyAny>>` + GIL serializes individual Python calls;
  `load_weights` runs before any awaited `play_game_dual` on that opponent.
- Elo formula sign/exponent (`src/selfplay/elo.rs:16-18`): verified against standard
  `1/(1+10^((Rb-Ra)/400))`.
- `continue 'pool_loop` on read/load failure: correctly skips one opponent without corrupting counters.
- `champion_backend` field stored but unused in `run()`: pre-dates this feature (not introduced
  by elo-promotion commits).
