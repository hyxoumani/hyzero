# Review Log

Running record of the scheduled bug-focused review routine. Each entry
names the reviewed HEAD SHA, the git range inspected, and the findings.
Future runs read the most recent entry to find the baseline SHA and
review only commits newer than it.

## 2026-11-15 — reviewed up to `bde4f9b`

- **Range**: `7b53e5d^..bde4f9b` (the elo-promotion feature; 11 commits)
- **Focus**: correctness bugs
- **Verdict**: no HIGH-severity defects; feature correct for its
  documented contract

### Low-severity findings

1. `src/selfplay/evaluation.rs:234-376` — in the pool-nonempty /
   opponent-handle-missing branch, `last_evaluated_version = current` is
   set before the `continue`, so a training version is silently
   consumed with zero games played. Dormant in production because
   `src/bin/selfplay.rs:529` always attaches an opponent; latent defect
   if a caller forgets to attach.
2. `src/selfplay/evaluation.rs:422`, `:460` — PGN `game_num` restarts
   at 0 for each pool opponent, so one cycle emits colliding
   `Eval Cycle N Game 1` / `Game 2` headers across opponents. Log noise
   only; no effect on ELO or promotion.
3. `src/selfplay/evaluation.rs:264` — `candidate_elo` is reset to
   `opponent_initial_elo` at the start of every cycle, so the
   promotion gate is per-cycle, not a long-term rating. Matches the
   docstring on line 219-220; name could mislead.
4. `src/selfplay/evaluation.rs:244` — `self.cycle += 1` fires before
   the skip-ladder `continue`, so the cycle counter advances on
   skipped cycles. Cosmetic.
5. `scripts/run_baseline.sh` candidate_elo awk extractor — if the Rust
   side ever emits `nan`, awk propagates it into `SCORE`. No path to
   NaN in current code; loud, not silent, if it ever did happen.

### Confirmed correct

- `src/selfplay/pool.rs:30-39` — strict `best_v{u64}.pt` match,
  deterministic sort by parsed version, current champion correctly
  excluded.
- Opponent `load_weights` under `Python::attach` — GIL held across the
  call; `play_game_dual` ensures no stale batched requests leak across
  weight swaps.
- `promote && cooldown_ok` with `promotion_cooldown_games == 0`
  short-circuits via `|| 0 == 0`, matching the documented no-op
  default.
- `src/selfplay/elo.rs` — K-factor, `expected_score`, `update_rating`,
  draw/tie handling, and the sequential table-driven test all
  numerically correct.
