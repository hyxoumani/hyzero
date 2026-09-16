# Scheduled Bug Review Log

Tracks which commits the automated bug-review routine has reviewed, and the findings from each pass.
Each entry lists the commit range covered, the SHAs, and a compact bug list. Newest entry first.

## 2026-09-16 — elo-ladder / elo-promotion feature

**Head at review:** `bde4f9b`
**Commits covered (oldest → newest, code-affecting only):**

- `7b53e5d` selfplay: add elo math module
- `9ddee3a` selfplay: add archive pool enumeration helper
- `2a38e77` selfplay: plumb opponent inference server for elo ladder
- `a93b077` selfplay: refactor eval task to per-opponent elo ladder
- `0c35f8f` selfplay: wire elo ladder env vars + startup notices
- `9450e38` baseline: extract candidate_elo from ladder_match and add to score
- `924f6be` selfplay: apply cargo fmt to new elo-promotion code (fmt only)
- `df794b3` selfplay: fix doc-list indentation in eval task run() doc (docs only)

**Findings (5, most severe first):**

1. **`src/selfplay/evaluation.rs:530-546` + `src/py/training.rs:594-598` — bootstrap promotion can succeed with no checkpoint.** If the eval task's first `win_rate >= promotion_threshold` fires before training reaches `checkpoint_interval_steps`, `latest_checkpoint_path.lock()` yields `None`. `promote()` bumps `champion_store.version()` but `persist_champion_checkpoint` is skipped. Next cycle: `champion_version > 0` yet `pool.latest_archive_versions` is empty ("pool empty despite champion_version>0" warning), and `champion_store.champion()` returns the challenger evaluator, so the ladder plays challenger vs. its own live-weighted server. `win_rate` hovers around 0.5 and stays below the default 0.55 threshold — system can stay unpromotable for many cycles.

2. **`src/selfplay/evaluation.rs:264` + `scripts/run_baseline.sh:210-219,251` — baseline `candidate_elo` score contribution reflects only the last cycle's within-cycle swing.** `candidate_elo` is reset to `opponent_initial_elo` (1500) at the top of every cycle, but the baseline scorer uses only the final `ladder_match` line's Elo. Score contribution `(last_candidate_elo - 1500.0) * ELO_SCORE_WEIGHT` is zero throughout bootstrap (empty-pool branch never calls `update_rating`). A run that promotes late but regresses on its final cycle scores lower than an identical run without the regression, though the champion pool ends in the same state.

3. **`src/selfplay/evaluation.rs:423,461` — PGN Event tag collisions across pool members.** `game_num` restarts at 1 for every opponent, so with `pool_size=3, gps=4` one cycle produces three PGNs tagged `[Event "Eval Cycle N Game 1"]`, three tagged `Game 2`, etc. Downstream PGN consumers keying on Event (e.g. `scripts/compare_peak_vs_end_games.py`) dedupe silently or mis-attribute. Fix: monotonic cycle-wide counter, or include `opponent_version` in the Event string.

4. **`src/selfplay/evaluation.rs:526-528` — cooldown gate blocks the FIRST promotion.** `total_games_since_last_promotion` starts at 0 at binary boot but is compared to `promotion_cooldown_games` before any promotion has fired. With `HYZERO_PROMOTION_COOLDOWN=100`, `pool_size=3`, `gps=4` (24 games/cycle), the first promotion is delayed ~5 cycles even though the doc at `evaluation.rs:44` says "Minimum games between promotion decisions" — the gate should not apply until at least one promotion has fired.

5. **`src/bin/selfplay.rs:22` vs. `src/selfplay/evaluation.rs:247` — latent checkpoints_dir split-brain.** `find_latest_archive_version()` hardcodes `"checkpoints"` for seeding `champion_store_version` at startup; the eval task enumerates via `self.config.checkpoints_dir` (defaulted to `PathBuf::from("checkpoints")` at `evaluation.rs:84`). No env var wires this today, but the field is `pub`; any future consumer that changes only the config gets a silent split-brain where startup version and runtime pool come from different dirs, causing `exclude_version` to filter the wrong entry.

**Not bugs (verified clean by the analyst):** `elo.rs` formula / K-factor / draw handling; `pool.rs` numeric sort / exclusion / k-truncation; `-outcome.game_outcome` Black-perspective negation; the awk `candidate_elo=` extractor; the `env::var(...).and_then(parse).unwrap_or(default)` pattern for the four new `HYZERO_*` vars.

**Not audited this pass:** older commits `5f30ea8` (Gumbel-Top-K + sequential halving), `7b5dd87` (policy entropy regularization), `aff97fb` (cosine LR re-apply).
