# Bug-Review Log

Persistent record of which commits the scheduled bug-review routine has already
reviewed. Update by appending a new entry at the top on each run.

The `last_reviewed_sha` in the most recent entry is the baseline the next run
should diff against — only commits after it need review.

---

## 2026-07-30 — elo-promotion feature series

- **Range reviewed:** `7b5dd87..bde4f9b` (7 code commits + 3 docs/log commits)
- **last_reviewed_sha:** `bde4f9b`
- **Focus:** bugs (correctness, concurrency, panics, serialization compat)
- **Files touched:** `src/selfplay/elo.rs`, `src/selfplay/pool.rs`,
  `src/selfplay/evaluation.rs`, `src/bin/selfplay.rs`,
  `scripts/run_baseline.sh`
- **Verdict:** **No bugs found.**
- **What was checked:**
  - Elo math (`src/selfplay/elo.rs`) — expected_score formula, K-factor,
    sign convention on wins/losses/draws; cross-checked with a sequential
    table-driven test.
  - Env-var parsing (`src/bin/selfplay.rs:104-160`) — uniform
    `env::var().ok().and_then(parse).unwrap_or(default)`, no panics, no
    unwraps on missing vars.
  - Pool enumeration (`src/selfplay/pool.rs`) — excludes current champion,
    sorts newest-first before truncate, empty on missing dir.
  - Concurrency around opponent `InferenceServer` — sequential per-cycle
    `load_weights` + `play_game_dual().await` guarantees no overlap; GIL
    serializes Python.
  - Score aggregation (`scripts/run_baseline.sh:210-217`) — candidate_elo
    per `ladder_match` line, last-cycle used, signed contribution
    `(elo-1500)*weight`, bootstrap fallback 1500.0 gives zero
    contribution as intended.
  - Color balance in the pool loop — `gps` games challenger=White then
    `gps` challenger=Black per opponent; `challenger_perspective`
    correctly flipped only on Black-side games.
  - Cooldown gate (`evaluation.rs:526-528`) — `>=` comparison correct,
    `== 0` fallback avoids blocking, counter reset only on
    `promote && cooldown_ok`.
  - Serialization — `EvaluationConfig` has no serde derive and is not
    persisted; new fields runtime-only.
- **Non-bug ambiguity flagged for author awareness:** docstrings at
  `evaluation.rs:41` and `evaluation.rs:224` promise "Elo gate activates
  once `best_v001.pt` lands," but `pool.rs` unconditionally excludes
  `champion_version` from the pool, so after the first promotion the only
  archive is filtered out and bootstrap re-runs. The author added an
  explicit WARN at `evaluation.rs:277-280` for exactly this transitional
  cycle, suggesting the behavior is intentional and the docstrings are
  imprecise rather than the code being wrong.
