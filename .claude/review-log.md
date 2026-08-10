# Automated Bug Review Log

Bug-focused reviews of this repository, one entry per scheduled run. Each
entry pins the HEAD SHA that was reviewed so the next run picks up from
there. New entries go on top; oldest at the bottom.

Format:

- Date, HEAD SHA reviewed, diff range covered.
- Findings grouped by severity (Critical / High / Medium / Low).
- "Verified clean" areas the reviewer explicitly checked.
- SUSPECTED findings need human confirmation before action.

---

## 2026-08-10 — bde4f9be (initial baseline review)

Range reviewed: `8511b99..bde4f9be`, source-scoped
(`src/**`, `python/**`, `scripts/**`). Docs- and logs-only commits
(`06e6129`, `bde4f9b`) excluded automatically by the filter.

Scope: the elo-promotion feature — elo math module, archive pool
enumeration, opponent inference server plumbing, per-opponent ladder in
the eval task, env-var wiring, baseline `candidate_elo` extraction.

### Critical

None.

### High

None.

### Medium

- SUSPECTED — please verify. `src/selfplay/evaluation.rs` around lines
  790-810 and 848-866, tests `test_evaluation_task_completes_one_cycle`
  and `test_evaluation_task_promotes_when_threshold_zero`. Both use the
  default `checkpoints_dir = "checkpoints"` via
  `..EvaluationConfig::default()`. If the workspace already contains any
  `checkpoints/best_v*.pt` (common during dev), the archive pool becomes
  non-empty, the opponent inference handle is `None`, and the code takes
  the fallback branch at lines 341-377 and `continue`s past promotion.
  `assert_eq!(store_ref.version(), 5)` then fails. This is a test-only
  flake, not a runtime bug in the binary. The new `bootstrap_path_*`
  tests correctly use `/nonexistent/…` and are unaffected.

### Low

- `src/selfplay/evaluation.rs:422-428, 460-466`. Inside the pool loop
  `game_num` restarts at `1..=2*gps` for every opponent, so PGN entries
  in a single cycle share the same `Event = "Eval Cycle X Game N"`
  label. Games remain distinct records; challenger/pool tags in the
  player names disambiguate. Cosmetic.
- `src/selfplay/evaluation.rs:264` + `scripts/run_baseline.sh:218,251`.
  `candidate_elo` is reset to `opponent_initial_elo` at the start of
  every cycle, so `LAST_CANDIDATE_ELO` in the baseline reflects only
  the last cycle's Elo. If that cycle hits the fallback branch (Elo
  stays at 1500), the baseline score's Elo term collapses to zero even
  if prior cycles achieved strong Elo. Confirm whether the intent was
  per-cycle or cumulative.
- `src/selfplay/evaluation.rs:526-528`. The disjunct
  `total_games_since_last_promotion >= promotion_cooldown_games ||
promotion_cooldown_games == 0` has a redundant RHS: for `usize`,
  `x >= 0` is always true, so the `== 0` clause is dead logic.

### Verified clean

- Elo formulas (`elo.rs:16-26`): standard
  `1/(1+10^((r_b-r_a)/400))` and `r + K*(s - E)`; asymmetric-only update
  (candidate updates, opponents pinned) is intentional and documented;
  hand-verified for the first three sequential steps.
- Pool enumeration (`pool.rs`): deterministic newest-first sort on
  unique `u64` versions, correct exclusion of the current champion,
  silent-empty on missing dir.
- Black-side sign convention (`evaluation.rs:328, 468`):
  `challenger_perspective = -outcome.game_outcome` — consistent with
  `game_task.rs:266`.
- Env-var wiring (`bin/selfplay.rs:144-159, 504-507`): every new
  `HYZERO_*` var is parsed with fallback to default, then threaded into
  `EvaluationConfig`. Serial `elo_env_lock` in tests guards
  process-global env mutations.
- Baseline awk extraction (`run_baseline.sh:210-219`):
  `^candidate_elo=` anchored on token boundary; matches the
  space-separated `println!` format.
- Opponent inference server plumbing: single
  `Arc<Mutex<Py<PyAny>>>` cloned via `clone_ref`, `load_weights` lock
  scope tight, batcher dedicated to the eval task (no concurrent
  inference callers) and quiescent between `await`ed games.

Files opened in full: `src/selfplay/elo.rs`, `src/selfplay/pool.rs`,
`src/selfplay/evaluation.rs`, `src/bin/selfplay.rs`,
`scripts/run_baseline.sh`, `src/selfplay/champion.rs`,
`src/selfplay/mod.rs` diff, plus `game_task.rs` grep-only for the
`DualGameOutcome` sign convention. Not opened:
`python/hyzero/inference/server.py` (relied on stated
`load_weights(bytes)` contract), `src/selfplay/inference.rs` (only the
`SwappableBackend` name referenced).

Next review picks up from `bde4f9be00d1c59b648a4f3c8e59d63c9121d99c`.
