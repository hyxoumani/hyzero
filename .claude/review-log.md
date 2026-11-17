# Review Log

Scheduled bug-review ledger. Each entry records the HEAD SHA reviewed up to and the bugs surfaced. The latest `reviewed_through` is the baseline for the next firing.

## 2026-11-17 — reviewed_through: bde4f9b

Scope: `aff97fb..bde4f9b` — ~15 commits adding the Elo ladder evaluator, promotion logic, archive pool enumeration, env-var wiring, baseline integration, docs, and log snapshots. First firing of the review routine; no prior baseline, so the elo-promotion feature series was used as the initial window.

### Findings (4)

1. **src/selfplay/evaluation.rs:218–336 — bootstrap path keyed on wrong condition.**
   `run()`'s doc says bootstrap is the only path that produces the FIRST promotion, but the branch key is `pool.is_empty()`, not `champion_version == 0`. If `checkpoints/best_v*.pt` files from a prior run exist before any in-run promotion, the Elo-gate path fires on cycle 1 using stale opponents, silently bypassing the win-rate bootstrap gate and `HYZERO_PROMOTION_THRESHOLD`. Not a panic — a correctness/UX gap vs. the stated plan.

2. **src/selfplay/evaluation.rs:264, 491–534 — candidate_elo reset vs. baseline score aggregation misaligned.**
   `candidate_elo` is reset to `opponent_initial_elo` at the start of every cycle, but `scripts/run_baseline.sh:218` scores off the _last_ cycle's terminal Elo. A single bad late cycle (e.g. post-promotion where `total_games` is wiped) can make the composite baseline score negative even when earlier cycles were strong.

3. **scripts/run_baseline.sh:218–219 — brittle fallback on empty awk output.**
   `LAST_CANDIDATE_ELO=$(awk … last+0)` returns literal `0` when the awk pipeline finds no match, and `${LAST_CANDIDATE_ELO:-1500.0}` does NOT substitute for a non-empty `"0"`. Today this can't trigger because `EVAL_CYCLES` is derived from the same filter, but any future rename of `candidate_elo=` in the log line would silently feed `(0 − 1500) · ELO_SCORE_WEIGHT = −75` into the baseline score.

4. **src/selfplay/evaluation.rs:395–404 — mutex held across Python checkpoint load.**
   `opp_handle.lock().unwrap()` is held across the entire `Python::attach` + `call_method1("load_weights", …)`. A prior-holder panic poisons the mutex and crashes the eval task; the std-Mutex is also held while blocking on Python inside an async task, stalling the tokio worker during multi-hundred-MB checkpoint loads. Single caller today, so no contention, but fragile under future reuse.

### Verified OK

- `elo.rs` math: signs, K=32, draw handling, no overflow — correct and unit-tested.
- `pool.rs` enumeration: missing dir, bad names, exclusion, truncation — tested.
- White/black POV sign conventions in bootstrap (line 328) and Elo (line 468) branches both negate correctly.
- `total_games` div-by-zero guarded at line 499.
- Env-var parsing: every `env::var(...).parse().ok()` falls back to defaults, no crash.
- `clone_ref` + `Mutex<Py<PyAny>>` ownership of opponent server: handle and backend reference same Py object, correct.
- Promotion cooldown counter: reset on promote, incremented per game, gate formula correct.
