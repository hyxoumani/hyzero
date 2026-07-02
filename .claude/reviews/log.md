# Review Log

Tracks what the automated review routine has already examined, so subsequent runs
review only new deltas. Each entry records HEAD at review time, the diff base,
files in scope, and findings.

## 2026-07-02 — HEAD bde4f9b (elo-ladder feature)

- **Baseline SHA:** `5f30ea8`
- **HEAD SHA:** `bde4f9b`
- **Scope:** elo-ladder / elo-promotion feature
- **Files reviewed:** `src/selfplay/elo.rs`, `src/selfplay/pool.rs`, `src/selfplay/evaluation.rs`, `src/bin/selfplay.rs`, `scripts/run_baseline.sh`
- **Focus:** bugs (correctness, races, env plumbing, off-by-one, panics)
- **Verdict:** BUGS_FOUND — 3 low-severity; Elo math and Rust concurrency clean

### Findings

- **[low] `scripts/run_baseline.sh:251`** — score formula hardcodes `1500.0` as the candidate-Elo baseline; if `HYZERO_OPPONENT_INITIAL_ELO` is overridden, the composite score is silently wrong. Failure: with `HYZERO_OPPONENT_INITIAL_ELO=1200` and logged `candidate_elo=1215`, the score contribution becomes `(1215 - 1500) * 0.05 = -14.25`, i.e. a large fake regression when the challenger actually gained rating.
- **[low] `scripts/run_baseline.sh:210-217`** — the awk extraction defaults `elo="1500.0"` for lines missing `candidate_elo=`; same 1500 coupling as above. Failure: cross-version log replay where the field is absent AND `HYZERO_OPPONENT_INITIAL_ELO` was overridden → stale-baseline delta feeds into the score.
- **[low] `src/bin/selfplay.rs:148-160`** — `HYZERO_POOL_SIZE`, `HYZERO_PROMOTION_ELO_DELTA`, `HYZERO_ELO_K_FACTOR`, `HYZERO_OPPONENT_INITIAL_ELO` are parsed via `.and_then(|v| v.parse().ok())` and silently fall back to defaults on parse errors. Failure: a typo like `HYZERO_POOL_SIZE=three` or a locale-style `HYZERO_PROMOTION_ELO_DELTA=20,0` is silently discarded; the startup notice (`bin/selfplay.rs:174-192`) only fires when `HYZERO_PROMOTION_THRESHOLD` is set, so misconfiguration is invisible.

### Residual uncertainty

Cross-batcher GIL scheduling: the pool loop serialises `load_weights` between game batches within one eval cycle, but this was not runtime-verified — no observation that an in-flight opponent-inference request from a torn-down `play_game_dual` cannot still be enqueued when `load_weights` runs. Structurally it looks safe.
