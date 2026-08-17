# Scheduled Review Log

Each entry records what a scheduled bug-focused review covered so the next run can pick up from a known baseline.

## 2026-08-17 — baseline `bde4f9be00d1c59b648a4f3c8e59d63c9121d99c`

**Scope:** ELO ladder promotion work (commits 7b53e5d..9450e38, files under `src/selfplay/`: `elo.rs`, `pool.rs`, `evaluation.rs`, plus env wiring in `src/bin/selfplay.rs`).

**Findings (confirmed):**

1. `src/selfplay/evaluation.rs:848` — `test_evaluation_task_promotes_when_threshold_zero` uses `EvaluationConfig::default()` whose `checkpoints_dir` is the CWD-relative `"checkpoints"`. If any `best_v*.pt` file exists in `./checkpoints/` (created by a prior `cargo run --bin selfplay`), the test hits the pool branch, falls through the `(None, None)` guard, warns, and continues without playing games. `candidate_elo` stays at the 1500.0 default, the Elo gate fails, and the final promotion assertion on line 885 fails. Reproducer: run the selfplay binary once, then `cargo test --lib` — this test flips from pass to fail without any code change.

**Non-findings (looked hard, dismissed):** dead `champion_backend` field (memory-only, no output corruption); duplicate PGN "Cycle X Game Y" headers when the pool has >1 opponent (cosmetic).

**Next-run baseline:** `bde4f9be00d1c59b648a4f3c8e59d63c9121d99c` — future runs review commits after this SHA.
