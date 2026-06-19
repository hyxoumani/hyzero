# Review Log

Newest entry first. One line per run: `YYYY-MM-DD — <short-sha> — <one-sentence outcome>`. A future run can read the top SHA and only review commits made after it.

- 2026-06-19 — bde4f9b — 1 bug found: docs/wiki/selfplay-coordinator.md:84 claims training gates on 200 trajectories but code gates on 200 total steps (src/py/training.rs:518). Plus a minor line-number drift in chess-engine.md (close but off by ~10 lines). bde4f9b itself only refreshes log artifacts — no code bugs.
