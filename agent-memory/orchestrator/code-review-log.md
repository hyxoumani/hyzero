# Code review log

Running state for the scheduled "claude code review" routine. Each entry records
the SHA range reviewed and the findings surfaced; the newest entry's `reviewed_through`
SHA is the anchor for the next run — only commits after it are candidates.

## 2026-10-27 (cold start)

- reviewed_through: bde4f9be00d1c59b648a4f3c8e59d63c9121d99c (= main at the time)
- branch: claude/modest-rubin-nu5h2i
- scope: top-15 commits (7b53e5d..bde4f9b) — the elo-promotion feature cluster
  plus docs/logs refresh
- level: high
- findings (bugs, most-severe first):
  1. scripts/run_baseline.sh:29 — dead `${LAST_CANDIDATE_ELO:-1500.0}` fallback
     (awk prints "0" not empty); legacy-format logs get spurious -75 score penalty
  2. src/selfplay/evaluation.rs:390 — PGN game numbering collides across pool
     opponents in the same cycle (same Round tag for distinct games)
  3. src/selfplay/evaluation.rs:326 — failed `load_weights` opponent still
     contributes to `opponents=...` label and `pool_size=` in logs
  4. src/selfplay/evaluation.rs:856 — preserved
     `test_evaluation_task_promotes_when_threshold_zero` depends on cwd lacking
     a `checkpoints/` dir (new default sets `checkpoints_dir: "checkpoints"`)
  5. src/bin/selfplay.rs:170 — startup notice reports `2*K*gps` games/cycle but
     bootstrap path plays only `2*gps` until first archive exists; cooldown
     tuning underestimates cycle count by `pool_size` during bootstrap
- action: findings surfaced to user via push notification; no fixes committed.
