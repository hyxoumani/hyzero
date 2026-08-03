# Scheduled review log

Bug-focused reviews run by the scheduled "review changes" routine.
Each entry records the HEAD SHA reviewed and any confirmed bugs.
Future runs review commits AFTER the most recent `reviewed_head`.

## 2026-08-03 — reviewed_head: bde4f9b

Scope: aff97fb..bde4f9b (Elo ladder promotion feature + wiki restructure).
Confirmed bugs:

- src/selfplay/evaluation.rs:277-281 + src/selfplay/pool.rs:33-35 — bootstrap
  re-fires after the first promotion. `latest_archive_versions(dir, exclude=N, k)`
  excludes the just-written `best_v{N}.pt`, so the pool is empty on cycle N+1
  and win-rate bootstrap fires a second time with a misleading
  "pool empty despite champion_version=N>0" WARN. Divergence from
  docs/plans/elo-promotion/plan.md:5,189 and docs/wiki/elo-ladder-eval.md:108-109.
- src/bin/selfplay.rs:183-192 — startup NOTE claims cycle games =
  2*pool_size*games_per_side; during bootstrap it's 2\*games_per_side, so
  HYZERO_PROMOTION_COOLDOWN sizing based on the notice is off by a factor
  of pool_size for the initial cycles.
- src/selfplay/evaluation.rs:341-376 — pool non-empty but opponent handle
  unset logs a ladder_match row with 0 games; run_baseline.sh would then
  read candidate_elo=1500.0 into the composite score. Only fires if the
  binary wiring changes (with_opponent is always called today).
  Clean areas: Elo math (sign, K-factor, symmetry, black-side flip), pool
  candidate_elo updates vs pinned opp_initial, mpsc channel drain vs
  load_weights sequencing, latest_archive_versions filename parsing,
  run_baseline.sh candidate_elo extractor, EVAL_CYCLES=0 fallback path.
