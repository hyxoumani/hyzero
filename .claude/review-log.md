# Scheduled Review Log

Track the head commit reviewed by each run of the "review changes for bugs"
scheduled routine, so the next run reviews only what's new.

## Runs

### 2026-09-05 — reviewed through `bde4f9b`

- **Range covered**: `7b53e5d..bde4f9b` (elo-ladder / elo-promotion feature)
- **Status**: CLEAN (no confirmed correctness bugs). 3 LIKELY concerns surfaced by notification:
  - `scripts/run_baseline.sh:218` — awk numeric coercion swallows malformed candidate_elo (NaN/inf) into 0; `${VAR:-1500.0}` doesn't rescue it. Score term becomes -75.
  - `src/bin/selfplay.rs:141-158` — four HYZERO\_\* env vars use `.parse().ok().unwrap_or(default)`; typos silently revert.
  - `src/bin/selfplay.rs:412-437` — opponent InferenceServer uses `.expect()`; `HYZERO_DEVICE=cuda` on CPU host or stale DEFAULT_CONFIG panics the selfplay binary at startup.
- **Clean**: elo math (`src/selfplay/elo.rs`), pool enumeration, color/sign handling, cooldown, pool-empty fallback.
- **Cosmetic notes** (not bugs): PGN `game_num` collides across pool members within a cycle; `opponents=` label lists members even if `load_weights` failed.
