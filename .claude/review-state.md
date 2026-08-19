# Bug Review State

Tracked by the scheduled `Review changes & give feedback` routine.

## Last reviewed
- HEAD: `bde4f9be00d1c59b648a4f3c8e59d63c9121d99c` (`bde4f9b`)
- Date: 2026-08-19
- Scope: recent substantive commits from `2a3e6ee` (exclusive) through `bde4f9b` — elo-ladder promotion feature (`7b53e5d`, `9ddee3a`, `2a38e77`, `a93b077`, `0c35f8f`, `9450e38`) and training tweaks (`aff97fb` cosine LR, `7b5dd87` policy entropy).

## Findings surfaced to user
- `scripts/run_baseline.sh:249` — score formula hardcodes 1500.0 baseline but `HYZERO_OPPONENT_INITIAL_ELO` is overridable in Rust → silent metric drift when overridden.
- `python/hyzero/training/trainer.py` `_policy_loss` (~632) vs `_policy_loss_per_sample` (~882) — entropy regularization docstring claims all unroll steps, but TB-active k≥1 path has no entropy term; iter-2 activated the weight, so it now silently applies only at k=0 for TB samples.

## Minor / logged only
- `src/selfplay/evaluation.rs:422-435, 464-476` — PGN `[Event ...]` reused verbatim across pool members within a cycle.
- `src/selfplay/evaluation.rs:379-381` — `ladder_match` log includes an opponent even when its load fails and the loop `continue`s.

## Untested edges
- `src/selfplay/elo.rs` — `expected_score` at large rating deltas (800+) may saturate to 0/1, making `update_rating` lossy. Not confirmed a bug.

## For the next run
Only review commits reachable from HEAD but not from `bde4f9b`. If no such commits exist, stay silent (no notification).
