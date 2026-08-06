# Code review — elo-promotion feature series

- **Date:** 2026-08-06
- **Scope:** commits `7b53e5d..bde4f9b` on `main` (elo-ladder-based
  promotion feature + follow-up formatting/docs/log commits)
- **Focus:** correctness bugs
- **Verdict:** no confirmed bugs

## Files reviewed

- `src/selfplay/elo.rs` (new)
- `src/selfplay/eval.rs`
- `src/selfplay/archive.rs`
- `src/selfplay/inference.rs`
- `src/selfplay/coordinator.rs`
- `src/bin/selfplay.rs`

Verified `924f6be` ("cargo fmt") is genuinely whitespace-only and
introduces no logic change.

## Notes (not confirmed bugs, but worth attention)

### 1. Dead `champion_backend` wiring (pre-existing)

`EvaluationTask.champion_backend` (`src/selfplay/eval.rs:103`) and
`with_champion_backend()` are wired from `src/bin/selfplay.rs:529`, but
`EvaluationTask::run()` never reads the field. Champion swap on
promotion actually happens via `ChampionStore::promote()`, which
overwrites the `Arc<dyn Evaluator>`. The `champion_backend_handle`
threaded through `src/bin/selfplay.rs:257,318,348,359` is effectively
dead. Present before this range too — flagged because the Elo feature
keeps calling `with_champion_backend`, so the misleading plumbing is now
part of the promotion path readers will trace.

### 2. Bootstrap re-entry after first promotion → self-vs-self match

After the first Elo promotion, `champion_version = N` and only
`best_v{N:03}.pt` exists on disk. `latest_archive_versions(dir,
exclude=N, k=3)` returns empty → the next eval cycle re-enters the
bootstrap branch (with the new "pool empty despite champion_version>0"
WARN at `src/selfplay/eval.rs:279`).

Because `ChampionStore::champion()` post-promotion returns the same
`challenger_evaluator` handle (both route to the live main
InferenceServer with the latest training weights),
that cycle plays challenger-vs-challenger against identical weights.
With near-50/50 tie-break vs. threshold 0.55, spurious promotion
is unlikely but not impossible.

The self-vs-self routing is inherent to the pre-existing
`promote(new_champ = self.challenger_evaluator.clone(), ...)` pattern,
not new in this range — but the new Elo path materially widens its
exposure since the pool-empty branch is now the normal state between
the 1st and 2nd promotions rather than a startup-only edge.

Consider gating the bootstrap branch on `champion_version == 0`
explicitly, or forcing archival of `best_v001.pt` at first promotion
into the pool so the second cycle has a real opponent.

### 3. Env-var parsing silently swallows bad input

`src/bin/selfplay.rs:144-159` — new env vars fall back to defaults on
parse failure (matches the file's pattern). No bounds checks on
`HYZERO_PROMOTION_ELO_DELTA`, `HYZERO_ELO_K_FACTOR`, `HYZERO_POOL_SIZE`,
`HYZERO_OPPONENT_INITIAL_ELO`. Example failure mode: a negative
`HYZERO_PROMOTION_ELO_DELTA=-20` sets threshold = 1480 while the
candidate starts at 1500, so every cycle promotes. User-triggered
misconfiguration, not a code defect, but a `.clamp()` + WARN on the
above four would remove the footgun.
