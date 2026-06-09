# Champion Pool & Promotion

The champion subsystem persists the current best model, archives a bounded
history of past champions, and recovers the champion version across restarts.
Code: `src/selfplay/champion.rs` (store + persistence), `src/selfplay/pool.rs`
(archive enumeration), and `src/bin/selfplay.rs` (boot-time recovery).

## ChampionStore (`champion.rs`)

```rust
struct ChampionStore {
    champion: RwLock<Arc<dyn Evaluator>>,   // current champion evaluator
    champion_version: AtomicU64,            // 0 = Random baseline
    archive_depth: usize,                   // max best_v{NNN}.pt to keep
    archive_files: RwLock<Vec<PathBuf>>,    // tracked archives, newest last
}
```

- `new(initial, archive_depth)` starts at version 0.
- `new_with_version(initial, archive_depth, starting_version)` boots from an
  existing champion (used on restart). The binary constructs the store with
  `archive_depth = 5`.
- `version()` — atomic read, no lock.
- `champion()` — clones the current evaluator under a short read lock.

### Promotion (`promote`)

`promote(new_champion, new_version, checkpoint_src) -> u64`:
1. Take the write lock and swap the evaluator.
2. Store `new_version` (Release).
3. If a `checkpoint_src` path is given, persist it (see below) and track the new
   `checkpoints/best_v{NNN:03}.pt` archive. **Prune** the oldest while
   `archive_files.len() > archive_depth`.

## Checkpoint Persistence (`persist_champion_checkpoint`)

Writes two files into `checkpoints/`:

- **`best.pt`** — the live champion, written **atomically**: copy `src` →
  `best.pt.tmp`, `sync_all()`, then `rename(best.pt.tmp → best.pt)`. The rename is
  the atomic publish step.
- **`best_v{NNN}.pt`** — an archive copy of `best.pt`, where `NNN` is the version
  zero-padded to 3 digits (`format!("checkpoints/best_v{:03}.pt", version)`).

Archive pruning removes the oldest `best_v{NNN}.pt` files once more than
`archive_depth` (default 5) exist, logging `[champion] pruned archive: …`.

## Archive Enumeration (`pool.rs`)

`latest_archive_versions(checkpoints_dir, exclude_version, k)` scans for
`best_v{NNN}.pt`, parses the version (`strip_prefix("best_v")` →
`strip_suffix(".pt")` → `parse::<u64>()`), excludes `exclude_version` (the live
champion's own version), sorts newest-first, and returns the top `k` as
`(version, path)`. Returns an empty vec on a missing/unreadable directory (never
panics). This is what the [Elo Ladder](elo-ladder-eval.md) calls to build its
opponent pool (with `k = pool_size`, default 3).

## Champion-version Recovery on Restart

`find_latest_archive_version()` in `src/bin/selfplay.rs` scans `checkpoints/` for
`best_v{NNN}.pt` and returns the highest `NNN` (same filename parsing as
`pool.rs`). On startup, if the resume checkpoint exists
(`HYZERO_RESUME_FROM`, default `checkpoints/best.pt`):

- starting version = `find_latest_archive_version()` if any archive exists, else
  **1** (with a notice that no archive was found);
- the resume bytes are loaded into a dedicated champion `InferenceServer`, and the
  store is built with `new_with_version(champion_eval, 5, starting_version)`.

If the resume checkpoint is missing or unreadable, the champion falls back to
`RandomEvaluator` at version 0, and the ladder uses the empty-pool win-rate
bootstrap until the first promotion.

## Promotion Gate & Scoring Weights

The promotion **decision** lives in the eval task (see
[Elo Ladder](elo-ladder-eval.md)):

- Empty-pool bootstrap: `win_rate ≥ promotion_threshold`
  (`HYZERO_PROMOTION_THRESHOLD`, default 0.55).
- Pool path: `candidate_elo > opponent_initial_elo + promotion_elo_delta`
  (`HYZERO_PROMOTION_ELO_DELTA`, default 20.0).

Scoring weights (consumed by `scripts/run_baseline.sh`, not the promotion gate):

- `HYZERO_CHAMPION_SCORE_WEIGHT` (default 2.0) — per-promotion weight in the
  composite score.
- `HYZERO_ELO_SCORE_WEIGHT` (default 0.05) — multiplier on the signed Elo term
  `(last_candidate_elo − 1500)`.

See [Baseline Scoring](baseline-scoring.md) for the full formula.

## Gotchas

- **`archive_depth` (5) is separate from `pool_size` (3).** The store keeps up to
  5 archives on disk; the ladder uses up to 3 of them as opponents per cycle.
- **`best.pt` is the live champion; `best_v{NNN}.pt` are the immutable archives.**
  The atomic `.tmp` rename guarantees `best.pt` is never seen half-written.
- **Recovery defaults to version 1** when `best.pt` exists but no `best_v{NNN}.pt`
  archive does — so the ladder doesn't restart at 0 with valid weights.
- `run_baseline.sh` wipes `model_v*.pt` and `best*.pt` (except the resume-from
  file) at startup — see [Baseline Scoring](baseline-scoring.md).

## Related

- [Elo Ladder Evaluation](elo-ladder-eval.md) — promotion decisions, opponent pool
- [Baseline Scoring](baseline-scoring.md) — startup wipe, scoring weights
- `src/selfplay/champion.rs`, `src/selfplay/pool.rs`, `src/bin/selfplay.rs`
