# Champion Pool & Promotion

The champion subsystem holds the live best model, keeps a bounded on-disk
history of past champions, and gates promotion of each new training checkpoint.
Pool membership is not a registry — it is a filesystem scan of
`checkpoints/best_v{NNN}.pt`, newest-k. As of 2026-07-09 the pool runs on the
110-plane categorical-value architecture; the legacy 102-plane champions are no
longer seeded. Code: `src/selfplay/champion.rs` (store + persistence),
`src/selfplay/pool.rs` (archive scan), `src/selfplay/evaluation.rs`
(promotion gate), `src/bin/selfplay.rs` (boot-time recovery).

## Key decisions

- **Pool = filesystem scan, no registry.** `pool.rs latest_archive_versions`
  scans `checkpoints/best_v{NNN}.pt` (strip `best_v`, strip `.pt`, parse `u64`),
  excludes the live champion's own version, sorts newest-first, returns top `k`
  (`pool_size`, default 3). Empty vec on a missing/unreadable dir — never panics.
- **Champion version = max archive filename.** `find_latest_archive_version()`
  in `selfplay.rs` picks the highest `NNN` on restart; version 1 if `best.pt`
  exists but no archive; version 0 (RandomEvaluator) if resume is missing.
- **Two-stage promotion gate** (`evaluation.rs`, ~l.472/924):
  - Empty-pool bootstrap: `win_rate ≥ HYZERO_PROMOTION_THRESHOLD` (0.55). Only
    path that can fire the FIRST promotion; single-shot — once any `best_v` lands
    every later cycle routes through Elo.
  - Pool path: `candidate_elo > opponent_initial_elo + HYZERO_PROMOTION_ELO_DELTA`
    (20.0). Both gates also require `cooldown_ok` (`promotion_cooldown_games`).
- **Legacy 102-plane re-seed is OFF by default.** `run_baseline.sh` only copies
  `backup_champion_v3806/v3905` into the pool when `HYZERO_LEGACY_POOL_SEED=1`.
  Those nets use the legacy scalar value head; seeding them made every pool
  member fail to load (POOL_DEAD) and starved the ladder. Skipped-by-default plus
  the startup cleanup lets a fresh-arch candidate found the pool via bootstrap.
- **First 110-plane pool founded 2026-07-09** — `best_v29965` at 0.562 bootstrap
  win-rate; 3 promotions that run, the first genuine promotions since 2026-06-10.

## Checkpoint persistence

`persist_champion_checkpoint` writes two files into `checkpoints/`:

- **`best.pt`** — live champion, written atomically: copy `src` → `best.pt.tmp`,
  `sync_all()`, then `rename` (the atomic publish; never seen half-written).
- **`best_v{NNN}.pt`** — archive copy, `NNN` zero-padded to min width 3
  (`format!("best_v{:03}.pt", version)`). Versions ≥1000 keep their full digits.

On `promote`, the store swaps the evaluator under a write lock, stores the new
version (Release), persists the checkpoint, and **prunes** oldest archives while
`archive_files.len() > archive_depth` (default 5), logging `[champion] pruned`.

## Gotchas

- **`run_baseline.sh` startup DELETES all `model_v*.pt` and all `best*.pt`**
  except the resume-from file. Champion continuity across baseline runs is NOT
  automatic — resume rides exclusively on `run_iter_guarded.sh`'s snapshot.
- **All-members-fail-load used to be a SILENT 0-game cycle.** A 102-vs-110 shape
  mismatch went unnoticed for ~a month. Now surfaced loudly as
  `[eval] ERROR: POOL_DEAD` (evaluation.rs ~l.871) — pool had members, 0 loaded,
  cycle void. The gate cannot promote (win_rate 0.0) but the operator sees it.
- **Eval cycles fire per new checkpoint** — 500ms version poll
  (`poll_interval_ms`), 8 games/side (`games_per_side`). A degraded cycle
  (champion/challenger inference error mid-cycle) is read-and-cleared and skips
  the promotion decision entirely (re-arm, no garbage promotion).
- **Ladder decisive results are mostly seeded-material adjudications.** Eval-side
  adjudication is ON by default (`HYZERO_EVAL_ADJUDICATE`); promotions therefore
  measure advantage-*holding* more than *outplaying*. Conversion probes remain
  the real skill metric — see [[conversion-levers]].
- **`archive_depth` (5) ≠ `pool_size` (3).** Store keeps up to 5 archives on
  disk; the ladder uses up to 3 of them as opponents per cycle.

## Related

- [[elo-ladder-eval]] — promotion decisions, opponent pool, Elo math
- [[conversion-levers]] — the skill metric behind seeded-material ladder wins
- [[selfplay-coordinator]] — checkpoint production, version channel
- `src/selfplay/champion.rs`, `src/selfplay/pool.rs`, `src/selfplay/evaluation.rs`
