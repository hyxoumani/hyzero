# Last reviewed

- **Branch:** claude/modest-rubin-exyp1v
- **HEAD reviewed:** bde4f9b
- **Reviewed on:** 2026-07-03
- **Scope:** commits 5f30ea8..bde4f9b (~21 SHAs, ~15 substantive Rust/Py files)
- **Focus:** bugs (correctness, concurrency, off-by-ones, panics, resource leaks)

## Findings summary

1. `src/selfplay/evaluation.rs:543` + `src/bin/selfplay.rs:276–348` — champion promotion aliases live training weights; `champion_backend`/`champion_server` never swapped. Latent today (Elo cycles use pool opponents); bootstrap fallback at `evaluation.rs:282` would misfire.
2. `src/selfplay/evaluation.rs:793–802, 848–857` — two EvaluationTask tests inherit `checkpoints_dir="checkpoints"` from `EvaluationConfig::default()`; non-hermetic if real `best_v*.pt` exists in cwd.
3. `src/selfplay/game_task.rs:363–367, 543–544` — `moves.push` runs before `board.process_move`; on illegal-move Err → break, phantom entry in PGN/trajectory.

Pre-existing / cosmetic (noted but not new): terminal reward at 300-move cap in `game_task.rs:600–603`; unbounded append to `/tmp/hyzero_diag_probe.txt` at `python/hyzero/training/trainer.py:662`.

## How the next review should use this

Diff `<last HEAD reviewed>..HEAD` and only inspect the delta. Update this file (overwrite) at the end of each review with the new HEAD SHA and date.
