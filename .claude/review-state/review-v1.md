# Bug review v1

- Reviewed commit: `bde4f9be00d1c59b648a4f3c8e59d63c9121d99c` (branch `claude/modest-rubin-kq9noo`)
- Base commit:     `51876f0` (parent of `5f30ea8 mcts: add Gumbel-Top-K…`; used as effective review scope base because HEAD == `origin/main` — no unmerged commits from `git merge-base`, so I fell back to the fork point of the Gumbel/Elo work per the task's fallback plan)
- Date: 2026-07-19
- Commit count in scope: 20 (`51876f0..HEAD`)
- Files changed in scope: 145 (135 insertions + 3 renames + related test/log churn)
- Files most closely inspected:
  - `src/mcts/gumbel.rs` (new; Gumbel-Top-K math)
  - `src/mcts/tree.rs` (Gumbel dispatch + `simulate_with_root_action` + `RootDiagnostics`)
  - `src/selfplay/elo.rs` (new; Elo math + tests)
  - `src/selfplay/pool.rs` (new; archive enumeration)
  - `src/selfplay/evaluation.rs` (per-opponent Elo ladder rewrite)
  - `src/selfplay/game_task.rs` (Gumbel env wiring, dual/single game)
  - `src/bin/selfplay.rs` (env-var plumbing + opponent InferenceServer)
  - `src/py/training.rs` (`HYZERO_TRAIN_BATCH_SIZE` override)
  - `python/hyzero/training/trainer.py` (`HYZERO_TB_ABS_PER_BATCH` override)
  - `scripts/run_baseline.sh` (candidate_elo extraction + scoring)

## Findings

**No confirmed correctness bugs introduced by this branch.**

### Suspicions investigated and dropped

Each of these was analysed in enough source to confirm the failure scenario is
NOT real (either pre-existing, cannot trigger under call sites, or code is
correct on close reading). Recording so the next review doesn't re-open them:

1. `src/mcts/tree.rs:442-443, 648` — "revisit-existing-terminal uses `child.q_value()` (parent's POV) as `value` argument (leaf's own POV) to `backpropagate`, sign-inverting the leaf value". This IS a sign asymmetry in the codebase, but it is **pre-existing** in `run_simulations_puct` at line 442-443 (untouched by this branch; base version has the identical block) and merely mirrored into the new Gumbel `simulate_with_root_action` at line 648 for behavioural consistency. Not a new bug — do NOT re-flag under "the branch introduced this".

2. `src/mcts/gumbel.rs:25-34` `sample_gumbel` upper clamp is a f32 no-op: `1.0 - 1e-9` rounds to `1.0` at f32 precision (ULP near 1.0 ≈ 5.96e-8). Verified `rand::rng().random::<f32>()` returns [0, 1) so it never actually reaches 1.0, and the closest possible input (~1.0 − 5.96e-8) yields a finite Gumbel of ~16.6, not infinity. Lower clamp (1e-9) IS effective. No bug.

3. `src/mcts/tree.rs:596` — sequential-halving `new_size = (considered.len() / 2).max(1)` jumps 3→1, skipping the 3→2 intermediate. Wastes one halving step for odd K but is not a correctness bug; number-of-rounds accounting via `num_rounds` is still consistent with the paper for power-of-2 K.

4. `src/selfplay/evaluation.rs:526-528` — `cooldown_ok` uses `>=` with `promotion_cooldown_games: usize`; the `|| == 0` disjunct is redundant (unsigned `>= 0` is always true) but harmless.

5. `src/selfplay/evaluation.rs:396-404` — `opp_handle.lock().unwrap()` while `opponent_batcher` holds a separate handle to the same `Py<PyAny>` could theoretically race `load_weights` vs `predict_batch`, but Python's GIL (`Python::attach`) serialises all Python calls in both paths, and the EvaluationTask is single-threaded per opponent (no in-flight games during load), so no race in practice.

6. `scripts/run_baseline.sh:210-219` — `LAST_CANDIDATE_ELO=${LAST_CANDIDATE_ELO:-1500.0}` fallback does NOT fire on `awk`-produced "0" (because "0" is non-empty), so a misparse would leak "0" into the score as `(0-1500)*0.05 = -75`. Verified in code: `CANDIDATE_ELO_SUMMARY` extraction uses the SAME `/\[eval\].*ladder_match/` pattern as `EVAL_CYCLES`, and the `else` branch at line 227-232 explicitly sets `LAST_CANDIDATE_ELO="1500.0"` when `EVAL_CYCLES == 0`, so the "0" fallthrough cannot actually be reached along any live control-flow path in the current script + Rust log format. Latent hazard if the log format changes, but not a bug today.

7. `src/selfplay/evaluation.rs:103, 148` — `champion_backend` field is set via `with_champion_backend()` (called from `src/bin/selfplay.rs:529`) but never read inside `run()`. Dead field, not a correctness bug.

8. `python/hyzero/training/trainer.py:498-503` — `HYZERO_TB_ABS_PER_BATCH` parse uses `.strip().isdigit() and int(_tb_abs) > 0`; verified the isdigit() gate rules out negatives, empty, and non-numeric, so `int()` cannot throw. Correct.

9. `src/selfplay/elo.rs` — hand-verified all Elo update tests numerically (16-point delta at K=32 equal ratings, sequential 5-move table, 1520-vs-1500 loss > 16). All correct.

10. `src/selfplay/pool.rs` — `latest_archive_versions` correctly parses `best_v{N}.pt`, sorts newest-first, excludes `exclude_version`, truncates to k. Test suite covers empty dir, no matches, ordering, exclusion, truncation.

11. `src/bin/selfplay.rs:257-366` — resume-checkpoint path constructs `champion_backend_handle` under multiple branches; verified all three branches (best.pt loaded / read error / no best.pt) return a valid `SwappableBackend` handle, so `.with_champion_backend(champion_backend_handle)` at line 529 always gets a live handle.
