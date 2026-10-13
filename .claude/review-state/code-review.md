# Code Review State

Tracks what the scheduled "claude code review" routine has already reviewed,
so each run only inspects commits newer than the marker.

## Last reviewed

- Commit: bde4f9b (`logs: refresh baseline, eval, and self-play artifacts`)
- Reviewed at: 2026-10-13 16:19 UTC
- Routine: claude code review (trig_01MjanyHZiHiUrjHn5UE9URg)
- Scope: ~15 most recent code-touching commits — elo-ladder evaluation chain
  (7b53e5d..9450e38), MCTS Gumbel-Top-K + sequential halving (5f30ea8),
  training iters cosine LR (aff97fb) and policy entropy regularization
  (7b5dd87), trainer partial-optimizer tolerance (0ba4cd8), and MCTS/
  selfplay-touched portions of the big wip refactor (2a3e6ee).

## Findings

**No bugs found** in the reviewed scope. Candidate findings were ruled out
on verification: Elo math matches the standard formula; pool enumeration
excludes `champion_version` by design (comment at evaluation.rs:224);
candidate_elo resets per-cycle intentionally; gate uses `>` while bootstrap
uses `>=` (minor inconsistency, not a bug); the Gumbel path correctly
suppresses Dirichlet noise and respects the total-sims budget; eval still
runs PUCT (gumbel_top_k=None in play_game_dual); legal_masks is always
provided from Rust (src/py/training.rs:261,319), so illegal positions
contribute 0 to the entropy term via softmax(-inf)+nan_to_num; and
`Optimizer.load_state_dict` raises before mutating state on group-count
mismatch, so the trainer's fresh-optimizer fallback is safe.

## Open question (not a bug)

`CosineAnnealingLR.state_dict()` persists `T_max` / `eta_min`. A run
resuming from a checkpoint that already has `lr_scheduler` state will
silently use the stale `T_max` rather than the new `HYZERO_LR_SCHEDULE`
env-var value. Pre-existing trainer behavior, not introduced by any
commit in the review scope — flagging so a future run considers whether
to clear scheduler state on resume when the schedule changes.

## Next run

Review commits newer than `bde4f9b` on the default branch (and any
long-lived feature branches the user names) with the same bug-only focus.
