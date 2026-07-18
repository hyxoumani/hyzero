# Scheduled Review Tracker

Maintained by the scheduled review routine. Records the last commit reviewed so
subsequent runs only diff new commits. Do not hand-edit unless you also want the
next scheduled run to skip re-review of the intervening range.

last_reviewed_commit: bde4f9be00d1c59b648a4f3c8e59d63c9121d99c
last_reviewed_at: 2026-07-18

## Findings from this review

Commits in scope: 5f30ea8, 7b53e5d, a93b077, 9ddee3a, 0c35f8f, 9450e38

- src/selfplay/evaluation.rs:396-404 (a93b077) — opponent `load_weights` called
  while the `opponent_batcher` is serving. A CUDA-forward GIL release could let
  a batch straddle the state_dict swap; affects one ladder game if triggered.
  Medium confidence, low severity.
- src/selfplay/evaluation.rs:397 (a93b077) — `opp_handle.lock().unwrap()`
  panics on mutex poisoning; permanent eval-task crash. Low likelihood.
- src/selfplay/evaluation.rs:277 (a93b077) — spurious "pool empty" WARN on the
  first eval cycle after the first promotion. UX only, not a bug.
- src/mcts/tree.rs:534 (5f30ea8) — raw `*const MCTSNode` held across await.
  Currently safe under `&mut self`; fragile, no live bug.
- src/mcts/tree.rs:498 (5f30ea8) — `HYZERO_GUMBEL_TOP_K=1` collapses halving to
  a single-action serial dump. Documented degenerate mode, not a bug.

Elo math, pool enumeration, promotion gate arithmetic, env-var plumbing, and
baseline SCORE formula all check out.
