# Autoresearch Program — apr13

## Instructions

Optimize the hyzero training score (higher is better) by implementing improvements
from the development roadmap (`docs/plans/next-steps/roadmap.md`).

### Score Formula
```
score = (8.55 - final_policy_loss) + (decisive_ratio * 10) - (avg_game_length / 100)
```

### Current Score: 4.78
- Policy loss: 3.23 (lever: each 0.1 drop = +0.1 score)
- Decisive ratio: 0.20 (lever: each 0.1 increase = +1.0 score — BIGGEST LEVER)
- Avg game length: ~254 (lever: each 10 shorter = +0.1 score)

### Priority Queue (ordered by expected impact / complexity)

**Quick wins (Python-only or config changes):**
1. Loss weight rebalancing — scale value/reward loss 10x (currently 99.9% policy)
2. LR scheduling — cosine decay with 100-step warmup
3. Training-to-self-play ratio — increase from 4 to 8-16 gradient steps per game

**Medium (Rust changes, single-file):**
4. Temperature schedule — smooth exponential decay instead of hard cutoff
5. Legal move masking in MCTS — renormalize policy over legal moves at root

**Complex (multi-file Rust+Python, shape-breaking):**
6. Legal move masking in network — mask logits before softmax in inference server
7. History planes — expand observation from 19→103 planes
8. Underpromotion — expand action space from 4096→4672

### Strategy
- Start with quick wins that don't break interfaces
- Each experiment must be self-contained and independently testable
- Shape-breaking changes (history planes, underpromotion) should be bundled
  to minimize retraining cycles
- If a change improves the score, keep it and compound with the next change

## Constraints

- Do NOT modify: `Cargo.lock`, `python/pyproject.toml`, `docs/wiki/`
- Do NOT change the score formula or metric extraction in `scripts/run_baseline.sh`
- All changes must pass `cargo test` (82 tests, 7 ignored is OK)
- All changes must pass `cargo clippy` with zero warnings
- Rust+Python interface contracts must stay compatible (tensor shapes, method signatures)
- Changes must be backward-compatible with existing checkpoint format UNLESS the
  change explicitly requires a new training run from scratch

## Stopping Criteria

- After 20 experiments, OR
- When score plateaus (no improvement for 5 consecutive experiments), OR
- When all priority queue items have been attempted
