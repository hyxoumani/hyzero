# MCTS Action Selection Mechanics

Two critical bugs fixed in commit 41f6681 produced systemic 83% Black-dominance color bias in self-play despite symmetric rules and identical weights. The bugs interacted at the MCTS action-selection boundary.

## Bug 1: Argmax Tie-Break Non-Determinism

**Location**: `src/mcts/tree.rs:MCTSTree::select_action()` when `temperature <= EPSILON`.

**Problem**: When temperature ≈ 0 (deterministic selection), argmax finds the move with highest MCTS visit count. Ties are common early in training (uniform priors, value≈0, low visits). The `max_by` iterator picks the **first-encountered** maximum (lowest index), deterministically biasing toward `legal_actions[0]`.

```rust
// WRONG: deterministic bias to index 0 on ties
let best_idx = visits.iter().enumerate()
    .max_by(|(_, &a), (_, &b)| a.cmp(&b))
    .map(|(i, _)| i)
    .unwrap();
```

**Fix**: Collect all tied-max indices, pick uniformly at random:

```rust
let max_visits = visits.iter().max().copied().unwrap_or(0);
let tied_indices: Vec<usize> = visits.iter().enumerate()
    .filter(|(_, &v)| v == max_visits)
    .map(|(i, _)| i)
    .collect();
let chosen_idx = *tied_indices.choose(&mut rng).unwrap();
```

## Bug 2: Legal Actions Are Color-Asymmetric

**Location**: `src/selfplay/game_task.rs` in `play_game()` and `play_game_dual()`.

**Problem**: `get_legal_moves()` iterates absolute squares (0–63), preserving that iteration order in the output Vec. White's pieces (sq 0–15) have knights (sq 1, 6) before pawns (sq 8–15); Black's pieces (sq 48–63) have pawns (sq 48–55) before knights (sq 57, 62). After POV-flipping via `flip_action()`, the VALUES are correct but POSITIONS differ:

- White: `legal_actions[0..3] ≈ Knight moves`
- Black: `legal_actions[0..15] ≈ Pawn moves`

Combined with Bug 1, this creates asymmetric tie-breaks:
- White: `legal_actions[0]` = Nc3 (knight)
- Black: `legal_actions[0]` = a6 (pawn)

**Fix**: Sort `legal_actions.sort_unstable()` after POV-flipping. Both colors now present identical sorted lists at equivalent positions.

## Interaction & Impact

Neither bug alone explains the 83% color bias. Separately:
- Bug 1: Both colors tie-break identically → symmetric bias (if any)
- Bug 2: Asymmetric ordering without tie-breaks is invisible

Combined: early training with uniform MCTS priors → frequent ties → Bug 1 picks index 0 → Bug 2 makes index 0 different for each color → systematic move asymmetry → accumulated over hundreds of games → self-reinforcing policy patterns → 83% Black dominance.

## Validation

- Random evaluator at 40 sims (N=50 games): 6% W / 70% B → 40% W / 44% B (within noise)
- Test `test_legal_actions_ordering_is_color_symmetric_after_sort()` verifies sorting works
- All 130 tests pass

## Deeper Lesson: POV Invariance is Fragile

Current-player-perspective encoding requires ALL consumers of `legal_actions` to be POV-aware:
- Action indices must be symmetric (sorting required)
- Tie-breaking must be random-in-principle (not deterministic first-max)
- Visit distributions must align with sorted action list

A future MCTS refactor could centralize action ordering (canonical sort order) to make this implicit rather than requiring every callsite to remember.

## Related

- [MCTS & Self-Play](mcts-selfplay.md) — main infrastructure
- [Board Encoding](board-encoding.md) — current-player perspective, action flipping
- `src/mcts/tree.rs` — select_action() implementation
- `src/selfplay/game_task.rs` — play_game() and action sorting
- `docs/wiki/mistakes.md` — detailed entry (2026-04-19)
