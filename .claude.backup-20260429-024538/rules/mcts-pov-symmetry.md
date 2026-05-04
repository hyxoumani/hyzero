---
paths:
  - "src/mcts/**/*.rs"
  - "src/selfplay/game_task.rs"
  - "src/data/encoding.rs"
---

# MCTS POV Symmetry

## Rule: Action List Ordering Must Be POV-Symmetric

When the board encoding uses current-player perspective (ranks flipped for Black-to-move), `legal_actions` is returned in absolute-square iteration order, NOT POV-symmetric order. This means the same move appears at different indices for different colors.

**MUST DO**: After calling `get_legal_moves()` and flipping action coordinates via `flip_action()`, call `legal_actions.sort_unstable()` to canonicalize the action list ordering. Both colors must then present identical `legal_actions[i]` at equivalent POV positions.

**Where**: `src/selfplay/game_task.rs` in both `play_game()` and `play_game_dual()`.

**Why**: Without sorting, bugs in selection logic (e.g., argmax tie-break to first-max) interact with the asymmetric ordering to produce color dominance. See 2026-04-19 mistakes.md entry "Color Asymmetry from legal_actions Ordering + Argmax Tie-Break."

## Rule: Tie-Breaking in Deterministic Selection Must Be Random

When selecting the best move via MCTS visit counts with `temperature ≤ ε`, ties are common early in training. **Never** use `max_by` alone — it picks the first-encountered maximum, biasing toward low indices.

**MUST DO**: Collect ALL indices where `visit_count == max_visit_count`, then pick uniformly at random from that set.

```rust
let max_visits = visits.iter().max().copied().unwrap_or(0);
let tied_indices: Vec<usize> = visits.iter().enumerate()
    .filter(|(_, &v)| v == max_visits)
    .map(|(i, _)| i)
    .collect();
let chosen_idx = *tied_indices.choose(&mut rng).unwrap();
```

**Why**: First-max determinism combined with asymmetric action ordering (Bug 2 above) produces systematic color dominance.

## Validation

Test: `test_legal_actions_ordering_is_color_symmetric_after_sort()` in `src/selfplay/game_task.rs` verifies that post-sort `legal_actions` is byte-identical between colors at the starting position.

Run before modifying any action selection or ordering code:
```bash
cargo test test_legal_actions_ordering --release -- --nocapture
```

