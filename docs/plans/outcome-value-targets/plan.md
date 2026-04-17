# Plan: Outcome-Based Value Targets

## Hypothesis

The value head currently learns from `step.root_value`, which is the MCTS tree
root Q-estimate at each position. With only 50 simulations and a freshly initialized
network, these Q-estimates are near-zero noise — giving the value head almost no
signal about who wins. Replacing the value target with the actual game outcome
(±1/0), perspective-adjusted per ply, gives the value head a strong, accurate
training signal from the very first game, at no throughput cost.

## Why it beats ±1.0 noise (specific mechanism)

The score formula has two levers the value head directly controls:
- `decisive_ratio * 10` — currently 3.0 points. The value head drives move selection:
  a well-trained value head pushes MCTS toward winning continuations, increasing
  decisive play. At present the value head predicts ~0 for all positions; MCTS
  selection is entirely prior-driven and effectively random.
- `avg_game_length / 100` — currently -1.37 points. Better value estimates shorten
  games by identifying and pursuing winning sequences earlier.

Mechanism: when `target_value = game_outcome * (-1)^k`, the value head immediately
learns "this position belonged to a game that white won (+1) or black won (-1)". MCTS
Q-backed subtree selection will start steering toward positions that correspond to
game wins within ~10-20 training steps. Expected: decisive_ratio rising from 0.30
toward 0.45-0.55 (+1.5 to +2.5 score), with possible policy_loss improvement
(better value → better MCTS targets → less noisy policy distributions).

The prior experiment (Dirichlet alpha fix) improved decisive_ratio from 0.20 → 0.30
(+1.0 score component) by improving exploration diversity. That was supply-side.
This is demand-side: the value head will now know what winning looks like.

## Risk / failure mode

**Risk 1 — sign convention**: The perspective must alternate per ply. If the sign
flips once too many or too few, the value head learns inverted targets and MCTS
steers toward losing positions. Mitigation: unit test verifies sign pattern for a
known decisive trajectory.

**Risk 2 — draw collapse**: If the value head overfits to ±1 targets and suppresses
draws in its estimates, MCTS may avoid draws even when they're the best available
outcome. Observation: current decisive_ratio is only 0.30, so draw collapse is not
the dominant failure mode right now. If decisive_ratio collapses to 0.0 or avg_length
spikes past 200 on the run, this is the signature.

**Risk 3 — value head now disagrees with policy distributions**: Policy targets are
still MCTS visit distributions (which were computed with the old near-zero Q signal).
Short-term noise in policy targets during the transition. Expected to resolve within
the first 20-30 model versions as the policy network adjusts to better value estimates.

**Not a risk**: The change is confined to `training.rs:98`. No shape changes, no
Python interface changes, no checkpoint format changes. Checkpoints from `d407281`
are compatible (value weights will be retrained, but no tensor dimension changes).

## Subtasks

### 1. Add `game_outcome` to `BatchArrays` and `assemble_batch_arrays`

- **Files**: `src/py/training.rs`
- **Changes**:
  - Add `game_outcomes: Vec<f32>` field to `BatchArrays` (shape [B], one float per sample).
  - In `assemble_batch_arrays`, populate `game_outcomes[bi] = sample.game_outcome`.
  - The outcome is already present on `TrainingSample.game_outcome` — no upstream changes needed.
- **Tests**: existing `test_batch_assembly_shapes` still passes; add assertion that
  `game_outcomes.len() == b` in that test.
- **Dependencies**: none

### 2. Use outcome-based value targets in the training batch loop

- **Files**: `src/py/training.rs`
- **Changes**:
  - In `assemble_batch_arrays`, replace line 98:
    ```rust
    target_values[bi * kp1 + k] = step.root_value;
    ```
    with perspective-adjusted outcome:
    ```rust
    // Perspective: game_outcome is from white's view (+1 = white wins).
    // Position at step k within this sample window alternates perspective.
    // k=0 is the root position (side-to-move is unknown but consistent
    // within a trajectory; sign alternates each ply).
    let sign = if k % 2 == 0 { 1.0f32 } else { -1.0f32 };
    target_values[bi * kp1 + k] = sign * sample.game_outcome;
    ```
  - No changes to the `train_batch` Python call or tensor shapes — `target_values`
    is still `[B, K+1]` f32.
- **Tests**: add `test_batch_assembly_outcome_value_targets` — creates a sample with
  `game_outcome=1.0`, calls `assemble_batch_arrays`, verifies `target_values[0] == 1.0`,
  `target_values[1] == -1.0`, `target_values[2] == 1.0` (alternating).
- **Dependencies**: Subtask 1

### 3. Verify no sign-flip bug with a decisive trajectory test

- **Files**: `src/py/training.rs` (test block)
- **Changes**: Add a test that simulates a 10-step trajectory with `game_outcome=-1.0`
  (black wins) and verifies that `target_values[k]` for k=0..10 alternates
  `[-1, +1, -1, +1, ...]`. This confirms the formula handles the losing-side case.
- **Tests**: `test_outcome_value_black_wins`
- **Dependencies**: Subtask 2

## Testing Strategy

1. `cargo test` — all 82 existing tests must pass; new tests in training.rs must pass.
2. `cargo clippy` — zero warnings.
3. `bash scripts/run_baseline.sh 900` — measure score. Target: score > 8.28 (current
   6.78 + 1.5 minimum threshold). A decisive_ratio improvement from 0.30 → 0.45+
   would move the decisive component from 3.0 → 4.5+ (+1.5 alone meets threshold).

## Expected score delta

- **Conservative** (+1.5 to +2.5): decisive_ratio 0.30 → 0.45-0.55; no policy
  loss change.
- **Optimistic** (+2.5 to +4.0): decisive_ratio rises AND policy_loss drops as
  MCTS Q estimates improve (value head provides better backpropagated signal to
  PUCT selection, tightening visit distributions).
- **Failure mode** (< +1.0): value head instability causes draw collapse
  (decisive_ratio drops); revert and investigate sign convention or add
  a mixed target (0.5 * MCTS_Q + 0.5 * outcome).

## Files modified

| File | Lines changed | Scope |
|------|--------------|-------|
| `src/py/training.rs` | ~15 (field + loop body + 2 tests) | Rust only |

No Python files, no tensor shape changes, no checkpoint format changes.
