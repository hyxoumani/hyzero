# Plan: Increase PUCT Exploration Constant 1.5 → 2.0

## Hypothesis

The exploration constant `c=1.5` was calibrated for AlphaZero-scale simulation counts
(hundreds to thousands). At 50 simulations per move, `sqrt(N_parent)` is small so the
exploration term `c * P * sqrt(N) / (1 + N)` is chronically under-weighted relative to
Q exploitation. Increasing c to 2.0 (+33%) should push MCTS to sample a wider variety
of moves per position, generating more diverse self-play games and reducing draw
collapse. The Dirichlet alpha fix (+2.65) demonstrated that a "wrong constant for this
simulation budget" error is the highest-yield class of bug in this codebase.

## Why it beats ±1.0 noise

- The alpha fix (+2.65) and the C=128 draw-collapse experiments both confirm that
  exploration diversity is the binding constraint, not model capacity.
- The change is a single constant, reproducible with zero randomness beyond the noise
  floor already present in the baseline.
- Mechanism is direct: more PUCT exploration → more unique moves sampled → more varied
  training positions → policy head sees richer signal.
- Expected decisive_ratio lift: 0.30 → 0.38–0.45. Even the low end (+0.08 decisive)
  contributes +0.8 to the score formula, well above the ±1.0 noise threshold.

## Risk (blast radius)

**Low.** This change touches only MCTS constants at runtime. The training pipeline
(trainer.py, py/training.rs), replay buffer, and model architecture are completely
unaffected. Failure signature: decisive_ratio falls or avg_game_length rises further
(over-exploration causing wandering play). That is an easy signal — revert if score
< 5.8 (below baseline −1.0). The only coupling risk is that both self-play AND
evaluation games use the same constant, so the eval signal changes too; this is
desirable (consistent measurement).

## Subtasks

### 1. Update PUCT exploration constant in all five call sites

- **Files**: `src/selfplay/game_task.rs`, `src/selfplay/evaluation.rs`,
  `src/bin/selfplay.rs`
- **Changes**:
  - `src/selfplay/game_task.rs:27` — `GameConfig::default()`: change `exploration_constant: 1.5` to `exploration_constant: 2.0`
  - `src/selfplay/game_task.rs:347` — test `GameConfig` literal: `exploration_constant: 1.5` → `2.0`
  - `src/selfplay/evaluation.rs:111` — `EvaluationTask` game config literal: `exploration_constant: 1.5` → `2.0`
  - `src/bin/selfplay.rs:179` — `SelfPlayConfig` construction: `exploration_constant: 1.5` → `2.0`
  - `src/mcts/tree.rs:84` — `MCTSConfig::default()`: `exploration_constant: 1.5` → `2.0` (used by unit tests only, keeps defaults consistent)
  - NOTE: `src/mcts/tree.rs:307,331,355` and `src/selfplay/game_task.rs:347` are
    test literals — update for consistency but they do not affect the binary.
- **Tests**: `cargo test` should pass unchanged; no behavior change expected in unit
  tests because they use mock evaluators with uniform priors where the Q/prior balance
  does not change decisively.
- **Dependencies**: none

## Testing Strategy

1. `cargo build --release` — verify compilation.
2. `cargo test` — confirm all 82 passing tests still pass.
3. `bash scripts/run_baseline.sh 900` — collect score.
4. Accept if score > 7.3 (baseline 6.78 + 0.5 buffer above noise floor).
5. Reject (revert) if score < 5.8. Collect two runs if first result is 6.5–7.3 (inside noise).

## Expected score delta

+0.8 to +1.5. Conservative path: decisive_ratio 0.30 → 0.38 (+0.8 from that term),
no regression in policy loss, slight game-length decrease from richer search variety.
Optimistic path: decisive_ratio → 0.45, policy loss −0.1 from better training signal,
total +1.5.

## Validation of implementability

- `exploration_constant` is a plain `f32` field on `GameConfig` and `MCTSConfig`.
- `selfplay.rs:179` hardcodes `exploration_constant: 1.5` directly in the
  `SelfPlayConfig::game_config` struct literal — it does NOT read from an env var.
  This means the change must be made in source; no runtime override path exists.
- All five occurrences grep-confirmed in `src/`. No occurrence in `python/`.
- Change is ≤6 lines across 3 files. Zero schema changes.
