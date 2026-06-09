# Training Signal (Signal-Starvation Fix)

Training stalled not because of a single bug but because of structural *signal
starvation*: the value target was `(1−β)·same-ply rootQ + β·outcome` with β=0.1,
and with ~93% of games drawn the target became a self-referential zero-attractor.
The value head empirically degenerated to "side-to-move ≈ +1" (its POV-antisymmetry
sum `v(obs)+v(flip(obs))` drifted to ≈ +1.9, eroding from healthy mid-training
checkpoints). Self-play produced almost no decisive games (300-ply cap, no
resignation), so the [Elo ladder](elo-ladder-eval.md) could not discriminate
candidates, and MCTS lacked Q-normalization/FPU so search degenerated when Q had no
scale. Merge `702e433` attacks all of these together, every change `HYZERO_*`-gated
and additive to serde/PyO3/log contracts. See also [Neural Networks](neural-networks.md)
for the target-construction baseline this modifies.

## Key decisions

- **N-step TD targets at sample time.** Computed in
  `replay_buffer.rs::compute_td_target` (the only site with the full trajectory),
  carried on the **non-serde** `TrainingSample`, NOT `StepRecord` — bincode is
  positional so no on-disk type may gain fields. Formula in step-t POV:
  `G_t = Σ (-1)^i γ^i r_{t+i} + (-1)^m γ^m · bootstrap`, with the terminal branch
  substituting the POV-converted `game_outcome` for the `root_value` bootstrap.
- **β collapses to 0 whenever a TD target exists** (`training.rs`) so the outcome is
  not double-counted through the TD tail; β/conditional-β keep their old meaning only
  on the legacy (`td_targets[k]==None`) path.
- **Value-based resignation for training games; material adjudication eval-only.**
  Resignation is value-based by design — passive shuffling still drives `root_value`
  negative — guarding against the documented *passivity attractor* that material
  adjudication caused in self-play (see `game_task.rs` GameConfig comment). Eval
  outcomes never enter training targets, so `play_game_dual` may safely adjudicate at
  the cap.
- **MinMaxStats normalization is selection-only.** Defined in `mcts/node.rs`
  (consumed by `puct.rs::select_child_normalized`, threaded through `tree.rs`);
  normalization affects child selection only — stored `total_value`/Q feeding the TD
  bootstraps is unchanged, so the sign-convention regression tests still pass.
- **Antisymmetry regularizer is flag-gated at weight 0** by default (`trainer.py`):
  zero extra forward passes until `HYZERO_ANTISYM_LOSS_WEIGHT > 0`.

## Env knobs (defaults verified against code)

| Var | Default | Effect |
|-----|---------|--------|
| `HYZERO_TD` | on | enable n-step TD value targets |
| `HYZERO_TD_NSTEP` | 5 | TD horizon n (min 1) |
| `HYZERO_TD_GAMMA` | 0.997 | TD discount γ (clamp 0..1) |
| `HYZERO_RESIGN` | on | value-based resignation (self-play only) |
| `HYZERO_RESIGN_THRESHOLD` | −0.90 | root_value at/below which a ply counts (clamp −1..−0.5) |
| `HYZERO_RESIGN_CONSECUTIVE` | 4 | consecutive losing plies before resigning |
| `HYZERO_RESIGN_MIN_PLY` | 30 | never resign during the exploration window |
| `HYZERO_TEMP_ANNEAL` | on | linear temp anneal 1.0→0.01 vs hard step |
| `HYZERO_TEMP_ANNEAL_PLIES` | 60 | anneal span past `temperature_moves` |
| `HYZERO_MCTS_QNORM` | on | MinMaxStats Q-normalization + FPU |
| `HYZERO_FPU` | 0.25 | FPU reduction for unvisited children (clamp 0..1) |
| `HYZERO_EVAL_ADJUDICATE` | on | eval-side cap adjudication |
| `HYZERO_EVAL_ADJ_MARGIN` | 5 | material lead to adjudicate a non-checkmate eval terminal |
| `HYZERO_ANTISYM_LOSS_WEIGHT` | 0 | antisymmetry regularizer weight |
| `HYZERO_ANTISYM_PROBE_N` | 8 | samples in the per-call `[antisym]` probe (clamp 1..64) |

Set the booleans off + weight 0 to recover pre-fix runtime behavior with no code
revert (no on-disk format changed).

## Gotchas

- **Bincode is positional.** Never add serde fields to `StepRecord`/`GameTrajectory`
  mid-format — `#[serde(default)]` does NOT make them additive. TD targets live on the
  non-serde `TrainingSample` for this reason.
- **The `[eval] … ladder_match` log line is a grep contract** for
  `run_baseline.sh` (`win_rate=`, `candidate_elo=`). The fix changes only the numeric
  values of existing fields, not the format.
- **`baseline_score.json`'s `last_win_rate` actually holds `DECISIVE_RATIO`** — a
  pre-existing mislabel in `run_baseline.sh` (the field is populated from
  `$DECISIVE_RATIO`). Read the `[eval]` line for the true win rate. See
  [Baseline Scoring](baseline-scoring.md).
- **New JSON field `last_antisym_mean_sum`** is extracted from the per-call
  `[antisym] step=… mean_sum=… corr=… (N=…)` line; mean_sum trending toward 0 means
  the value head is approaching POV-antisymmetry.
- **Resignation has no calibration fraction yet.** Add a `HYZERO_RESIGN_DISABLE_FRAC`
  knob (a fraction of games that play to the end ungated) before the value head has
  learned real negatives, or early resignation can reinforce a bad value head. Not
  present in code today.
- **Env-var test helpers read per-call** and env-mutating tests serialize via a
  module `Mutex`; parallel `cargo test` can rarely flake with `PoisonError` — rerun
  the affected module in isolation.
- **Clippy note:** the merge leaves the tree clippy-clean of errors (a handful of
  pre-existing warnings remain, e.g. a loop-index lint in `training.rs`). An earlier
  brief cited 5 `erasing_op` errors in `encoding.rs`/`training.rs` — that is NOT
  present in `702e433`.

## Related

- Plan / research: `docs/plans/signal-starvation-fix/plan.md`,
  `docs/plans/signal-starvation-fix/research.md`
- Wiki: [Neural Networks](neural-networks.md), [Baseline Scoring](baseline-scoring.md),
  [Elo Ladder Evaluation](elo-ladder-eval.md), [MCTS](mcts.md),
  [Replay Subsystem](replay-subsystem.md)
- Code entry points:
  - `src/data/replay_buffer.rs::compute_td_target` (n-step TD)
  - `src/py/training.rs::assemble_batch_arrays` (TD/β composition)
  - `src/selfplay/game_task.rs` (resignation + temp anneal helpers; GameConfig
    adjudication fields)
  - `src/mcts/node.rs::MinMaxStats`, `src/mcts/puct.rs::select_child_normalized`,
    `src/mcts/tree.rs` (per-search min/max threading)
  - `src/selfplay/evaluation.rs` (eval adjudication config)
  - `python/hyzero/training/trainer.py` ([antisym] probe + regularizer)
  - `scripts/run_baseline.sh` (`last_antisym_mean_sum` extraction)
