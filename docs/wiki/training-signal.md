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

That merge restored the **value** signal (validated 2026-06-09). The policy head
kept flattening regardless; the 2026-06-10 overnight loop root-caused that to
three layers of target-side noise, fixed in `58067e5`/`d8cded7`/`58beff5` — see
*Policy-target noise* below and [Run History](run-history.md) for the
run-by-run evidence (including the v3840 promotion under the fixed settings).

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

## Policy-target noise (root-caused 2026-06-10)

The policy loss is distillation toward MCTS visit-count targets; it flattened
because the targets (and one loss term) were noisy, not because the trainer
under-fit. Three layers, peeled in order across the 2026-06-10 runs:

- **The policy "entropy" term is a bonus, not a penalty.** `trainer.py
  _policy_loss` computes `loss = ce + β·Σp·log p`, which flattens the policy
  toward uniform-over-legal (its comment "penalize over-sharp output
  distributions" means exactly that). For MuZero-style distillation,
  exploration belongs to root Dirichlet noise + selection temperature — never
  to the trained policy's entropy. Default is now 0.0; any β>0 measurably
  flattened k0 (2026-06-09/10 runs at 0.01 and 0.003). The earlier advice to
  "lower" the weight could never have converged — the issue was direction, not
  magnitude.
- **Tablebase rows polluted the policy CE.** TB trajectory rows (tb_frac ≈ 0.45
  of batches) carry uniform-over-Syzygy-optimal policy targets (48% of
  positions have ≥2 optimal moves; mean legal support 19.8) and entered the
  policy CE at all k: the trajectory cache sets `is_tablebase=False`, so the
  snapshot-row masking never engaged. Gated since `d8cded7` by
  `HYZERO_TB_POLICY_WEIGHT`, which scales TB rows' policy CE *only* — TB
  value/reward supervision flows untouched at all k (that is what TB data is
  for). The diagnostic locks that found it: blended tgt_entropy ≈
  0.45·0.72 + 0.55·1.15 ≈ 0.95; k1–5 pred_entropy plateau = log(19.8) = 2.986.
- **Dirichlet noise is baked into visit targets — the persistent driver.**
  Stored visit-count targets are read *after* root noise is mixed into the
  priors (`tree.rs::extract_visit_distribution` returns raw visits, no
  de-noising). In draw-dominated play the value function cannot re-concentrate
  visits, so target entropy floors at ~2.0 nats at ε=0.25 and the policy
  faithfully distills the floor. ε=0.10 (run 4) lowered inferred replay-target
  entropy to ~1.6 and held the policy at ~2.06 / top1 0.36 over a full run, vs
  2.55 / 0.26 collapsed. Knobs since `58beff5`: `HYZERO_DIRICHLET_EPS` and
  `HYZERO_DIRICHLET_ALPHA` — the eps var was **renamed** from
  `HYZERO_DIRICHLET_EPSILON`, which nothing ever set.

Diagnosis metrics added along the way (`58067e5`, `d8cded7`):
`pred_entropy_legal` (k0, legal-masked — the honest policy-sharpness number;
the raw `pred_entropy` includes gradient-orphaned illegal logits and overstates
collapse) and `pred_entropy_legal_replay` / `pred_top1_replay` (non-TB rows
only — immune to the 45% TB blend). Rule of thumb: before touching loss
weights, compare target-side entropy against these — if the targets are flat,
the problem is search/self-play signal, not the trainer.

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
| `HYZERO_RESIGN_DISABLE_FRAC` | 0.1 | fraction of games that play ungated to the end (calibration; clamp 0..1) |
| `HYZERO_TEMP_ANNEAL` | on | linear temp anneal 1.0→0.01 vs hard step |
| `HYZERO_TEMP_ANNEAL_PLIES` | 60 | anneal span past `temperature_moves` |
| `HYZERO_MCTS_QNORM` | on | MinMaxStats Q-normalization + FPU |
| `HYZERO_FPU` | 0.25 | FPU reduction for unvisited children (clamp 0..1) |
| `HYZERO_DIRICHLET_EPS` | 0.25 | root Dirichlet noise fraction (baseline script: 0.10; renamed from `HYZERO_DIRICHLET_EPSILON`) |
| `HYZERO_DIRICHLET_ALPHA` | 0.3 | root Dirichlet concentration α |
| `HYZERO_EVAL_ADJUDICATE` | on | eval-side cap adjudication |
| `HYZERO_EVAL_ADJ_MARGIN` | 5 | material lead to adjudicate a non-checkmate eval terminal |
| `HYZERO_POLICY_ENTROPY_WEIGHT` | 0.0 | policy entropy *bonus* — any β>0 flattens; keep at 0 |
| `HYZERO_TB_POLICY_WEIGHT` | 1.0 | TB rows' policy-CE scale (baseline script: 0.0; TB value/reward unaffected) |
| `HYZERO_ANTISYM_LOSS_WEIGHT` | 0 | antisymmetry regularizer weight |
| `HYZERO_ANTISYM_PROBE_N` | 8 | samples in the per-call `[antisym]` probe (clamp 1..64) |

Set the booleans off + weights 0 to recover pre-fix runtime behavior with no code
revert (no on-disk format changed). The 2026-06-10 knobs default to *legacy*
behavior in code (`HYZERO_TB_POLICY_WEIGHT=1.0`, ε=0.25); `run_baseline.sh`
overrides both (0.0 and 0.10).

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
- **Resign calibration:** `HYZERO_RESIGN_DISABLE_FRAC` (default 0.1) plays that
  fraction of games ungated to the end, guarding against early resignation
  reinforcing a bad value head; 2026-06-09/10 calibration probes logged 0 false
  positives.
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
- Wiki: [Run History](run-history.md) (2026-06-10 policy-flattening runs;
  2026-06-09 validation), [Neural Networks](neural-networks.md),
  [Baseline Scoring](baseline-scoring.md),
  [Elo Ladder Evaluation](elo-ladder-eval.md), [MCTS](mcts.md),
  [Replay Subsystem](replay-subsystem.md)
- Code entry points:
  - `src/data/replay_buffer.rs::compute_td_target` (n-step TD)
  - `src/py/training.rs::assemble_batch_arrays` (TD/β composition)
  - `src/selfplay/game_task.rs` (resignation + calibration fraction + temp anneal
    helpers; GameConfig adjudication fields)
  - `src/mcts/node.rs::MinMaxStats`, `src/mcts/puct.rs::select_child_normalized`,
    `src/mcts/tree.rs` (per-search min/max threading; Dirichlet ε/α env parsing;
    `extract_visit_distribution` — visit targets read post-noise)
  - `src/selfplay/evaluation.rs` (eval adjudication config)
  - `python/hyzero/training/trainer.py` (`_policy_loss` entropy bonus + TB
    policy-CE gating + legal/replay entropy metrics; [antisym] probe + regularizer)
  - `python/hyzero/data/tablebase.py` (uniform-over-Syzygy-optimal policy targets)
  - `scripts/run_baseline.sh` (`last_antisym_mean_sum` extraction; 2026-06-10
    knob overrides)
