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

The same 2026-06-10 diagnosis showed self-play was **92.6% draws** (repetition
63.5%, insufficient-material 20.4%, move-cap 11.9%, fifty-move 3.2%, stalemate
0.9%) — all formerly scored 0.0, so value targets clustered at +0.16±0.6.
Merge `f64ba38` therefore **enables material shaping in `run_baseline.sh`**
(`HYZERO_MATERIAL_SHAPING=1`) after making it safe: terminal scoring is now
routed through `score_board_terminal`, which shapes only *rule* draws and never
*true* draws. See *Material shaping* below.

## Key decisions

- **N-step TD targets at sample time.** Computed in
  `replay_buffer.rs::compute_td_target` (the only site with the full trajectory),
  carried on the **non-serde** `TrainingSample`, NOT `StepRecord` — bincode is
  positional so no on-disk type may gain fields. Formula in step-t POV:
  `G_t = Σ (-1)^i γ^i r_{t+i} + (-1)^m γ^m · bootstrap`, with the terminal branch
  substituting the POV-converted `game_outcome` for the `root_value` bootstrap.
  A shaped rule-draw outcome enters here as that terminal `game_outcome` and
  backfills value targets along the n-step TD tail.
- **β collapses to 0 whenever a TD target exists** (`training.rs`) so the outcome is
  not double-counted through the TD tail; β/conditional-β keep their old meaning only
  on the legacy (`td_targets[k]==None`) path.
- **Value-based resignation for training games; material adjudication eval-only.**
  Resignation is value-based by design — passive shuffling still drives `root_value`
  negative — guarding against the documented *passivity attractor* that material
  adjudication caused in self-play (see `game_task.rs` GameConfig comment). Eval
  outcomes never enter training targets, so `play_game_dual` may safely adjudicate at
  the cap.
- **Material shaping splits draws into true vs rule draws** (`f64ba38`,
  `score_board_terminal` in `game_task.rs`). Checkmate stays ±1. *True* draws
  (stalemate, insufficient material) ALWAYS store 0.0 — drawn by position
  regardless of material. *Rule* draws (threefold repetition, fifty-move, and
  the move-cap where the board is still `Ongoing` at `MAX_GAME_LENGTH=300`) store
  `tanh(Δmaterial/scale)` when `HYZERO_MATERIAL_SHAPING=1`. The PGN result label
  stays 1/2-1/2 for every shaped rule draw (decoupled from the shaped value).
- **MinMaxStats normalization is selection-only.** Defined in `mcts/node.rs`
  (consumed by `puct.rs::select_child_normalized`, threaded through `tree.rs`);
  normalization affects child selection only — stored `total_value`/Q feeding the TD
  bootstraps is unchanged, so the sign-convention regression tests still pass.
- **Antisymmetry regularizer is flag-gated at weight 0** by default (`trainer.py`):
  zero extra forward passes until `HYZERO_ANTISYM_LOSS_WEIGHT > 0`.

## Material shaping (enabled in baseline 2026-06-10, `f64ba38`)

With 92.6% of self-play games drawing to 0.0 (breakdown above), the value head
saw almost no non-zero targets — the same starvation that motivated TD targets,
now from the *terminal* side. `f64ba38` enables `HYZERO_MATERIAL_SHAPING=1` in
`run_baseline.sh` and refactors terminal scoring into `score_board_terminal`:

- **True draws never shaped.** `GameResult::Stalemate` and `InsufficientMaterial`
  return `(0.0, is_draw=true)` unconditionally — the position is drawn whatever
  material sits on the board, so `tanh(Δ)` would teach a false value (a stalemate
  with a queen up is still a draw, not a near-win).
- **Rule draws shaped.** `ThreefoldRepetition`, `FiftyMoveRule`, and the move-cap
  exit (board still `Ongoing` at `MAX_GAME_LENGTH=300`) return
  `(tanh(Δmaterial/scale), true)`. `Δmaterial` is the white-absolute material
  diff in pawn units (P1/N3/B3/R5/Q9, king 0; `compute_material_diff`); `scale`
  is `HYZERO_MATERIAL_SHAPING_SCALE` (default 5.0, clamped [0.5, 100]).
- **PGN label decoupled.** A shaped rule draw still reports 1/2-1/2 and
  `is_draw=true` (for the trainer's draw penalty). Conflating the two was the
  2026-04-23 PGN labeling confusion (shaped Δ>0.5 got tagged 1-0/0-1).
- **The shaped value flows through TD.** It enters as the terminal `game_outcome`
  and backfills value targets via `replay_buffer.rs::compute_td_target`.

This is why the knob was *unsafe to enable before* `f64ba38`: the old `_`
catch-all shaped every non-checkmate terminal, including true draws — e.g. a
K+B-vs-K insufficient-material draw with a residual lead stored ~+0.55,
re-teaching the shuffle/passivity attractor. The three regression tests in
`game_task.rs` lock both arms (repetition-with-lead shapes; stalemate and
insufficient-material with a lead stay 0.0).

**Resignation is not a separate bug.** The resign threshold clamp [-1.0, -0.5]
is unreachable while values cluster near 0 — resignation simply never fires, so
resign tuning is pointless until the value spread recovers (which is exactly
what shaping + TD targets aim to restore).

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
| `HYZERO_MATERIAL_SHAPING` | off | shape rule-draw value targets via `tanh(Δ/scale)` (baseline script: on; true draws never shaped) |
| `HYZERO_MATERIAL_SHAPING_SCALE` | 5.0 | tanh denominator; larger shrinks the signal (clamp 0.5..100) |
| `HYZERO_RESIGN` | on | value-based resignation (self-play only) |
| `HYZERO_RESIGN_THRESHOLD` | −0.90 | root_value at/below which a ply counts (clamp −1..−0.5; unreachable while values cluster near 0) |
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
| `HYZERO_LR_COSINE_T_MAX` | (script-computed) | cosine LR decay span; `run_baseline.sh` sets `DURATION/60*18` (≈18 trainer steps/min), floored at 100 — no longer hardcoded 14000 |
| `HYZERO_ANTISYM_LOSS_WEIGHT` | 0 | antisymmetry regularizer weight |
| `HYZERO_ANTISYM_PROBE_N` | 8 | samples in the per-call `[antisym]` probe (clamp 1..64) |

Set the booleans off + weights 0 to recover pre-fix runtime behavior with no code
revert (no on-disk format changed). The 2026-06-10 knobs default to *legacy*
behavior in code (`HYZERO_TB_POLICY_WEIGHT=1.0`, ε=0.25, shaping off);
`run_baseline.sh` overrides them (0.0, 0.10, shaping on).

## Gotchas

- **Bincode is positional.** Never add serde fields to `StepRecord`/`GameTrajectory`
  mid-format — `#[serde(default)]` does NOT make them additive. TD targets live on the
  non-serde `TrainingSample` for this reason.
- **Material shaping must split true vs rule draws.** Pre-`f64ba38` the old `_`
  catch-all shaped *every* non-checkmate terminal — a true draw with a material
  lead (K+B vs K → ~+0.55) re-taught the shuffle attractor, which is why the knob
  was unsafe to enable. `score_board_terminal` now hard-codes 0.0 for stalemate
  and insufficient material; never fold them back into the rule-draw arm.
- **Move-cap is `GameResult::Ongoing`.** At `MAX_GAME_LENGTH=300` the board never
  reached a terminal, so the cap match-arm pairs `Ongoing` with the rule draws —
  it is shaped, not a true draw.
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
- **Resign threshold is unreachable in a starved value head.** The clamp
  [-1.0, -0.5] can't be hit while values cluster near 0, so resignation never
  fires — don't tune resign params until shaping + TD restore the value spread.
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
  - `src/data/replay_buffer.rs::compute_td_target` (n-step TD; shaped outcome backfill)
  - `src/py/training.rs::assemble_batch_arrays` (TD/β composition)
  - `src/selfplay/game_task.rs::score_board_terminal` (true-vs-rule-draw split;
    `compute_material_diff`, `material_shaping_scale/_enabled`; resignation +
    calibration fraction + temp anneal helpers; GameConfig adjudication fields)
  - `src/mcts/node.rs::MinMaxStats`, `src/mcts/puct.rs::select_child_normalized`,
    `src/mcts/tree.rs` (per-search min/max threading; Dirichlet ε/α env parsing;
    `extract_visit_distribution` — visit targets read post-noise)
  - `src/selfplay/evaluation.rs` (eval adjudication config)
  - `python/hyzero/training/trainer.py` (`_policy_loss` entropy bonus + TB
    policy-CE gating + legal/replay entropy metrics; [antisym] probe + regularizer)
  - `python/hyzero/data/tablebase.py` (uniform-over-Syzygy-optimal policy targets)
  - `scripts/run_baseline.sh` (material shaping on; cosine T_max = DURATION/60*18;
    `last_antisym_mean_sum` extraction; 2026-06-10 knob overrides)
