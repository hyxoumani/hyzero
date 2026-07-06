# Conversion Campaign — July 2026

Durable reference for the endgame-conversion campaign `runs/auto-20260702-151405`
(43 iterations, ~4 days continuous, 2026-07-02 → 07-06), the successor to the
[signal-starvation fix](training-signal.md). The objective shifted from bench
*score* to **mate conversion**: the fraction of won endgame starts a net can drive
to a real checkmate. This page supersedes stale rows in `training-signal.md` and
`baseline-scoring.md` (see [Corrections](#corrections-to-stale-wiki-entries)) —
those pages could not be edited under the one-shot wiki gate.

## Verdict summary

- **Fine-tuning from `mate_pretrained` cannot escape the conversion plateau.**
  ~50 fine-tune runs across every labeling/horizon/search/LR/duration lever hold
  conversion at **10–17% (plateau ≈13.9%)**, knob-invariant, and it *degrades*
  with training time (6h runs → 10.0% flat-WDL / 12.5% DTZ-graded). The knob-space
  is exhausted; escaping the plateau needs an architectural value change.
- **From-scratch + categorical (HL-Gauss) value head is the one real break.**
  A random-init net with a distributional value head produced the **first monotone
  DTZ→value ordering of the entire two-campaign history** (Pearson −0.85, stable
  from 4h onward — every fine-tuned lineage was flat/saturated). Its conversion
  trajectory is *monotone with training* (unlike fine-tune): **6.7%@4h → 9.2%@12h
  → 14.2%@24h → 14.2%@30h** — it *caught* the fine-tune plateau at 24h with a
  rising trajectory rather than a decaying one.
- **Moves-left head (MLH) trained correctly but the search bonus is flat.** The
  head's `m` prediction has the correct ordering (far-from-mate → more plies) but
  is contrast-squashed at `MLH_CAP=100` (gap +0.049, ~0.8σ, compressed 0.24–0.29
  band); the PUCT bonus moves conversion by **zero games across 0.05/0.10/0.20**.
  A `MLH_CAP=30` retrain (wider target range) was in flight as of writing.

## Root causes (verified)

Code forensics found **no implementation bugs** — all four are *design* issues,
each confirmed by numeric probe (consistent, not speculative):

1. **Shaped draw labels invert incentives.** `tanh(Δmaterial)` rule-draw
   surrogates teach the value head off-objective; a fifty-move draw with a lead
   stored **+0.995 undiscounted**. No SOTA precedent (MuZero/AZ train on exact
   outcomes). Shaping is now **OFF**.
2. **Atari-style 5-step TD misapplied to board games.** MuZero board-game agents
   use the full-game outcome `z` (γ=1, n=∞) at every step; the 5-step TD tail
   starves the mate signal. Fixed by `HYZERO_VALUE_TARGET_MODE=outcome`.
3. **Flat TB labels / value saturation.** Binary WDL/outcome targets make value
   ≈+1 across the *whole* won region → zero distance-to-mate gradient → PUCT
   cannot prefer a mating move over a shuffle/hang. DTZ-*graded* MSE targets do
   **not** fix this on a tanh head — they wash out through tanh + the ±1-outcome
   majority (proven flat @iter38 / saturated @iter39 / desat-but-unordered
   @iter40). Only a **categorical/HL-Gauss head from scratch** installed a real
   ordering.
4. **Latent repetition blindness.** No repetition/rule50 planes → the net cannot
   see a shuffle developing; conversion failure is 72% "piece kept but
   shuffling/stalemate", only 19% hangs. Repetition planes remain **deferred**
   (checkpoint-incompatible, see [Open problems](#open-problems)).

## New env knobs

Defaults are code defaults; "bench" is the `run_baseline.sh` / campaign setting.

| Var | Semantics | Default | Bench |
|-----|-----------|---------|-------|
| `HYZERO_VALUE_TARGET_MODE` | `outcome` = full-outcome-z (γ=1, n=∞) to every step (MuZero board-game regime); else legacy TD | td | `outcome` |
| `HYZERO_TB_RESCORE` | replace late-game value targets with exact Syzygy WDL on TB-covered tail | off | on |
| `HYZERO_TB_WDL_PATH` | path to Syzygy WDL files for rescoring (939,703 entries loaded) | unset | set |
| `HYZERO_TB_RESCORE_GRADED` | use the DTZ-graded CSV (468k positions / ~1.08M join hits) instead of flat WDL | off | on |
| `HYZERO_VALUE_HEAD` | `categorical` = HL-Gauss distributional head (value = support expectation); else scalar tanh. Loading a categorical ckpt REQUIRES this set | tanh | `categorical` (scratch) |
| `HYZERO_FROM_SCRATCH` | random-init instead of `mate_pretrained`; pair with `HYZERO_RESUME_FROM=backup_*` for continuations | off | `1` (scratch) |
| `HYZERO_MOVES_LEFT_HEAD` | MCTS reads `m` (normalized plies-remaining) from `root_setup_batch` | off | `1` (MLH lineage) |
| `HYZERO_MLH_CAP` | ply cap normalizing the moves-left target; lower = wider target dynamic range | 100 | 30 (retrain) |
| `HYZERO_MLH_LOSS_WEIGHT` | masked-MSE aux loss weight on the MLH head (raising it risks hang-mass regression) | ~1.0 (unverified) | default |
| `HYZERO_MLH_SEARCH_BONUS` | PUCT bonus scale applied to `m`; 0 = head inert in search | 0 | **0** (flat across 0.05/0.10/0.20) |
| `HYZERO_MLH_Q_THRESHOLD` | only apply the bonus above this Q (winning side only) | unverified | default |
| `HYZERO_EVAL_MIRRORED_STARTS` | mirror FEN pairs across color halves in eval (kills start-material luck; enabled first-ever promotions) | off | on |
| `HYZERO_SELFPLAY_ADJUDICATE` | adjudicate self-play by material at the move cap | off | on |
| `HYZERO_SELFPLAY_ADJ_MARGIN` | pawn margin to adjudicate (bracket resolved: 10 & 14 worse) | — | 12 |
| `HYZERO_MAX_GAME_LENGTH` | move cap; cap exit is `GameResult::Ongoing` (a rule draw, shaped when shaping on) | 300 | 300 |
| `baseline_score.json.conversion` | KQvK/KRvK self-play mate audit over the **rotated** current-run PGN | — | emitted |
| `baseline_score.json.probe` | conversion probe: 120 fixed won-endgame starts ×2 colors, adjudication OFF, checkmate count | — | emitted (primary metric) |

## Corrections to stale wiki entries

This page is authoritative where it conflicts with older pages:

- **`HYZERO_FPU`** — `training-signal.md` documents `HYZERO_FPU=0.25`. An
  experiment renamed it `HYZERO_FPU_REDUCTION` (default 0.2, iter-1) then
  **reverted**; the flat **0.25** knob remains the live one.
- **`HYZERO_TB_POLICY_WEIGHT`** — `training-signal.md` records the baseline as
  `0.0`, root-caused as TB-policy CE pollution. Under the conversion objective
  this was **deliberately reversed to 0.5** (TB-optimal DTZ moves strip prior
  mass off the observed queen-hangs); rationale is documented in
  `run_baseline.sh`. TB value/reward supervision was always untouched.
- **Material shaping** — `training-signal.md` documents `HYZERO_MATERIAL_SHAPING=1`
  as kept at scale 3.0/5.0. It is now **OFF** (root cause #1: shaped draw labels
  invert incentives; no SOTA precedent).

## Measurement doctrine

- **Bench score is unusable for single-run decisions** — sd ≈2.6 with a bimodal
  ±2 promotion lottery. No single lever was ever statistically resolved at 1800s
  (the keeps A/B was Welch NS, t=1.19); adopt levers on *mechanism + direction*,
  not demonstrated effect. See [Baseline Scoring](baseline-scoring.md).
- **The conversion probe is the primary metric**: 120 fixed won-endgame starts
  ×2 colors vs self, **adjudication OFF**, count real checkmates. Low-noise
  relative to score; remaining spread is genuine training variance.
- **Per-run PGN rotation is mandatory.** `logs/selfplay_sample.pgn` /
  `eval_games.pgn` are open create+append and were never truncated — the first
  conversion claim triple-counted a month-old frozen tail (contamination
  incident, `iter-35.md`). `run_baseline.sh` now rotates both to
  `*_prev_<epoch>.pgn` before each run; **never audit an un-rotated PGN**.
- **Pre-registered blocks only** (PROTOCOL v3+): declare a 3-run block before any
  run executes, all runs count, no post-hoc dropping. Policy-loss is a red-herring
  (improves without strength). Trainer-side metric levers never moved strength;
  only game-generation/labeling/architecture changes ever did.

## Checkpoint lineage & how to resume

Protected `backup_*` names (the from-scratch categorical lineage):

- `backup_scratch_s1_v604.pt` — 4h, first monotone ladder (6.7%)
- `backup_scratch_s2_v11034.pt` — 12h continuation (9.2%)
- `backup_scratch_s3_v34015.pt` — 24h, caught the plateau (14.2%)
- `backup_mlh_v68325.pt` — 30h + MLH head (14.2%, best conversion ckpt this lineage)

**Wipe trap:** in `HYZERO_FROM_SCRATCH` mode, `model_v*.pt` files are **WIPED at
run start**. Always `cp` a checkpoint to a `backup_*` name before resuming, and
continue via `HYZERO_RESUME_FROM=backup_*`. The probe always loads the newest
checkpoint by mtime.

**Load requirements:** a categorical ckpt loads only with
`HYZERO_VALUE_HEAD=categorical`; an MLH ckpt additionally needs
`HYZERO_MOVES_LEFT_HEAD=1` (loading is tolerant — old ckpts load without the head,
but a categorical/MLH ckpt under mismatched envs will not read correctly).

## Open problems

- **Repetition/rule50 planes** — the highest-value remaining architectural lever
  (root cause #4) but blocked on `NUM_OBS_PLANES` const threading and is
  **checkpoint-incompatible**; only reachable via another from-scratch run.
- **PGN concurrent-append splicing** — the lock-free append splices headers into
  ~16% of games (mutex fix candidate); corrupts naive PGN audits.
- **Hang-mass regression under MLH aux** — queen-safety worsened 0.384 → 0.524
  during MLH training; the aux loss may trade off policy queen-safety (watch;
  do not up-weight `HYZERO_MLH_LOSS_WEIGHT` to compensate).
- **`replay.rs` viewer bug** — the old promotion-suffix bug persists (viewer-only,
  does not affect training). See [Replay Subsystem](replay-subsystem.md).
- **`MLH_CAP=30` verdict pending** — the wider-contrast retrain was in flight;
  verdict rule was ≥+5pp = MLH confirmed, else MLH exhausted → synthesis
  (repetition planes from-scratch / plain training / report).

## Related

- [Training Signal](training-signal.md) — the signal-starvation predecessor (note
  the FPU / TB-policy / shaping corrections above)
- [Baseline Scoring](baseline-scoring.md) — score formula, `baseline_score.json`
- [Run History](run-history.md), [MCTS](mcts.md), [Neural Networks](neural-networks.md)
- Run artifacts: `runs/auto-20260702-151405/{STATE.md,PROTOCOL.md,CONSOLIDATION.md,results.tsv,iter-*.md}`
