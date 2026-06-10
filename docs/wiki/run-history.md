# Run History — 2026-06-10 Overnight Loop: Policy Flattening Root-Caused

The 2026-06-10 overnight autonomous loop (4 runs, 4 merged fixes, 1
promotion) root-caused the policy flattening to target-side noise, peeled in
three layers: an entropy bonus in the policy loss, uniform tablebase policy
targets in the CE, and root Dirichlet noise baked into visit-count targets —
causally confirmed by run 4. All runs resumed `mate_pretrained.pt` (internal
step base 3800) with the eval pool seeded v3806+v3905, `GAMES_PER_SIDE=8`,
antisym weight 0.01, LR `T_max=14000`.

## Run 1 — entropy weight β=0.003 (killed at 23 min)

- k0 raw `pred_entropy` 2.38 → 5.72 by step ~464 — identical to the legacy
  run-2 divergence.
- **Discovery:** the entropy term in trainer.py `_policy_loss` is an entropy
  BONUS (`ce + β·Σp·log p` — flattens the policy; its comment says "penalize
  over-sharp"). The prior recommendation "lower the weight" was
  wrong-direction: any β>0 is harmful for distillation.
- Merged 58067e5: β default 0.0, plus new `pred_entropy_legal` metric — the
  old `pred_entropy` reads pre-mask logits, inflated by gradient-orphaned
  illegal logits (legal-uniform ≈3.5 vs full-uniform 8.45).

## Run 2 — β=0 (killed at ~80 min)

- `pred_entropy_legal` 0.90 → 1.71 and still climbing — the bonus is ruled
  out as sole cause.
- **Found:** TB trajectory rows are 45% of every batch (tb_frac=0.45) with
  uniform-over-Syzygy-optimal policy targets (48% of positions multi-optimal;
  mean legal support 19.8), entering policy CE at ALL k because the
  trajectory cache sets `is_tablebase=False`.
- Two quantitative locks: blended tgt_entropy 0.45·0.72 + 0.55·1.15 ≈ 0.95
  (matches observed ~1.0–1.2); k1–5 entropy plateau = log(19.8) = 2.986
  (matches 2.9–3.0 exactly).
- Merged d8cded7: `HYZERO_TB_POLICY_WEIGHT` gates TB rows out of policy CE
  only (TB value/reward supervision preserved at all k; code default 1.0 =
  legacy, baseline script 0.0), plus regime-split metrics
  `pred_entropy_legal_replay` / `pred_top1_replay`.

## Run 3 — TB gated (killed at ~80 min)

- Replay-only rows STILL flattened, but to a stable measurable floor:
  inferred replay target entropy ~1.9–2.1, preds ~2.5–2.6, top1 0.26. Eval
  candidates degraded vs v3905 (0.531 → 0.406), 0 promotions.
- **Diagnosis:** ε=0.25 root Dirichlet noise is stored in visit-count targets
  (tree.rs `extract_visit_distribution` reads raw post-noise visits), and
  draw-dominated play (value ≈ 0 in-search, resign never fires) cannot
  re-concentrate visits → targets noise-floored at ~2.0 nats.
- Merged 58beff5: `HYZERO_DIRICHLET_EPS`/`HYZERO_DIRICHLET_ALPHA` env knobs
  (renamed from `HYZERO_DIRICHLET_EPSILON`, which nothing ever set); baseline
  ε=0.10; SIMS 200 → 300.

## Run 4 — ε=0.10 (completed, 4h30m, clean exit)

- **Causal confirmation:** inferred replay targets ~1.6 vs ~1.9 at matched
  steps. Policy held end-to-end — `pred_entropy_legal_replay` ~2.06 / top1
  ~0.36 vs run 3's collapsed 2.55/0.26. k0 value MSE ~0.035; antisym
  2.0 → ~0.3.
- **Champion promotion: v3840** (cycle 3, Elo 1544.1 > 1520 gate). The gate
  is Elo-vs-pool: candidate win_rate 0.562 in cycle 1 was correctly rejected
  at Elo 1519.7, and a champion's own version is excluded from its opponent
  pool by design.
- 6 eval cycles total; cycles 4–6 scored 0.500/0.469/0.484 vs {v3905, v3806},
  no further promotions.
- 316 games at 18.4 steps/min; final optimizer step 4,960 of `T_max=14000` —
  the LR never left the high regime (size T_max ≈ expected steps next run).
- Flip side: decisive-target fraction fell ~3.7x as the buffer filled with
  ordinary draws (±0.1 bins 446 → 837; resign 0/22 calibration probes, 0
  false positives).

## Same-day infra merge — eval-ladder hardening (f867c4a)

Found while investigating a mid-run eval silence that turned out to be slow
cadence (eval actually completed all 6 cycles). Dropped champion-batcher
replies now surface as a recoverable `EvalError` instead of a silent panic
(the eval task's `JoinHandle` is dropped, so panics vanish); the champion
batcher is kept alive across promotions; cycles that saw ANY inference error
skip the promotion decision (no Elo from neutral-eval games).
Regression-test-proven.

## Preserved checkpoints

- `checkpoints/backup_champion_v3840_20260610.pt` — current champion
- `checkpoints/backup_final_model_v004109_20260610.pt`

## Next-run recommendations

In priority order:

1. Decisiveness levers: resign-threshold calibration, selfplay adjudication,
   decisive-start curriculum via `HYZERO_STARTS_FILE`, temperature schedule.
2. Size the LR cosine `T_max` ≈ expected steps (~18 steps/min at SIMS 300 /
   `GAMES_PER_SIDE=8`).
3. Optional: ε sweep 0.05–0.15 now that it is an env knob.

## Related

- Wiki: [Training Signal](training-signal.md),
  [Elo Ladder Evaluation](elo-ladder-eval.md),
  [Champion Pool Promotion](champion-pool-promotion.md)
- Scripts: `scripts/run_baseline.sh`

# Run History — 2026-06-09 Signal-Starvation-Fix Validation

Two validation runs on 2026-06-09 confirmed the [signal-starvation
fix](training-signal.md) restores learning signal. Both resumed from the
degenerate `mate_pretrained.pt` (antisym ≈ 2.0) with all fixes active
(`HYZERO_TD` n=5 γ=0.997, resignation −0.90/4/min-ply-30, temp anneal, MCTS
qnorm+FPU, eval adjudication margin 5). Run 2 additionally enabled
`HYZERO_RESIGN_DISABLE_FRAC=0.1` and `[Termination]` PGN headers. Net result:
the value head un-degenerated and tracks targets; the policy head is now the
bottleneck.

## Run 1 — 30-minute smoke (baseline_20260609_185443)

- 784 training steps; 55 self-play games averaging 91 plies (123 pre-fix).
- Antisym sum fell 2.0 → ~0.52 plateau.
- Eval decisive rate 27.4% (~7% pre-fix).
- First-ever promotion (bootstrap fallback into an empty champion pool).
- Composite baseline score 7.70 vs 5.29 pre-fix.

## Run 2 — 3h16m long run (baseline_20260609_195343, terminated early by user)

5520 steps, v3800→v4145, SIGTERM at 23:09. Zero errors logged.

- **Value head recovered.** k0 value MSE 1.08053 → 0.01459; predictions
  un-saturated and tracking targets.
- **Antisym broke 2.0 permanently**, settling to a floor of ~0.80 (min 0.186;
  Q1→Q4 mean 0.802 → 0.831, a slight upward drift).
- **Promotions:** v3806 (cycle 2, wr 0.562) and v3905 (cycle 8, wr 0.625);
  none in the last 9 cycles. The [Elo ladder](elo-ladder-eval.md) engaged
  post-v3905 with real pool matches, candidate_elo 1419–1516.
- **Resign calibration:** 35 ungated games, 0 false positives.
- Eval decisive rate 25.7%.
- **Policy is the bottleneck:** k0 policy `pred_entropy` diverges 2.42 → 5.63
  against targets at ~1.0–1.26.
- LR cosine `T_max=7000` was exceeded — the last third of the run sat at
  `eta_min`.
- `baseline_score.json` was NOT refreshed (script killed mid-sleep) — analyze
  this run from the log, not the stale JSON. See
  [Baseline Scoring](baseline-scoring.md).

## Preserved checkpoints

- `checkpoints/backup_champion_v3905_20260609.pt` — final champion
- `checkpoints/backup_champion_v3806_20260609.pt`
- `checkpoints/backup_final_model_v004143_20260609.pt`

**WARNING:** `run_baseline.sh` purges `best_v*.pt` and `model_v*.pt` at the
start of the next run — only `backup_*` names survive.

## What's proven / what's open

Proven:

- The value-target pipeline produces real learning signal: value MSE collapses,
  predictions track targets, and the antisym degenerate fixed point (≈2.0) is
  escaped and stays escaped.
- Resignation + adjudication raise decisive rates ~4x without false positives.
- The promotion/ladder loop functions end to end (two real promotions, ladder
  Elo on real pool matches).

Open:

- Policy entropy diverges instead of converging — the policy head has not
  started learning meaningfully.
- Antisym floor of ~0.8 (and slight upward drift) — not zero-symmetric yet.
- Promotion stalled after cycle 8; 8-game win rates are noise-dominated.

## Next-run recommendations

In priority order:

1. Lower `HYZERO_POLICY_ENTROPY_WEIGHT` from 0.01 (entropy divergence above)
   and/or raise sims.
2. Extend the LR cosine `T_max` beyond the planned run length.
3. Seed the eval pool with the two backed-up champions and raise games/cycle —
   8-game win rates swung 0.125–0.625, noise-dominated.
4. Optional: `HYZERO_ANTISYM_LOSS_WEIGHT` ~0.01 to push the 0.8 antisym floor
   (not stability-critical).

Same-day infra merges that affect reading future logs: PGN `[Termination]` +
`[SetUp]`/`[FEN]` headers; `HYZERO_PGN_SAMPLE_RATE` (`run_baseline.sh` sets
1.0 — full self-play streaming from the NEXT run); live web visualizer
(`cd python && python -m hyzero.viz.live_viewer --logs-dir ../logs --port
8642`); hermetic eval tests (no longer pollute `logs/eval_games.pgn`).

## Related

- Wiki: [Training Signal](training-signal.md),
  [Baseline Scoring](baseline-scoring.md),
  [Elo Ladder Evaluation](elo-ladder-eval.md),
  [Champion Pool Promotion](champion-pool-promotion.md)
- Scripts: `scripts/run_baseline.sh`
