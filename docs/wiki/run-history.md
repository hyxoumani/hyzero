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
