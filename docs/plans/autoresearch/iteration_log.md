# Autoresearch Iteration Log — 2026-04-15 session

Started after landing dual-model-eval + value-head fix (β=0.1).
All runs 1800s unless noted. Metric: `(8.55 - policy_loss) + (champion_version * 2.0) - (avg_length / 100)`.

## Run Plan

| # | Change | Env / code | Why | Status |
|---|---|---|---|---|
| 1 | β=0.1 baseline establish | `HYZERO_VALUE_OUTCOME_BETA=0.1` (default) | First measurement on new metric | done — score 6.7634 |
| 1b | Cross-run champion loading fix | commit d419f08 | Champion best.pt not found on fresh run start → RandomBackend never swapped | done — commit d419f08 merged |
| 2 | β=0.2 | `HYZERO_VALUE_OUTCOME_BETA=0.2` | Can more outcome signal accelerate value learning without polluting h/g? | running |
| 3 | β=0.05 | `HYZERO_VALUE_OUTCOME_BETA=0.05` | If 0.1 already too much, is 0.05 safer? | queued |
| 4 | LR cosine no warmup | trainer.py wrap CosineAnnealingLR | Prior warmup experiment (c5440f2) failed because warmup-100 ate early steps | queued |
| 5 | Temperature smoothing | selfplay/game_task.rs exp decay | Hard cutoff → smooth decay for exploration-exploitation tradeoff | queued |

## Decisions

- Keep if score improves >1.5 points (beyond ±1.0 noise floor)
- Revert if regresses
- After 2 consecutive regressions in same direction, pivot

## Observations

- **Signal check (run 1, step 1)**: `value=0.0145, reward=0.0757` → value head ALIVE for the first time. Prior all runs: `value=0.0000, reward=0.0006`.
- **Signal check (run 1, step 64)**: `value=0.0011, reward=0.0003` → value loss settled low because target magnitude ~0.1 (β=0.1 × outcome ±1). Expected.
- Reward head still effectively dead (not addressed by this fix — separate class-imbalance issue).

## Run 1 Results (β=0.1, baseline)

- Score: 6.7634 (vs prior 5.67 under old formula — not directly comparable)
- Games: 302, Training steps: 4816
- Policy loss: 7.74 → 2.78
- Eval cycles: 13, Promotions: 1 (v0→v1 on first cycle at 0.562)
- Ladder status: stalled at v1 with 12 consecutive 0.500 win rates (symmetry collapse within-run)
- Value signal: ALIVE (0.0145 → 0.0011 — first measurable value loss ever)
- Interpretation: β=0.1 outcome blend insufficient to break challenger↔champion symmetry when both are snapshots of same training run drifting in parallel. Promotion happens once when the challenger first diverges from the frozen Random champion, then stalls because the new champion IS the training distribution.

## Metric Correction Note (2026-04-15)

Run 2 (β=0.2) initially reported a score of **28.3289**, which was inflated by the formula using `max_champion_version=12` (the training-version-number tag on the winning checkpoint) rather than `promotions=2` (the actual count of promotion events). Because training runs ~10-15x faster than eval cycles, a single promotion can jump the champion_version tag from 1 to 12, yielding 24 points of phantom "skill gain" instead of 4.

**Corrected formula (formula_version=2):** `score = (8.55 - policy_loss) + (promotions * weight) - (avg_length / 100)`

Corrected scores:

| Run | policy_loss | promotions | avg_length | score |
|-----|------------|------------|------------|-------|
| 1 (β=0.1) | 2.7798 | 1 | 100.7 | **6.763** |
| 2 (β=0.2) | 2.9351 | 2 | 128.6 | **8.329** |

β=0.2 improvement over β=0.1: **+1.57 points**. This is just above the ±1.0 noise floor — a modest real gain rather than a clear win. The policy loss was actually slightly higher (2.94 vs 2.78), but the extra promotion more than compensated.

All future experiments use the corrected formula. `max_champion_version` remains in JSON output for debugging.
