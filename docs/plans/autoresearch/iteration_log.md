# Autoresearch Iteration Log — 2026-04-15 session

Started after landing dual-model-eval + value-head fix (β=0.1).
All runs 1800s unless noted. Metric: `(8.55 - policy_loss) + (champion_version * 2.0) - (avg_length / 100)`.

## Run Plan

| # | Change | Env / code | Why | Status |
|---|---|---|---|---|
| 1 | β=0.1 baseline establish | `HYZERO_VALUE_OUTCOME_BETA=0.1` (default) | First measurement on new metric | done — score 6.7634 |
| 1b | Cross-run champion loading fix | commit d419f08 | Champion best.pt not found on fresh run start → RandomBackend never swapped | done — commit d419f08 merged |
| 2 | β=0.2 | `HYZERO_VALUE_OUTCOME_BETA=0.2` | Can more outcome signal accelerate value learning without polluting h/g? | done — score 8.329 (corrected) |
| 3-invalid | β=0.3 (ratcheted) | `HYZERO_VALUE_OUTCOME_BETA=0.3`, kept best.pt from run 2 | — | **INVALID** — challenger faced run-2's pretrained champion, not Random; score 4.977 discarded |
| 3 | β=0.3 (fresh) | `HYZERO_VALUE_OUTCOME_BETA=0.3`, `rm -f checkpoints/best*.pt` before launch | Does more outcome signal keep lifting promotions faster than it hurts policy loss? | done — score 11.629 (fresh start) |
| 4 | β=0.5 | `HYZERO_VALUE_OUTCOME_BETA=0.5`, fresh start | Next probe; if plateau/regression, β=0.3 is sweet spot | running |
| 5 | β=1.0 | `HYZERO_VALUE_OUTCOME_BETA=1.0`, fresh start | Upper bound: pure outcome target | queued |
| 6 | Loss rebalancing | tune value/policy loss weights in trainer.py | Policy loss still dominates gradient; rebalance may amplify β gains | queued |

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
| 3-invalid (β=0.3 ratcheted) | 2.53 | 0 | 104.1 | 4.977 — **discarded** |
| 3 (β=0.3 fresh) | 3.40 | 4 | 151.6 | **11.629** |

β=0.2 improvement over β=0.1: **+1.57 points**. This is just above the ±1.0 noise floor — a modest real gain rather than a clear win. The policy loss was actually slightly higher (2.94 vs 2.78), but the extra promotion more than compensated.

β=0.3 (fresh) improvement over β=0.2: **+3.30 points**. Strong signal — promotion count jumped from 2 to 4 while policy loss only rose modestly (2.94 → 3.40). Promotion component dominates: 8 of the 11.6 score points come from promotions.

All future experiments use the corrected formula. `max_champion_version` remains in JSON output for debugging.

## Run 3 (β=0.3 fresh) Results

- Score: **11.629**, promotions: 4, eval_cycles: 5
- Policy loss: 7.76 → 3.40 (worse than β=0.1's 2.78 and β=0.2's 2.94 — outcome signal competes with policy gradient)
- Avg game length: 151.6 (much longer than β=0.1's 100.7 — more exploration, less decisive play)
- Promotion component dominates: 8 of 11.6 score points come from promotions (4 × 2.0)
- Conclusion: as β rises, policy loss increases but promotion count increases faster → net score improves. The value-head outcome blend is helping the ladder climb even as it hurts raw policy quality.
- β=0.5 is the next natural probe; if score plateaus or regresses, β=0.3 is the sweet spot.

## Experiment Protocol (established 2026-04-15)

**Controlled experiments (sweeps — β sweep, loss rebalancing, etc.):**

- Delete `checkpoints/best*.pt` before each run: `rm -f checkpoints/best*.pt`
- This ensures each challenger starts vs the Random backend, not a pretrained champion from a prior run.
- Results are comparable across runs in the sweep.
- Rationale: cross-run champion loading (merged d419f08) is correct for production ratcheting but breaks controlled comparisons. If `best.pt` exists, subsequent challengers face a pretrained opponent and the promotion count becomes a function of that opponent's strength, not the config being tested.

**Production / ratchet mode:**

- Keep `best.pt` across runs (matches user directive that "best.pt should survive across runs").
- Used for final validation after a sweep identifies the best config.
- The cross-run ratchet (d419f08) is intentional here — the champion accumulates skill across sessions.

**Comparison rule:**

- All β sweep runs must be compared only to other fresh-start runs.
- Do not compare a fresh-start run score to a ratcheted-run score.
