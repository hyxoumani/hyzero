# Post-Phase 2 Decision Tree

Context: 10-hour baseline run started 2026-04-17 05:22 with Phase 1 fixes + f9012ee
intervention (material-at-cap restored, adjudication still removed). First 6 cycles
showed the pipeline is stable but **plateaued at 0 challenger wins** — neither
challenger nor Random champion can force decisive outcomes.

## Decision matrix (run at ~06:15 UTC, ~1h elapsed)

| Outcome at 10h | Interpretation | Next experiment |
|---|---|---|
| Score ≥ 10 + ≥ 2 promotions | Phase 1 + material-at-cap works; pipeline is sound | Proceed to Phase 3b (plane 101 cleanup), measure incremental wins |
| Score 6-10, 1 promotion | Partial success; slow but forward progress | Run another 10h before concluding; or try β=0.5 to strengthen outcome signal |
| Score 4-6, 0 promotions (current trajectory) | Material-at-cap alone is insufficient; need signal amplification | Reintroduce adjudication at HIGH threshold (10-15) — fires only on truly decisive positions; weak passivity risk |
| Score < 4, failing in new ways | Something worse than expected | Full diagnostic; possibly revert f9012ee or Phase 1 fixes selectively |

## Current forecast (as of cycle 6, 06:15 UTC)

Score 4.80, 0 promotions. Projecting to 10h:
- Policy loss will keep dropping (now 2.36, maybe lands 1.8-2.2)
- Promotions: likely 0 if no win breakthrough happens
- Score projection: 5.0-6.5 if no promotions, 10+ if one promotion fires

Most likely endpoint: **plateau at 5-6**. Better than broken baseline (3.66) but far from peak (14.51).

## The "material-at-cap isn't enough" hypothesis (likely winner)

**Root cause**: `tanh(Δmaterial/5)` gives the value head *some* signal, but since most self-play games converge to Δmaterial ≈ 0 at termination (both sides preserve material similarly), the target distribution is peaked near 0. Gradient descent minimizes MSE by predicting the mean (0), losing the fine-grained distinction between positions.

The historical peak 14.51 had adjudication firing at threshold 6 for 10 plies. This:
- Ended ~70% of games before natural terminals
- Produced ±1 outcomes on ~70% of games instead of tanh(Δ/5) spread across all of them
- Gave value head a much higher ratio of "strong ±1 signal" to "weak material signal"
- Came with a passivity cost that eventually dominated in longer runs, but provided the bootstrap

## Post-run intervention candidates, ranked

### Primary: High-threshold adjudication
- Location: `src/selfplay/game_task.rs` (restore the adjudication block, but with threshold=10 or 15 instead of 6)
- Gives rare but strong ±1 signal without firing on normal material fluctuations
- Still provides bootstrap benefit of old adjudication
- Weaker passivity risk because threshold is high enough that merely-preserving-material won't trigger it

### Secondary: Loss weighting by target magnitude
- Location: `python/hyzero/training/trainer.py` value loss computation
- Weight each sample's MSE contribution by `|target|` or `tanh(|target|·2)` so near-zero targets contribute less gradient
- Effectively prioritizes strong signals without changing outcome formula
- Doesn't require re-enabling adjudication

### Tertiary: β=0.5 outcome blend
- Env var only: `HYZERO_VALUE_OUTCOME_BETA=0.5`
- Historical evidence: β=0.5 was tested at 8.07 (vs 11.63 for β=0.3) in the pre-Phase-1 era. Regression then, may differ now.
- Simplest change, easiest to test

### Stretch: Priority replay
- Oversample trajectories containing non-zero outcomes in batch sampling
- Requires `ReplayBuffer.sample_batch` changes
- ~50 LOC moderate complexity
- Would be valuable longer-term even if primary fix works

## What NOT to try

- β > 0.7 (historical evidence: regresses via "fast-training paradox")
- `HYZERO_VALUE_LOSS_WEIGHT > 2` (previously regressed 11.63 → 4.84)
- Full adjudication restore at threshold=6 (causes original passivity trap)
- Removing material-at-cap entirely (causes dead value head per attempt 1)

## Sign check for next run

Before committing any intervention, verify:
- [ ] Value loss exits 0.005 plateau and shows meaningful range (0.02-0.15)
- [ ] Challenger scores at least 1 win in first 3 cycles (not 0 across the run)
- [ ] Games continue to terminate naturally (avg length 100-200, <10% at cap)
- [ ] No return of `b1a3` + rook-shuffle pattern in self-play PGN (if Phase 3a logger enabled)

If any of those fail, the intervention has introduced a new failure mode and should be rolled back.
