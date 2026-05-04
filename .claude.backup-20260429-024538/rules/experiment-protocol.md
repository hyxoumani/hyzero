---
paths:
  - "scripts/run_baseline.sh"
  - "docs/plans/autoresearch/**"
  - "docs/plans/**/*.md"
---

# Experiment Protocol & Validation

## Closed-Loop Self-Play Metric Alignment

MuZero training is a **closed-loop multi-head system**: the model generates training data
(via MCTS), trains on that data, and the updated model generates the next games. Metrics
measuring *local* signal (policy loss) can diverge from metrics measuring *global* quality
(actual play). This is known as the "Fast-Training Paradox."

**Rule**: Never accept a score improvement based on training loss alone.

**Validation checklist** before claiming a win:
1. **Check promotions**: Promotions must be ≥ baseline (or reasonably close). If policy loss
   improves but promotions drop to 0, you've hit the closed-loop paradox. Example:
   - value_weight=5.0: policy loss 2.70 (best), promotions 0 ✗, score 4.84 ✗
   - β=0.3 baseline: policy loss 3.40, promotions 4 ✓, score 11.63 ✓

2. **Check early eval cycles**: Log win_rate at cycle 1-3, not just the final cycle. If early
   cycles show the challenger losing to Random while late cycles look strong, the model may
   be over-converging to draws (self-play symmetry collapse) or MCTS may have poor value
   estimates early on. Example: β>0.3 configs show 0% win rate at cycle 1-4 despite best policy loss.

3. **Check game length**: Longer games (~150 moves) indicate more exploration, which often correlates
   with better training data. Configs that cut game length (faster convergence) often regress
   despite lower loss. Healthy range: 100-150 moves depending on β.

4. **Run at least once more** if score improvement is <1.5 points. Baseline has ±1 point variance
   from eval noise (binomial variance on 10-game samples) and training step count variance (±50% jitter).

## Fresh-Start Protocol for Fair Comparison

When running controlled experiments (sweeps, parameter tuning):

```bash
# Delete checkpoints BEFORE each run to ensure fair baseline comparison
rm -f checkpoints/best*.pt

# Set your experiment parameter
export HYZERO_VALUE_OUTCOME_BETA=0.4

# Run baseline
bash scripts/run_baseline.sh 1800
```

**Why**: Cross-run champion loading (commit d419f08) is correct for production, but it breaks
controlled comparisons. If `best.pt` exists from a prior experiment, the next run's challenger
faces a pretrained opponent, making promotion counts a function of *that opponent's strength*
instead of the parameter being tested.

**Exception**: Use ratchet mode (keep checkpoints) for final validation after a sweep identifies
the best config. This allows the champion to accumulate skill across sessions.

## Metric Definition Precision

Score metrics that multiply by derived values need explicit event counting, not proxy variables.

**Wrong**: `max_champion_version` as multiplier (training-version tag from checkpoint filename)
- Reason: Training runs ~10-15x faster than eval cycles. A single promotion can jump
  the version tag from v1→v12, yielding 24 phantom points instead of 2.

**Right**: `PROMOTIONS=$(grep -c "\[eval\] promoted" "$run_log")` as multiplier
- Counts discrete promotion events, not checkpoint indices.
- Immune to training/eval rate differences.

## Value Head Verification

The value head is "barely alive" as of 2026-04-15 (non-zero loss but unreliable). Before
claiming improvements from value-head changes, run verification tests:

1. **Held-out MSE**: Train on 90% of trajectory, measure MSE on held-out 10%. If training
   loss → 0 but held-out MSE >> 0, the head is overfitting or fitting stale targets.

2. **Ablation test**: Set `HYZERO_VALUE_OUTCOME_BETA=0.0` (pure MCTS Q-estimates, no outcome
   signal). If score regresses >2 points, that confirms the outcome signal is carrying improvement.

3. **Cycle-1 diagnostic**: Log win_rate at cycle 1 eval. If <0.40 (worse than 50%), value head's
   root-level initialization is not discriminative yet.

See `docs/wiki/neural-networks.md` section "Value Head Status: Barely Alive" for context.
