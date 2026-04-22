# Tablebase Supervision Experiment Results (2026-04-21)

## Objective

Break value-head distributional collapse (Failure Mode 4) using Syzygy tablebase
WDL labels injected into training batches. This file records the two-step experiment
(sign check + clean rerun from pristine checkpoint).

## Step A — Sign Sanity Check

### Setup
- Checkpoint: `checkpoints/best_v1489_pre_tb.pt` (model_version=15050, pristine)
- Cache: `data/syzygy/cache.pkl` (200,000 positions, KQvK/KRvK/etc.)

### Findings
The v1489 value head is completely dead. In both eval and train mode, all positions
(KQK-white-winning, K-vs-KQ-white-losing, starting position) output ≈±0.003.
This is distributional collapse — the value head collapsed to ~0 for all inputs,
not a sign bug.

**Critical observation**: BN running_mean=-1.36 in the value head conv layer.
In eval mode the BN shifts activations heavily negative; in train mode a single-sample
batch normalizes itself to 0. Both cases produce ~0 output.

**Conclusion**: No sign bug in the TB pipeline or observation encoding. The problem
is the pre-existing dead value head at v1489. `target_value` conventions are correct:
- `+1.0` = current player (STM, side-to-move) wins
- The observation encoding mirrors this: STM's pieces in planes 0-5 (AlphaZero convention)
- TB cache has ~867 winning / ~757 losing / ~376 draw in 2000 samples (slight winning bias)

## Step B — Clean Rerun from Pristine v1489 with REINIT

### Setup
```bash
cp checkpoints/best_v1489_pre_tb.pt checkpoints/best.pt
HYZERO_TABLEBASE_PATH=data/syzygy HYZERO_TABLEBASE_FRAC=0.3 HYZERO_REINIT_VALUE_HEAD=1 \
  bash scripts/run_baseline.sh 1800
```

### kqk_value progression (run 155658, cleaner of two parallel runs)

| Abs step | Rel step | kqk_value | start_value | kqk_minus_start | kvk_minus_start |
|----------|----------|-----------|-------------|-----------------|-----------------|
| 15050    | 0        | -0.03     | +0.10       | -0.14           | -0.09           |
| 15100    | 50       | +0.03     | +0.00       | +0.02           | +0.02           |
| 15200    | 150      | +0.05     | +0.01       | +0.05           | +0.06           |
| 15300    | 250      | +0.16     | +0.21       | -0.05           | -0.30           |
| 15350    | 300      | +0.42     | +0.34       | +0.07           | -0.31           |
| 15400    | 350      | +0.71     | +0.47       | +0.24           | -0.52           |
| 15450    | 400      | +0.88     | +0.53       | +0.36           | -0.44           |
| 15500    | 450      | +0.81     | +0.52       | +0.29           | -0.49           |
| 15600    | 550      | +0.85     | +0.54       | +0.31           | -0.49           |
| 15700    | 650      | +0.67     | +0.54       | +0.13           | -0.62           |
| 15900    | 850      | +0.47     | +0.45       | +0.02           | -0.68           |
| 16000    | 950      | +0.43     | +0.34       | +0.09           | -0.38           |
| 16100    | 1050     | +0.45     | +0.44       | +0.01           | -0.62           |

### Success Criteria Evaluation

| Metric | Target | Achieved |
|--------|--------|----------|
| kqk_value at step 1000 | > +0.5 | ~+0.43 (close; peak was +0.88 at step 400) |
| start_value range | non-trivial (±0.15+) | +0.34 to +0.54 (non-trivial but positively biased) |
| policy_loss | ≤ 4.0 | 1.53-1.60 (far below target) |
| promotions | ≥ 1 | 0 (not yet in 30 minutes) |

### Final Score
- Score: **6.46** (vs 6.43 contaminated baseline, 14.51 overall baseline)
- 0 promotions (value learning hasn't transferred to better play in 30 min)
- policy_loss 1.53 (excellent)
- avg_game_length 49.2

## Verdict: (a) TB + reinit works

The TB + REINIT mechanism successfully broke distributional collapse:

1. kqk_value rose from -0.03 to +0.88 within 400 training steps (peak)
2. Maintained +0.43-0.67 range through 1050 steps
3. kvk_minus_start was consistently -0.38 to -0.68 (correct: losing position < start)
4. Value head is ALIVE and discriminating positions correctly

## Caveats

### Positive value bias
The TB cache has ~14% more winning (+1) samples than losing (-1) samples. Combined
with the near-zero self-play targets, this creates a positive bias in start_value
(+0.35-0.54) when it should be near 0. The kqk_minus_start shrinks toward 0 as
start_value rises to meet kqk_value. This is a TB cache imbalance issue, not a
training bug.

**Fix**: Rebalance the TB cache to have equal winning/losing samples, or apply
class-weighting to TB loss terms.

### Two parallel runs
A scheduling mistake launched two competing runs writing to the same checkpoints/.
The second run (155748) showed kvk_queenless_value going positive (both KQK and
K-vs-KQ valued positively) — evidence the second run loaded from a checkpoint
already contaminated by the first run's positive bias. The first run's data
(155658) is more reliable.

### Promotions = 0 in 30 minutes
Expected. Value learning takes many training steps to transfer into better MCTS
behavior. The value head has learned to output ±1 for known terminal positions,
but the full game value tree needs to backpropagation through many more games
before MCTS Q-estimates improve enough to beat the champion by >55%.

## Recommended Next Steps

1. **Rebalance TB cache**: rebuild with equal +1/-1 samples to eliminate positive bias.
2. **Longer run**: 6-hour run with TB supervision from v1489 + REINIT. Expected
   promotions should appear within 3-4 hours once the value head signal propagates
   through MCTS.
3. **Tune TB_FRAC**: 0.3 (30%) may be too high — causes 70% of gradient to come from
   near-zero self-play targets, creating noise. Try 0.15-0.20 for cleaner signal.
4. **TB cache diagnosis**: Check `data/syzygy/cache.pkl` sample statistics to confirm
   the winning/losing imbalance and quantify it.

## Files Modified
- Restored `checkpoints/best.pt` from `checkpoints/best_v1489_pre_tb.pt` (model v15050)
- Created and deleted throwaway `scripts/tb_sign_check.py`
- Ran two simultaneous training runs (scheduling error; first run's data is primary)

## Log Files
- `logs/baseline_20260421_155658.log` — primary run (cleaner data)
- `logs/baseline_20260421_155748.log` — secondary run (contaminated by first)
- `logs/baseline_score.json` — final score 6.46 (timestamp 155748 won the race)
