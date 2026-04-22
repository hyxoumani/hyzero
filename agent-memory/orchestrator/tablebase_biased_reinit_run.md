# Tablebase + Biased Value-Head Reinit 4-Hour Validation

**Date**: 2026-04-21
**Experiment**: Test whether biased value-head reinit (HYZERO_REINIT_VALUE_BIAS=0.3) fixes
the KQK value collapse seen in previous tablebase supervision runs.

## Hypothesis

When the value head is reinitialized (HYZERO_REINIT_VALUE_HEAD=1), setting the final
linear layer's bias to +0.3 (tanh(0.3) ≈ 0.296) should place the initial output in the
positive half-plane, preventing immediate collapse to a near-zero attractor dominated
by near-zero self-play value targets.

## Configuration

```
HYZERO_TABLEBASE_PATH=data/syzygy
HYZERO_TABLEBASE_CACHE_PATH=data/syzygy/cache_balanced.pkl
HYZERO_TABLEBASE_FRAC=0.2
HYZERO_REINIT_VALUE_HEAD=1
HYZERO_REINIT_VALUE_BIAS=0.3
```

Starting checkpoint: best_v1489_pre_tb.pt (model_version=15050)

## What Happened

The run was killed at step 15400 (350 steps post-reinit) after failing kill-gate 1.

Kill-gate 1: kqk_value went negative within 100 steps of reinit.

### kqk_value trajectory

| Step | kqk_value |
|------|-----------|
| 15050 (reinit) | +0.1756 |
| 15100 (+50 steps) | -0.1482 |
| 15150 | -0.1437 |
| 15200 | -0.0527 |
| 15250 | -0.1099 |
| 15300 | -0.2280 |
| 15350 | -0.0862 |
| 15400 | -0.0609 |

The bias offset of +0.3 was immediately washed out within 50 training steps.

### Root Cause: Target Imbalance

The training target histogram at step 15050-15400 showed consistently:
- ~1500 near-zero targets (-0.1 to +0.1 range)
- ~40 terminal targets (+0.9/-0.9 range, includes TB +1/-1 samples)

At HYZERO_TABLEBASE_FRAC=0.2, TB samples are ~20% of each batch. But the batch
contains ALL positions from sampled trajectories, not just terminal states. Since
games average ~47 plies, the near-terminal fraction is tiny.

The gradient from 1500 near-zero targets completely overwhelms the 40 terminal targets.
The bias of +0.3 is erased in <50 gradient steps.

## Result Table

| Metric | v1489 baseline (pre-TB run) | After biased-TB run (350 steps) |
|--------|-----------------------------|---------------------------------|
| kqk_value end | approx 0.0 (pre-reinit) | -0.06 (oscillating around 0) |
| start_value end | approx 0.0 | -0.085 |
| policy_loss end | 1.60 | ~1.62 (no improvement) |
| checkmate count | 0 | 0 |
| eval promotions | 0 | 0 (killed early) |
| White first move b1a3% | 77% (prev runs) | ~5% (current PGN) |
| Avg game length | ~47 plies | ~47 plies |

**Note on b1a3 improvement**: The selfplay_sample.pgn shows b1a3 dropped from 77% to
5% in the overall dataset. This improvement was achieved in a PREVIOUS run (likely the
tablebase_run2 from earlier today), not specifically from the biased reinit.
The current run only generated 3 games before being killed.

## Verdict: (c) Regression / Insufficient

Biased init failed. Kill-gate 1 triggered. Hypothesis refuted.

The fundamental problem is **class imbalance in training targets**:
- 20% TB fraction samples trajectories (not just terminal states)
- Near-zero positions dominate ~97% of the batch
- A scalar bias of +0.3 cannot survive gradient from 1500:40 imbalance ratio

## Root Cause Analysis

The value-head collapse is NOT primarily a bias problem. It's an information-density
problem:

1. TB supervision teaches only the terminal states (v=+1 for KQK winning position)
2. Each trajectory has ~47 positions, of which 1-2 are terminal
3. With 1534 total batch positions and ~20% TB fraction, we get ~30 TB positions
4. But each TB position comes from a different game trajectory, so the
   "near-terminal" positions in those trajectories are NOT labeled with TB values
5. The result: 30 TB signals vs ~1500 near-zero signals per gradient step

## Required Fix

To make TB supervision effective, the training loop must either:

**Option A (recommended)**: Sample TB positions at the TERMINAL level, not trajectory level.
Use a separate TB replay buffer with only terminal positions. Override the whole-game
value target for positions near a TB-confirmed win with exponentially decayed v=+1.

**Option B**: Massively increase HYZERO_TABLEBASE_FRAC (e.g., to 0.8) but this destroys
self-play quality by drowning game-context learning.

**Option C**: Implement a per-batch terminal weighting: multiply TB terminal loss terms
by a large constant (e.g., 20x) to compensate for class imbalance.

## Files Modified

- `python/hyzero/training/trainer.py` — biased reinit (commit e5d1a02)
- `python/tests/test_training.py` — 2 new tests for biased reinit

## Previous Runs for Reference

- tablebase_supervision_experiment_20260421.md — 4h run without biased init, same failure
- tablebase_4h_validation.md — earlier tablebase run
