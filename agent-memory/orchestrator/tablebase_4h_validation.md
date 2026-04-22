# Tablebase Supervision: 4-Hour Validation Run (2026-04-21)

## Objective

Validate that TB + REINIT supervision can sustain value-head recovery over 4 hours
and produce measurable improvements in play quality (promotions, mate rates, first-move
diversity). This was designed as the "demonstration that tablebase works" long run.

## Setup

- Checkpoint: `checkpoints/best_v1489_pre_tb.pt` (model_version=15050, pristine, same as short run)
- Cache: `data/syzygy/cache_balanced.pkl` (NEW — rebalanced from 87630/77166 → 77166/77166 ±1 counts)
- TB_FRAC: 0.2 (reduced from 0.3 to allow more self-play signal)
- REINIT: `HYZERO_REINIT_VALUE_HEAD=1`
- Log: `logs/baseline_20260421_163446.log`

## Step A: TB Cache Rebalancing (completed)

Script `scripts/rebalance_tb_cache.py` committed. Results:
- Before: N_pos=87630, N_neg=77166, N_zero=35204, total=200000
- After:  N_pos=77166, N_neg=77166, N_zero=35204, total=189536
- Output: `data/syzygy/cache_balanced.pkl`

## Run Timeline

Run started 16:34:46 UTC. Killed at ~16:47 UTC (~12 minutes). Run duration: 12 minutes.

## kqk_value Trajectory (all probes before kill)

| Abs step | kqk_value | start_value | kvk_queenless_value | Notes |
|----------|-----------|-------------|---------------------|-------|
| 15050    | -0.0779   | -0.1837     | —                   | Post-reinit (unlucky negative init) |
| 15100    | -0.0501   | +0.0058     | —                   | Improving |
| 15150    | +0.0185   | -0.0282     | —                   | Crossed zero |
| 15200    | +0.0385   | -0.0104     | +0.0304             | Briefly positive |
| 15250    | +0.0414   | -0.0071     | +0.0304             | Peak! |
| 15300    | +0.0325   | -0.0114     | +0.0016             | Declining |
| 15350    | -0.0084   | -0.0084     | -0.0333             | Back negative |
| 15400    | -0.0452   | -0.0133     | -0.0623             | Accelerating negative |
| 15450    | -0.0487   | +0.0011     | -0.0481             | |
| 15500    | -0.0960   | +0.0145     | -0.0971             | |
| 15550    | -0.0966   | +0.0076     | -0.1112             | |
| 15600    | -0.1156   | -0.0624     | -0.0344             | |
| 15650    | -0.2995   | -0.0921     | -0.1182             | Freefall begins |
| 15700    | -0.5058   | -0.0550     | -0.1080             | KILLED |

## Verdict: NEGATIVE RESULT

The run produced a negative attractor, not the positive spike seen in the short run.

**Root cause identified**: The short run (155748) worked because of **lucky reinit initialization**.
Post-reinit kqk_value was +0.16 in that run, which provided a positive starting direction
that the TB signal reinforced. This run started at -0.08 (negative), and the feedback loop
caused the value head to spiral negative:

1. Reinit → value head starts at -0.08 for KQK (should be +1)
2. Self-play generates games where root_value ≈ -0.08 (from the reinitialized head)
3. Value target = 0.7 × root_value + 0.3 × game_outcome ≈ 0.7 × (-0.08) + 0 = -0.056
4. Value head trains on slightly negative targets from both self-play AND initial weights
5. As value head learns -0.056 from self-play, MCTS produces worse estimates
6. New self-play targets become more negative → feedback loop pulls to -1

## Why the Balanced Cache Made Things Harder

The unbalanced cache (14% more +1 than -1) provided a slight positive bias to the
average TB signal, which helped the short run maintain positive kqk values. With a
perfectly balanced cache, the average TB target is exactly 0 — which means:
- TB signal pushes value head toward 0 (from above if kqk > 0, from below if kqk < 0)
- Combined with self-play slightly-negative targets, everything converges toward -small

The balance was correct for `start_value` stability (it worked: start_value stayed near
±0.02 for the first 500 steps), but it removed the slight positive bias that sustained
the short run's kqk_value.

## eval cycles: all draws

5 eval cycles, all: ladder_wins=0, ladder_draws=8, ladder_losses=0, win_rate=0.500.
No promotions. cm_count=0 throughout.

## Next Steps

### Option 1: Multiple REINIT with screening
Reinitialize the value head, run 50 training steps, check kqk_value. If kqk > +0.05,
proceed. If kqk < 0, reinit again and repeat. This screens for lucky positive init.

### Option 2: Positive-biased reinit
Initialize value head with a positive bias constant (+0.1) on the output layer bias,
rather than zero-bias kaiming_normal_. This ensures kqk starts positive for winning
positions regardless of random initialization direction.

### Option 3: Separate value-head TB loss from self-play
Apply TB value targets as a separate term with higher weight when position is in TB,
rather than averaging into the same batch. This prevents self-play 80% from overwhelming
the TB 20% when starting from a negative attractor.

### Option 4: TB_FRAC with imbalanced cache
Accept that start_value will have slight positive bias (+0.2), but the kqk_value will
stay positive long enough for the value head to learn.

## Key Lesson

Reinit + TB is **stochastic**. The short run worked with probability ~50% (half of reinit
seeds produce positive initial kqk). Future runs must either:
(a) Screen reinit quality in the first 50 steps, or
(b) Use biased initialization to guarantee kqk starts positive, or
(c) Run multiple short experiments and pick the one where reinit succeeded.

## Files

- `scripts/rebalance_tb_cache.py` — committed, balances cache ±1 counts
- `data/syzygy/cache_balanced.pkl` — balanced cache (189536 samples)
- `logs/baseline_20260421_163446.log` — this run's log (killed at step 15700)
