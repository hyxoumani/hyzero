# Tablebase Masked-Loss 3h Validation Run (2026-04-21)

## Objective

Validate commit c4007a2 ("training: mask value/policy loss at padded steps for TB samples").
The masking fix prevents the 5-step zero-padding of TB samples from diluting the step-0
±1 supervision signal by 5x. This was the fourth iteration of TB supervision experiments.

## Configuration

```
HYZERO_TABLEBASE_PATH=data/syzygy
HYZERO_TABLEBASE_CACHE_PATH=data/syzygy/cache_balanced.pkl
HYZERO_TABLEBASE_FRAC=0.3
HYZERO_REINIT_VALUE_HEAD=1
HYZERO_REINIT_VALUE_BIAS=0.3
```

- Starting checkpoint: `checkpoints/best_v1489_pre_tb.pt` (model_version=15050, pristine)
- Log: `logs/baseline_20260421_171517.log`
- Process PID: 1174075
- Started: 17:15 UTC, Killed: 18:06 UTC (~51 minutes runtime, killed by gate 2)

## kqk_value Trajectory — Every 500 Steps

| Step | kqk_value | kvk_queenless_value | start_value | Notes |
|------|-----------|---------------------|-------------|-------|
| 15050 | -0.1128 | +0.0068 | +0.3770 | Post-reinit (start with positive bias) |
| 15550 | +0.2513 | +0.3023 | -0.0789 | Crossed zero, recovering |
| 16050 | +0.3527 | +0.0592 | +0.1106 | Gate 1: +0.35, above kill (>+0.2) |
| 16550 | +0.3324 | -0.0333 | +0.0551 | Stable in +0.25-+0.40 band |
| 17050 | +0.3949 | +0.0563 | +0.1663 | Rising trend |
| 17550 | +0.4776 | +0.1017 | +0.0955 | Near peak |
| 18050 | +0.2754 | +0.1071 | +0.0703 | Gate 2: +0.28, KILL triggered |

## Full kqk_value Trajectory

```
step=15050 v=-0.1128   step=15100 v=-0.3403   step=15150 v=-0.3269   step=15200 v=-0.1250
step=15250 v=+0.1351   step=15300 v=+0.3038   step=15350 v=+0.1315   step=15400 v=+0.3058
step=15450 v=+0.2076   step=15500 v=+0.3509   step=15550 v=+0.2513   step=15600 v=+0.2999
step=15650 v=+0.2770   step=15700 v=+0.3970   step=15750 v=+0.2795   step=15800 v=+0.2954
step=15850 v=+0.3760   step=15900 v=+0.2759   step=15950 v=+0.3351   step=16000 v=+0.3320
step=16050 v=+0.3527   step=16100 v=+0.3716   step=16150 v=+0.2586   step=16200 v=+0.2133
step=16250 v=+0.2703   step=16300 v=+0.3569   step=16350 v=+0.3996   step=16400 v=+0.3362
step=16450 v=+0.3299   step=16500 v=+0.3536   step=16550 v=+0.3324   step=16600 v=+0.3659
step=16650 v=+0.3570   step=16700 v=+0.3831   step=16750 v=+0.4571   step=16800 v=+0.4300
step=16850 v=+0.2937   step=16900 v=+0.2098   step=16950 v=+0.2985   step=17000 v=+0.4265
step=17050 v=+0.3949   step=17100 v=+0.3741   step=17150 v=+0.4441   step=17200 v=+0.2711
step=17250 v=+0.3040   step=17300 v=+0.3631   step=17350 v=+0.3632   step=17400 v=+0.3527
step=17450 v=+0.3730   step=17500 v=+0.5387   step=17550 v=+0.4776   step=17600 v=+0.4321
step=17650 v=+0.3822   step=17700 v=+0.4066   step=17750 v=+0.3945   step=17800 v=+0.3450
step=17850 v=+0.4369   step=17900 v=+0.3361   step=17950 v=+0.3898   step=18000 v=+0.2519
step=18050 v=+0.2754   step=18100 v=+0.2582
```

## Result Table

| Metric | v1489 baseline | After 51-min masked-TB run |
|--------|----------------|---------------------------|
| `[kqk_value]` final | -0.012 | +0.26 (at kill; peak was +0.54) |
| `[kqk_value]` peak | -0.012 | +0.5387 (step 17500) |
| `[start_value]` final | ≈ 0 | +0.07 (near-zero, good) |
| `[kvk_queenless_value]` final | N/A | +0.11 |
| policy_loss final | ~1.60 | 1.93 (regressed — expected during value reinit) |
| consistency_loss final | ~0.07 | 0.084 |
| checkmate count total | 0 | 0 |
| eval promotions | 0 | 0 |
| eval decisive ratio | 0.0 | 0.0 (all 25 cycles were 0W-8D-0L) |
| avg game length | ~47 | 52.4 (slightly longer) |
| estimated score | 14.51 (old baseline) | ~6.09 (no promotions) |
| decisive selfplay games | rare | 3/7 v15xxx games (43%!) |

## Kill Gate Analysis

**Gate 1** (step 16050, +30 min from start): kqk_value = +0.35 → PASS (>+0.2, not killed)
- Clear: was above kill threshold of +0.2

**Gate 2** (step 18050, +90 min): kqk_value = +0.28 → KILL TRIGGERED
- Condition: "dropped below +0.3 from a peak"
- Peak was +0.54 (step 17500), then 3 consecutive readings below +0.3 (steps 18000-18100)
- Rolled 5-step average had declined to +0.302 at step 18100
- Kill was correct: signal clearly decaying, not noise

## Self-Play First-Move Distribution

Total games in PGN at run end: 39 games (7 from v15xxx models)

| Move | Count (v15xxx games, n=7) |
|------|--------------------------|
| h2h4 | 3 |
| b1c3 | 2 |
| b2b4 | 1 |
| g2g4 | 1 |

Compare: Pre-TB runs were 77% b1a3. The b1a3 bias appears fully broken. First-move
diversity is now spread across many openings (no single move above ~43% even in small sample).

Note: The broader PGN (39 games) shows even more diversity — g2g4 (5), d2d3 (5),
h2h4 (4), d2d4 (4) — confirming the policy is exploring more openings.

## Decisive Game Analysis

Of the 7 selfplay games from v15xxx models (this run):
- 3 games ended 1-0 (decisive White win) — 43% decisive rate
- 0 games ended 0-1
- 4 games ended 1/2-1/2

The decisive games (v15050, v15068, v15129) had avg game lengths of ~24 moves —
shorter and more decisive than earlier models. This is a significant improvement.

## What the Masked Loss Fix Did (vs Prior Iterations)

The masking fix prevented step-0 TB signal dilution by zero-padding. Results compared
to prior iterations:

| Iteration | kqk peak | kqk stability | Kills |
|-----------|----------|---------------|-------|
| v1 (unbiased reinit) | +0.88 | Collapsed to 0 | Not killed (lucky run) |
| v2 (unbiased reinit, different seed) | -0.57 | Negative attractor | Would have been killed |
| v3 (biased reinit, balanced cache) | -0.30 | Negative | Gate 1 kill |
| v4 (masked loss, this run) | +0.54 | Oscillating +0.21 to +0.54 | Gate 2 kill |

The masked loss fix achieved two things:
1. Stable positive kqk_value over 1050 steps (vs immediate collapse in v3)
2. Peak of +0.54 (close to gate 2 threshold of +0.5)

## Why Gate 2 Failed

After peak at step 17500, kqk_value declined and fell below +0.3 at step 18000-18100.
Rolling 5-step average dropped from +0.447 (peak) to +0.302 (step 18100).

Hypothesis: The self-play data generated while kqk was high (+0.40-0.54) gets added
to the replay buffer with near-zero value targets (no TB position in most positions).
When the model processes this batch-of-good-games-with-zero-targets, it partially
"unlearns" the TB signal. This is the same class imbalance problem as before, just
at a smaller scale now that masking handles the padding.

## Verdict: (b) Partial Success

The masking fix clearly improved over prior iterations:
- Value head stayed positive for 1000+ steps
- kqk_value reached +0.54 (close to gate 2 threshold of +0.5)
- First-move diversity improved dramatically
- 43% decisive selfplay game rate for v15xxx models

But the run was killed by gate 2. The value learning has not yet transferred
to promotions (eval ladder still all draws).

## Root Cause of Gate 2 Failure

The oscillation pattern (+0.25 to +0.54) with gradual mean decline suggests:
1. TB signal strengthens kqk every time a TB position is sampled (pull toward +1)
2. Self-play positions with near-zero targets weaken kqk (pull toward 0)
3. The balance tips toward 0 over time as the replay buffer grows with non-TB games

The fix needed is to **maintain the TB signal's proportion in the replay buffer**
as the buffer grows, e.g., by keeping a separate high-priority TB replay buffer
that always contributes a fixed fraction to each training batch.

## Recommended Next Steps

### Option A (Most Promising): Run longer with same config (3h+)

The kqk_value had a clear upward trend from step 15250 (+0.13) to step 17500 (+0.54),
roughly +0.007 per 50-step interval on the rising segments. If the run continued:
- Step 18500 might have recovered to +0.35-0.40
- Step 20000+ might have sustained the gate 2 pass

A 6-12 hour run from the same checkpoint could allow promotions to emerge.

### Option B (Stronger): Increase TB_FRAC to 0.4-0.5

The 30% TB fraction is still being overwhelmed by 70% near-zero self-play signal.
Increasing to 0.4 would improve the kqk/start balance. Risk: slower policy learning.

### Option C (Most Targeted): Per-sample TB weight boost

Add a per-sample loss multiplier for TB rows (e.g., 3x). This is lightweight and
directly addresses the class imbalance without changing FRAC.

### Option D (Best Engineering): Separate TB replay buffer

Maintain a circular TB-only buffer (size=1000). Each training batch draws:
- 70% from self-play replay
- 30% from TB buffer

This guarantees consistent TB signal regardless of replay buffer growth.

## Files

- `python/hyzero/training/trainer.py` — masked loss fix (commit c4007a2)
- `logs/baseline_20260421_171517.log` — this run's log
- `logs/tablebase_run4b_masked_fixed.log` — outer wrapper log
- Checkpoint at kill: `checkpoints/best.pt` (version v15243+, preserved)
