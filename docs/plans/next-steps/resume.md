# Autoresearch Resume Point — 2026-04-14

Paused mid-session. This file is the handoff for the next run.

## Current git tip
`0bf906f throughput: num_simulations 50 → 40` on branch `autoresearch/apr13`.

**Implemented but NOT measured.** Baseline run was killed mid-execution on user
request. Checkpoints were wiped before the kill, so the next baseline starts clean.

## Baseline to beat
**6.70 mean score** at commit `6faac55` (AdamW). Components:
- policy_loss: 2.31 mean
- decisive_ratio: 0.10-0.20 (high-variance)
- avg_game_length: ~105
- games/30min: 321
- training_steps/30min: ~5342

Score formula: `(8.55 - policy_loss) + (decisive_ratio * 10) - (avg_length / 100)`
Noise floor: ±1.0 points per single run. Claim wins only at >1.5 points or multi-rep.

## Kept experiments (4.78 → 6.70)
| Commit | Change | Score |
|---|---|---|
| `46c3d0d` | Batch 1: history + underpromotion + legal masking | 4.78 → 5.76 |
| `d407281` | Dirichlet α: 0.03 → 0.3 (AlphaZero chess value) | 4.13 → 6.78@900s |
| `704a056` | Eval reliability (games=10, sims=25, interval=25) | infra |
| `5be8fe0` | train_steps_per_game 8 → 16 | 4.85 → 5.36 |
| `6faac55` | Adam → AdamW | 5.36 → **6.70** |
| `e44583a` | infra: PyO3Backend reads hidden_channels from config | infra |

## Recently discarded (3 in a row this session)
| Commit | Change | Score | Reason |
|---|---|---|---|
| `3f600c6` | num_res_blocks 4 → 6 | 5.37 | throughput starved (321→199 games) |
| `90e9a54` | train_steps 16 → 32 | 5.84 | self-play starved (321→146 games) |
| `fb8c7cf` | weight_decay 1e-4 → 5e-4 | 5.67 | flatter policy (length 105→137) |

**Pattern**: Every simple parameter bump regresses because the baseline is well-tuned.
Capacity/intensity bumps starve the 1800s time budget. Regularization bumps flatten policy.

## Root-cause insight (captured this session)
Training log confirms `value=0.0000, reward=0.0006` throughout training.
**Value and reward heads are completely dead.** Chess has sparse rewards (terminal
only), and `root_value` bootstraps off the untrained value head → targets stay at 0 →
no gradient → dead head.

Past attempts to fix:
- Loss weight 10x (`b64875e`): no effect — weighting zero targets still gives zero.
- Hard outcome targets (`83b9244`): -4.6 score regression. Hard ±1 pollutes shared
  representation/dynamics networks, killed policy learning too.

## Recommended next experiments (in priority order)

### 1. Measure sims=40 (already implemented at tip)
Simplest unfinished task. Run `bash scripts/run_baseline.sh 1800`, compare to 6.70.
- If ≥7.0: keep, explore sims=30.
- If 5.5-7.0: rerun once for variance, then decide.
- If <5.5: revert.

### 2. Soft value-outcome blend (PRINCIPLED FIX for dead value head)

**Prerequisites**: Verify perspective consistency (see below).

In `src/py/training.rs` near line 98:
```rust
target_values[bi * kp1 + k] = step.root_value;
```
Change to a **small β blend**:
```rust
let outcome_sign = /* side-to-move sign at step k, derived from observation plane 101 */;
let outcome_target = game_outcome * outcome_sign;
target_values[bi * kp1 + k] = 0.9 * step.root_value + 0.1 * outcome_target;
```

Why β=0.1 and not the failed 83b9244 approach: past experiment fully replaced root_value
with hard ±1 — too strong, polluted shared networks. β=0.1 keeps soft MCTS Q as dominant
signal but injects small outcome-aligned gradient to kickstart the value head. Once value
head learns anything, root_value itself becomes informative and the feedback loop closes.

Side-to-move sign lives in observation plane 101 (past agent verified this). Need access
to `game_outcome` — currently stored only on last step's `reward` field, so TrainingSample
needs augmentation to carry the full-trajectory outcome to each step.

**BEFORE implementing**: Verify that observation plane 101 (side-to-move) correctly encodes
whose perspective the value target should be in. Document finding in mistakes.md.

**Complexity**: medium (touches Rust batch assembly + TrainingSample struct).
**Blast radius**: moderate (value head only, not policy directly).
**Expected impact**: value_loss > 0 (head starts learning), decisive_ratio may improve
(better value → stronger MCTS search → more decisive play).

### 3. LR schedule without warmup
Past LR experiment (`c5440f2`) failed because warmup-100 stole too many early steps.
Try **cosine decay from 1e-3 to 2e-4 with zero warmup** across estimated ~5000 steps.
One-file change in `python/hyzero/training/trainer.py` — wrap optimizer with
`CosineAnnealingLR`. Call `scheduler.step()` at end of each `train_batch()`.

### 4. Temperature schedule smoothing
Currently hard cutoff: temp=1.0 first 15 moves, 0.01 after. Switch to smooth exponential
decay. In `src/selfplay/game_task.rs` around line 99-103. Low risk, unexplored direction.

## DO NOT try again
- Any variant of "hard outcome targets for all steps" — `83b9244` pattern.
- `num_res_blocks ≥ 6` at current 1800s budget — `3f600c6` pattern.
- `train_steps_per_game ≥ 24` — `90e9a54` pattern (16 is the sweet spot).
- `hidden_channels 128` without first speeding up inference — symmetry collapse pattern.
- `HYZERO_SIMS ≥ 100` — symmetry collapse (`46c3d0d-sims100`).
- Recency-weighted replay with small decay — catastrophic forgetting (`003eaf9`).
- LR warmup-then-decay within 1800s — warmup eats too many steps.

## Checkpoint state
`checkpoints/` is empty (wiped before the killed baseline). Safe to start fresh.

## How to resume
```bash
# Option A: measure the already-implemented sims=40
bash scripts/run_baseline.sh 1800

# Option B: discard sims=40 and target value-head revival
git revert --no-edit 0bf906f
# then implement experiment #2 above

# Either way, log result to results.tsv:
# commit  score  policy_loss  decisive  avg_length  verdict  description
```

## Session progress (2026-04-15 autonomous)

- Dual-model eval infra implemented and merged (commits 8387fce..4a73c05, 725fef4)
- Value head soft outcome blend merged (commit 618ff46, 8856f20)
- Cross-run champion loading merged (commit d419f08)
- Metric formula corrected (commit 2a273d4): uses `promotions` count, not `max_champion_version`
- Env-var tunable loss weights added (commit 17dce57, opt-in, defaults preserve behavior)
- Cosine LR schedule added (commit a64e547, opt-in via HYZERO_LR_SCHEDULE=cosine)
- Reward soft-blend γ added (commit 294e63e, opt-in via HYZERO_REWARD_OUTCOME_GAMMA)

**11 experiments completed. Session CLOSED.**

| # | Config | Score |
|---|---|---|
| 1 | β=0.1 defaults | 6.76 |
| 2 | β=0.2 defaults | 8.33 |
| 3 | β=0.3 defaults | **11.63** ← winner |
| 4 | β=0.4 defaults | 6.80 |
| 5 | β=0.5 defaults | 8.07 |
| 6 | β=0.3 + value_weight=5 | 4.84 |
| 7 | β=0.3 + num_sims=60 | 8.01 |
| 8 | β=0.3 + eval_sims=15 | 5.69 |
| 9 | β=0.3 + games_per_side=6 | 5.10 |
| 10 | β=0.3 + LR_cosine(T_max=5000) | 6.47 |
| 11 | β=0.3 + reward_γ=0.1 | 6.81 |

**Findings:**
- β=0.3 is Pareto-optimal; single-knob tuning space exhausted
- Every deviation from β=0.3 defaults regressed ("fast training / low promotions" anti-pattern)
- Root cause: MCTS depends on value head quality; amplifying training speed without matching MCTS quality degrades self-play data (garbage-in/garbage-out)
- Baseline established: **11.63** (commit 294e63e)

**Next recommended direction:** reward head fix (sparse bootstrap targets analogous to value β) combined with architecture scaling (capacity/depth). Single-knob changes in the current architecture appear unable to beat β=0.3 defaults. A combined fix (reward-blend + capacity + longer eval window) may unlock the next score range.
