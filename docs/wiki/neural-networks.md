# Neural Networks (MuZero)

Three networks on 8×8 boards, C=64 channels:

| Network | Input | Output | Role |
|---------|-------|--------|------|
| **h** | [B, 19, 8, 8] | [B, 64, 8, 8] | Encode observation → hidden state |
| **g** | [B, 67, 8, 8] | [B, 64, 8, 8] + [B] | Dynamics: next hidden + reward |
| **f** | [B, 64, 8, 8] | [B, 4096] + [B] | Policy logits + value |

Observation planes (19): 6 white pieces + 6 black pieces + 4 castling rights + en passant + side to move + halfmove clock.

## Network Shapes

```
h:  Conv2d(19→64, k=3, p=1) → BN → ReLU → 4×ResBlock → [B, 64, 8, 8]

g:  Conv2d(67→64, k=3, p=1) → BN → ReLU → 4×ResBlock
      state path:  [B, 64, 8, 8]
      reward path: Conv2d(64→1, k=1) → Flatten → Linear(64,1) → Tanh → [B]

f:  policy: Conv2d(64→2, k=1) → BN → ReLU → Flatten[B,128] → Linear(128,4096) → [B, 4096]
    value:  Conv2d(64→1, k=1) → BN → ReLU → Flatten[B,64] → Linear(64,64) → ReLU → Linear(64,1) → Tanh → [B]

ResBlock: Conv(C,C,3,p=1) → BN → ReLU → Conv(C,C,3,p=1) → BN + skip → [B, C, H, W]
```

## Inference Batch Methods (Python → Rust)

```
root_setup_batch(observations [B,19,8,8])
  → hidden [B,64,8,8], policies [B,4096] (softmax), values [B]

expand_leaf_batch(hidden [B,64,8,8], actions [B,3,8,8])
  → next_hidden [B,64,8,8], rewards [B], policies [B,4096] (softmax), values [B]
```

All arrays: `float32` numpy. Policies are post-softmax. Values tanh-bounded [-1, 1].

## Training (K-Step Unrolling)

```
Batch: observations [B,19,8,8], actions [B,K,3,8,8],
       target_policies [B,K+1,4096], target_values [B,K+1], target_rewards [B,K+1]

Step 0:  h0 = h(obs); p0,v0 = f(h0)
         loss += CE(p0, target_p[:,0]) + MSE(v0, target_v[:,0])
Steps k: hk,rk = g(h_{k-1}, act[:,k-1]); pk,vk = f(hk)
         loss += CE(pk, target_p[:,k]) + MSE(vk, target_v[:,k]) + MSE(rk, target_r[:,k])/K
Total loss = sum / (K+1). Dynamics gradient scaled 1/K.
```

## Canonical MuZero Value Target for Board Games

In the MuZero paper (Schrittwieser et al. 2020), the n-step value bootstrap is:
```
z_k = sum_{j=0}^{n-1} γ^j * r_{k+j} + γ^n * v(s_{k+n})
```

For board games (chess, shogi, Go), the paper sets **n = ∞** and **γ = 1**. This eliminates the value term and collapses to:
```
z_k = sum of all remaining rewards = game outcome
```

**Our current approach** (value target = MCTS root_value) is **not canonical**. We use MCTS Q-estimates as targets, which should theoretically improve training speed (shorter bootstrap, less variance). However, in practice this creates a self-referential loop: when the value head is untrained, root_value ≈ 0, so targets are ≈ 0, producing no gradient. This is a known failure mode in self-supervised learning.

**Comparison**:
- **AlphaZero**: Value target = game outcome (like canonical MuZero). Deterministic board state allows terminal detection; outcome enters backup directly.
- **Canonical MuZero (Atari)**: Value target = game outcome (n=∞, γ=1). No terminal detection in latent space.
- **Our implementation**: Value target = MCTS root_value. Hypothesis: iterative refinement via bootstrapping should work, but empirically fails due to the self-referential loop.

**Proposed fix**: Soft blend of root_value and outcome (e.g., β=0.1: `0.9 * root_value + 0.1 * game_outcome * side_sign`). This injects small outcome-aligned gradient to break the bootstrap loop while preserving the soft MCTS Q signal. Once the value head learns anything, root_value becomes informative and the feedback loop closes. A prior hard-outcome attempt (β=1.0) regressed -4.6 points, likely due to shared-network pollution.

## MCTS as Policy Improvement Operator

MCTS tree search is a policy improvement operator in the sense that better play (higher expected return) should increase visit counts and refine the policy. However, **this requires Q-estimates to be informative**. When Q ≈ 0 everywhere (as with our dead value head), PUCT reduces to:
```
PUCT(s, a) = P(a) * sqrt(N(s)) / (1 + N(a))
```

Without the Q term, selection is noise plus prior bias. Visit counts approximate the prior distribution, not the improved policy. Policy loss may decrease (network memorizes which moves to avoid), but the policy doesn't *improve* — it self-imitates. This explains the "hollow learning" pattern: low loss, but evaluations show unchanged or degraded play.

## Value Head Bootstrap Crisis and Material-Signal Recovery

**Root cause (2026-04-15)**: Self-play hit a 99% cap-draw rate. Games reaching the 300-move limit wrote `outcome = 0.0` into the replay buffer. With β=0.3 blend (`target = 0.7 * mcts_root_value + 0.3 * outcome`), every value-loss target was ≈ 0, training the value head to output 0 everywhere. Timid play (all moves looked equally bad) meant more games hit the cap, more zeros → closed-loop collapse. Classic positive-feedback bootstrap failure.

**Diagnostic signature**: 
- `avg_game_length ≈ 300` (games hitting cap)
- `value_loss → 0` (training on zero targets)
- `promotions = 0` (no play improvement)
- `policy_loss decreasing` (false signal — network memorizing bad move priors instead of learning)

**Fix (commit 1846b78)**: Two-part surgical intervention in `src/selfplay/game_task.rs`:

1. **Material-at-cap**: Replace synthetic `outcome = 0` with `outcome = tanh(Δmaterial / 5.0)`, where Δmaterial = white_material − black_material (standard piece values: P=1, N=3, B=3, R=5, Q=9). Preserves White-absolute sign convention; trainer at `src/py/training.rs:136` applies ply-flip to convert to step perspective.

2. **Adjudication**: New state machine inside game loop. If `|Δmaterial| ≥ HYZERO_ADJ_THRESHOLD` sustained for `HYZERO_ADJ_PLIES` consecutive plies, end game early with `outcome = sign(Δmaterial)`. Counter resets to 0 if diff drops below threshold. Env vars allow smoke testing at lower thresholds without rebuild.

**Outcome**: Avg game length drops 4x (165 → 40 moves), games become decisive, material-correlated targets flow into value head, value loss becomes non-zero and meaningful. As value head learns material, adjudication fires on fewer trajectories, material proxy naturally fades. Curriculum learning transition from synthetic to real signals.

**Expected behavior**: This is a **primitive version of KataGo's auxiliary target approach** — extract more signal per trajectory by replacing synthetic-draw `outcome=0` with position-correlated material proxy. Works because material dominance is a strong early-game signal; once the network learns material, real terminal outcomes dominate and the proxy gracefully decays in importance.

## Outcome Blend Protocol (β Parameter)

The value-outcome blend coefficient β controls the mix of MCTS Q-estimates and game outcomes in value targets. This is the highest-leverage knob in the current pipeline.

**Established optimum**: β=0.3 (commit 294e63e, 2026-04-15)
- Produces sustained promotions (4 in 5 eval cycles, 80% rate)
- Score 11.63 — peak of entire autoresearch program
- Games average 151.6 moves (longer = more exploration = better training data)
- Policy loss 3.40 (healthy — not too fast convergence)

**Protocol for β changes**:
```bash
# Test a new β value
rm -f checkpoints/best*.pt  # Fresh start, no prior ladder state
HYZERO_VALUE_OUTCOME_BETA=0.5 bash scripts/run_baseline.sh 1800
```

**Why fresh start matters**: If `best.pt` from a prior β setting exists, the next run starts with biased champion (trained on different blend). Always delete checkpoints between β experiments for fair comparison.

**Deviations regress**:
- β < 0.3 (e.g., β=0.2): Too little outcome signal. Over-relies on noisy Q-estimates. Result: 2 promotions vs 4.
- β > 0.3 (e.g., β=0.4, 0.5): Destabilizes training. Model converges faster to poor local optima. Result: 1–2 promotions, challenger loses to Random. Policy loss lower but play regresses (closed-loop paradox).

**Related env vars**:
- `HYZERO_VALUE_LOSS_WEIGHT` (default 1.0) — DO NOT increase above 1.0. See "Value Loss Weight Overshoot" mistake entry for why amplifying weight creates feedback loop instability.
- `HYZERO_POLICY_LOSS_WEIGHT` (default 1.0) — keep at 1.0
- `HYZERO_REWARD_LOSS_WEIGHT` (default 1.0) — keep at 1.0 (reward head is already class-imbalanced)
- `HYZERO_LR_SCHEDULE` — leave empty (no schedule). Cosine schedule tested, didn't help.
- `HYZERO_REWARD_OUTCOME_GAMMA` — leave at default (no soft blending of outcome). γ=0.1 test regressed once at β=0.3.

## Loss Weight Tuning — Multi-Head Feedback Loop

Loss weights (HYZERO_{POLICY,VALUE,REWARD}_LOSS_WEIGHT) default to 1.0 and should stay near that. A 2026-04-15 experiment boosted value_loss_weight to 5.0 expecting faster value head training (since value loss was ~60x smaller than policy loss). Result: catastrophic regression from 11.63 to 4.84, with 0 promotions. Notably, policy loss achieved a new best (2.70 vs baseline 3.40), yet the challenger **lost to Random** at eval cycles 3–4.

**Root cause**: MuZero training is a **closed-loop multi-head system**. The value head's quality directly controls which moves MCTS expands (via PUCT selection). When 5x amplification made value estimates oscillate wildly early in training, MCTS made poor move selections, generating low-quality training data. The policy head then learned on garbage targets (how to avoid costly moves in positions that shouldn't have existed). Policy loss *appeared good* locally (network faithfully memorized bad move labels), but global play quality collapsed because the data generator (MCTS under poor value guidance) was corrupt from the start.

**To increase value signal**: Prefer tuning the outcome blend coefficient β instead. For example, use β=0.5 (soft 50/50 outcome–Q-estimate target) rather than increasing the loss weight. This scales the target without amplifying gradient instability in the closed-loop system.

## Key Gotchas

1. **Policy**: Network outputs logits. Inference server applies softmax; training uses raw logits + CE.
2. **Value**: Tanh [-1, 1]. Currently predicts MCTS root_value (bootstrapped Q-estimates) + soft outcome blend (β=0.3 by default).
3. **Reward**: Per-step (immediate), not cumulative. Real rewards come from trajectory — terminal reward only.
4. **Action encoding**: 4096 = 64×64, queen-default promotion. Underpromotion (4672) unimplemented.
5. **Value not negated per ply** in backup — intentional (same sign across turns), verify during training.
6. **Reward loss K not K+1**: Only K reward terms (steps 1..K), policy/value have K+1 (steps 0..K). Divide reward loss by K.
7. **Gradient hook on g output**: `register_hook(lambda grad: grad * 0.5)` on dynamics OUTPUT for correct chained K-step scaling (MuZero Appendix G).
8. **torch.load deprecation**: Use `weights_only=False` explicitly in PyTorch 2.x to avoid FutureWarning.
9. **Loss weights at 1.0**: Keep `HYZERO_{POLICY,VALUE,REWARD}_LOSS_WEIGHT` at default 1.0. Amplifying (e.g., value_weight=5.0) destabilizes the multi-head feedback loop and regresses play despite better training loss.
10. **Reward head dead from class imbalance**: ~99% of reward targets are 0.0 (only terminal steps). MSE-optimal solution is 0.
11. **Value head outcome target conversion**: Game outcome is White-absolute (+1 White win, -1 Black win). When used as value target, must apply ply-flip to convert to the perspective of whoever is to move: `target = outcome * side_sign * (1.0 if ply_even else -1.0)`. Done automatically at `src/py/training.rs:136` during batch assembly.

## Related

- [MCTS & Self-Play](mcts-selfplay.md) — value/reward head dead analysis, replay buffer dynamics
- `docs/wiki/mistakes.md` — entries on dead value/reward heads and perspective consistency

## Related Files

- `python/hyzero/models/*.py` — network definitions
- `python/hyzero/training/trainer.py` — training loop (Task 25)
- `python/hyzero/inference/server.py` — batch inference (Task 26)
- `src/data/encoding.rs` — board → observation encoding
- `docs/TASKS_PYTHON.md` — task specs (Tasks 24-26)
