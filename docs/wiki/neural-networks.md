# Neural Networks (MuZero)

Three networks on 8×8 boards, C=128 channels:

| Network | Input | Output | Role |
|---------|-------|--------|------|
| **h** | [B, 102, 8, 8] | [B, 128, 8, 8] | Encode observation → hidden state |
| **g** | [B, 131, 8, 8] | [B, 128, 8, 8] + [B] | Dynamics: next hidden + reward |
| **f** | [B, 128, 8, 8] | [B, 4096] + [B] | Policy logits + value |

Observation planes (102): 96 piece planes (8 positions × 12, current + 7 history slices, current-player perspective) + 6 game-state planes (4 castling + 1 en passant + 1 halfmove clock). Side-to-move is NOT a plane (Phase 3b removal); color is implicit in the perspective convention. Planes 0–5 = current player's pieces, 6–11 = opponent, rank-mirrored for Black to move. See `board-encoding.md` for full plane definitions.

## Network Shapes

```
h:  Conv2d(102→128, k=3, p=1) → BN → ReLU → 4×ResBlock → [B, 128, 8, 8]

g:  Conv2d(131→128, k=3, p=1) → BN → ReLU → 4×ResBlock
      state path:  [B, 128, 8, 8]
      reward path: Conv2d(128→1, k=1) → Flatten → Linear(64,1) → Tanh → [B]

f:  policy: Conv2d(128→2, k=1) → BN → ReLU → Flatten[B,128] → Linear(128,4096) → [B, 4096]
    value:  Conv2d(128→1, k=1) → BN → ReLU → Flatten[B,64] → Linear(64,64) → ReLU → Linear(64,1) → Tanh → [B]

ResBlock: Conv(C,C,3,p=1) → BN → ReLU → Conv(C,C,3,p=1) → BN + skip → [B, C, H, W]
```

## Inference Batch Methods (Python → Rust)

```
root_setup_batch(observations [B,102,8,8])
  → hidden [B,128,8,8], policies [B,4096] (softmax), values [B]

expand_leaf_batch(hidden [B,128,8,8], actions [B,3,8,8])
  → next_hidden [B,128,8,8], rewards [B], policies [B,4096] (softmax), values [B]
```

All arrays: `float32` numpy. Policies are post-softmax. Values tanh-bounded [-1, 1].

## Training (K-Step Unrolling)

```
Batch: observations [B,102,8,8], actions [B,K,3,8,8],
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

## Value-Head Failure Modes & Diagnosis

Four failure modes have been identified under β-blended outcome targets (2026-04-20, Mode 4 added 2026-04-21):

**Failure Mode 1: Material-Shaping Exploitation + β>0 → Shuffle Exploit**
Under `HYZERO_DISABLE_MATERIAL_SHAPING=0` with default `HYZERO_MATERIAL_SHAPING_SCALE=5`, non-checkmate outcome is `tanh(Δmaterial/5)`. A +3 material gap yields `tanh(0.6) ≈ 0.54`, above the 0.5 decisive threshold. Network learns to grab material then force repetition (e.g., rook shuffle a1↔b1). Seen in 2026-04-20 run #1 (v1099). **Prevention**: Use weak shaping (SCALE=20 or higher) to keep material-only outcomes below decisive threshold (~0.3 max), or disable shaping entirely and use conditional β.

**Failure Mode 2: Shaping OFF + β>0 → Signal Attenuation**
Without material shaping, non-checkmate outcomes are 0. Value target formula: `(1-β) × root_value + β × 0`. With untrained network, max target magnitude on decisive games is `β × 1 ≈ ±0.3` (at β=0.3). Value head learns a ±0.3 range, showing zero discrimination across mate-in-1, KQ-vs-K, and starting position. **Prevention**: Use conditional β (decisive games → β=1.0) or weak shaping to give drawn games non-zero targets.

**Failure Mode 3: Conditional β + Reinit → Sparse-Signal Decay**
Fixes enable value-head response at checkmate arrival (+0.35 measured), but decay back to 0 within 500 training steps. Root cause: 99.5% of batch samples are drawn games (target=0); only ~5 checkmates per 14k steps. The mechanism works; the signal rate doesn't. **Prevention**: Use weak shaping (SCALE=20) so drawn games receive `0.7 × root_value + 0.3 × tanh(Δ/20)` — continuous weak signal keeping drawn targets in [−0.3, +0.3] range, well below shuffle-exploit threshold.

**Failure Mode 4: Distributional Overfitting (2026-04-21)**
Value head (and reward head) fit training-distribution targets in aggregate but collapse to ~0 on out-of-distribution positions. Discovered via closed-form derivation from batch-aggregate stats: reward predictions appeared ±0.99 at checkmate arrival in training logs, but a probe on 90 held-out positions from eval_games.pgn showed [−0.008, +0.004] for all transitions, including actual mates. The network overfit to in-distribution self-play terminals and lost generalization to other model versions' checkpoint positions. **Diagnostic technique**: Per-checkmate-arrival canonical-position probe in train mode. If value head shows zero discrimination on KQ-vs-K or mate-in-1 positions while aggregate batch stats look alive, the network is distributionally collapsed. **Prevention**: External supervision (tablebase WDL labels, PGN corpus) to break the in-distribution overfitting loop.

**Recovery: External Supervision (2026-04-21)**

When the value head has collapsed to a narrow distribution (kqk_value ≈ 0 for 15k+ steps, 2026-04-21 evidence), self-play alone cannot restore it because the training distribution has drifted to shuffle patterns that don't contain decisive outcomes. Recovery requires injecting external ground-truth labels.

Validated approach: Syzygy tablebase supervision (3-4-5-man, WDL+DTZ labels) mixed into training batches at 45% fraction with masked padded-step loss (targets only at step 0, zero-out loss at steps 1–K), biased value-head reinit (+0.3 output bias), and balanced TB cache (equal +1/−1 samples). Evidence from 2-hour run (PID 1206967, 2026-04-21):
- kqk_value: sustained +0.85 (vs −0.012 baseline)
- **First promotion in eval ladder**: v15283 beat v15051 (win_rate=0.562)
- **2 actual checkmates detected**: Appeared in self-play (reward head responding)
- **Score 8.1572** (vs 6.05 pre-TB baseline, +2.11 delta; vs 14.51 absolute β=0.3 baseline)
- **White first-move diversity**: 77% concentrated → spread across ~8 openings
- **43% decisive self-play** (vs ~1% pre-TB)

Root cause of success: (1) Masked loss at padded steps prevents dilution of the ±1 TB signal across K-step pseudo-trajectories. (2) Biased reinit eliminates 50% stochasticity in initial response direction. (3) 45% TB fraction means every gradient step has ~45% ground-truth supervision. (4) Balanced cache prevents drift toward either attractor.

Remaining issue: kqk_value oscillates (peaks at +0.85 → drops to −0.34 → recovers). Root cause: replay buffer dilution. As self-play games accumulate, effective TB signal shrinks (TB buffer fixed, self-play buffer grows, dilution ∝ time). **Fix for next session**: Dedicated TB circular buffer (refreshed periodically from Syzygy cache) to maintain constant 45% proportion throughout training.

**Diagnostic technique**: Per-checkmate-arrival value probe. Parse training logs for `[cm_count]` lines; find step K where total_cm increments. Inspect `[start_value]` / `[kqk_value]` / `[kvk_queenless_value]` at K−100, K, K+100, K+500 to visualize value-head response amplitude and decay rate. If probe values stay in [−0.1, +0.1] range despite checkmate arrival, distributional collapse is active.

## Key Gotchas

1. **Policy**: Network outputs logits. Inference server applies softmax; training uses raw logits + CE.
2. **Value**: Tanh [-1, 1]. Currently predicts MCTS root_value (bootstrapped Q-estimates) + soft outcome blend (β=0.3 by default). **Critical**: Monitor canonical-position probes ([start_value], [kqk_value], [kvk_queenless_value] in logs). If these stay in [−0.1, +0.1] for >1000 steps, value head is dead. This indicates distributional collapse (overfitting to self-play).
3. **Reward**: Per-step (immediate), not cumulative. Real rewards come from trajectory — terminal reward only.
4. **Action encoding**: 4096 = 64×64, queen-default promotion. Underpromotion (4672) unimplemented. Actions are in current-player space; flipped to absolute board space at MCTS boundary (commit bb39db6). See [Board Encoding](board-encoding.md).
5. **Value not negated per ply** in backup — intentional (same sign across turns), verify during training.
6. **Reward loss K not K+1**: Only K reward terms (steps 1..K), policy/value have K+1 (steps 0..K). Divide reward loss by K.
7. **Gradient hook on g output**: `register_hook(lambda grad: grad * 0.5)` on dynamics OUTPUT for correct chained K-step scaling (MuZero Appendix G).
8. **torch.load deprecation**: Use `weights_only=False` explicitly in PyTorch 2.x to avoid FutureWarning.
9. **Loss weights at 1.0**: Keep `HYZERO_{POLICY,VALUE,REWARD}_LOSS_WEIGHT` at default 1.0. Amplifying (e.g., value_weight=5.0) destabilizes the multi-head feedback loop and regresses play despite better training loss.
10. **Reward head dead from class imbalance**: ~99% of reward targets are 0.0 (only terminal steps). MSE-optimal solution is 0.
11. **Value head outcome target conversion**: Game outcome is White-absolute (+1 White win, -1 Black win). When used as value target, must apply ply-flip to convert to the perspective of whoever is to move: `target = outcome * side_sign * (1.0 if ply_even else -1.0)`. Done automatically at `src/py/training.rs:136` during batch assembly.
12. **Underpromotion action spatial encoding is color-aware** (commit cc58506): Underpromo indices are color-agnostic at the action ID level, but `encode_action_spatial(action, white_to_move)` returns color-specific spatial planes. Under color augmentation, `encode_action_spatial(flip_action(a), flipped_color) == flip_action_planes(encode_action_spatial(a, original_color))` must hold. Regression test added; fix: use `encode_action_spatial_for_color(action, white_to_move)` when color matters.

## Related

- [Board Encoding](board-encoding.md) — current-player perspective convention, action flipping
- [MCTS & Self-Play](mcts-selfplay.md) — value/reward head dead analysis, replay buffer dynamics
- `docs/wiki/mistakes.md` — entries on dead value/reward heads and perspective consistency

## Related Files

- `python/hyzero/models/*.py` — network definitions
- `python/hyzero/training/trainer.py` — training loop (Task 25)
- `python/hyzero/inference/server.py` — batch inference (Task 26)
- `src/data/encoding.rs` — board → observation encoding
- `docs/TASKS_PYTHON.md` — task specs (Tasks 24-26)
