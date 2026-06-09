# Neural Networks (MuZero)

Three networks on 8×8 boards, `hidden_channels = 128`, `num_res_blocks = 4`
(`python/hyzero/config.py`):

| Network | Input | Output | Role |
|---------|-------|--------|------|
| **h** (representation) | `[B, 102, 8, 8]` | `[B, 128, 8, 8]` | Encode observation → hidden state |
| **g** (dynamics) | `[B, 128, 8, 8]` + `[B, 3, 8, 8]` action planes | `[B, 128, 8, 8]` + reward `[B, 1]` | Next hidden state + immediate reward |
| **f** (prediction) | `[B, 128, 8, 8]` | policy `[B, 4672]` + value `[B, 1]` | Policy logits + value |

The policy head emits **4672** logits (`num_actions` = 4096 base + 576
underpromotion), not 4096 — see [Board Encoding](board-encoding.md). Observation
planes (102) are described there too.

## Network Shapes

```
h:  Conv2d(102→128, k=3, p=1) → BN → ReLU → 4×ResBlock → [B, 128, 8, 8]
    + SimSiam projector/predictor heads (training only)

g:  cat(hidden[B,128,8,8], action_planes[B,3,8,8]) → Conv2d(131→128, k=3, p=1) → BN → ReLU → 4×ResBlock
      next_hidden: [B, 128, 8, 8]
      reward:      Conv2d(128→1, k=1) → Flatten → Linear(64,1) → Tanh → [B, 1]

f:  policy: Conv2d(128→2, k=1) → BN → ReLU → Flatten → Linear(128, 4672) → [B, 4672]
    value:  Conv2d(128→1, k=1) → BN → ReLU → Flatten → Linear(64,128) → ReLU → Linear(128,1) → Tanh → [B, 1]

ResBlock: Conv(C,C,3,p=1) → BN → ReLU → Conv(C,C,3,p=1) → BN + skip → ReLU
```

(`python/hyzero/models/{representation,dynamics,prediction,common}.py`.)

The representation network also exposes a SimSiam-style `project()` (→ proj_dim
256) and `predict()` head used only by the training-time consistency loss
(EfficientZero, Ye et al. NeurIPS 2021). They are not part of inference.

## Inference Batch Methods (Python ← Rust)

`python/hyzero/inference/server.py`, all under `torch.no_grad()`, returning
float32 numpy:

```
root_setup_batch(observations [B,102,8,8], legal_masks [B,4672] bool | None)
  → hidden [B,128,8,8], policies [B,4672] (masked softmax), values [B]

expand_leaf_batch(hidden [B,128,8,8], actions [B,3,8,8])
  → new_hidden [B,128,8,8], rewards [B], policies [B,4672] (softmax), values [B]
```

`root_setup_batch` masks illegal logits to `-inf` before softmax when
`legal_masks` is supplied; `expand_leaf_batch` runs in latent space with no mask
(no real board to derive legality from). Values are tanh-bounded in `[-1, 1]`.

> Note: some docstrings in `server.py` still read `[B, 103, ...]` / `[B, 4096]`;
> the live config is 102 planes and 4672 actions.

## Training Loop (K-step unroll)

`Trainer.train_batch(batch)` in `python/hyzero/training/trainer.py`. Default
`unroll_k = 5`, `train_batch_size = 256` (env `HYZERO_TRAIN_BATCH_SIZE`),
assembled in `src/py/training.rs::assemble_batch_arrays`.

```
batch: observations [B,K+1,102,8,8], actions [B,K,3,8,8],
       target_policies [B,K+1,4672], target_values [B,K+1],
       target_rewards [B,K+1], legal_masks [B,4672] (root only),
       is_tablebase [B] (Python-only, popped before tensor conversion)

Step 0:  h0 = h(obs[:,0]); p0,v0 = f(h0)
         policy_loss += CE(p0, tgt_p[:,0], legal_mask);  value_loss += MSE(v0, tgt_v[:,0])
Steps k: hk,rk = g(h_{k-1}, act[:,k-1]);  hk.register_hook(grad → grad*0.5)
         pk,vk = f(hk)
         policy/value/reward losses += MSE/CE; TB rows masked at k≥1 (see below)

avg_policy/value_loss = total / (K+1);  avg_reward_loss = total / K
total_loss = w_p·avg_policy + w_v·avg_value + w_r·avg_reward + w_c·consistency
```

Per-network loss weights default to 1.0 (`HYZERO_{POLICY,VALUE,REWARD}_LOSS_WEIGHT`).
The dynamics-output gradient is scaled ×0.5 at each boundary (MuZero Appendix G).

**SimSiam consistency loss** (`HYZERO_CONSISTENCY_LOSS_WEIGHT`, default 0.5):
for each k in 1..K, `g`'s latent is projected+predicted and matched against
`h(obs_k)` (stop-grad target) via cosine similarity. Gives `g` a direct training
signal independent of `f`. Tablebase rows are excluded (their step-1..K obs are
zeros).

## Target Construction & Outcome Blend

`src/py/training.rs::assemble_batch_arrays` builds the value/reward targets:

- `game_outcome` is White-absolute (+1 White win, −1 Black win). It is converted
  to the step-k side perspective via `ply_flip` (alternates per ply) and the
  root side sign, then negated under color-flip augmentation.
- **Value target**: `(1 − β)·root_value + β·outcome_in_step_perspective`, where β
  comes from `HYZERO_VALUE_OUTCOME_BETA`. With `HYZERO_CONDITIONAL_BETA=1`,
  decisive (non-draw) games use β=1.0 (full outcome) while drawn games keep the
  configured β.
- **Reward target**: `(1 − γ)·reward + γ·outcome`, γ from `HYZERO_REWARD_OUTCOME_GAMMA`.

The reward is only non-zero on the trajectory's terminal step; the same POV flip
is applied so the reward head sees a consistent sign convention.

## Tablebase / Mate Supervision

When `HYZERO_TABLEBASE_PATH` is set, the trainer loads a `TablebaseCache`
(`python/hyzero/data/tablebase.py`) and mixes supervised rows into each batch via
`_maybe_mix_tb_samples`. TB rows carry real signal only at step 0 (and a mating
reward at step 1); at steps k≥1 their padded-zero targets are masked out so they
do not dilute the ±1 ground-truth supervision. `HYZERO_REINIT_VALUE_HEAD`
reinitializes the value head on checkpoint load.

## Weights, Checkpoints, Pretraining

- `get_weights()` → bytes (`torch.save` of h/g/f state dicts) for the inference
  server. `load_weights(bytes)` deserializes with `weights_only=False`.
- `save_checkpoint(path, eval_metrics)` persists h/g/f + optimizer + lr_scheduler
  + `model_version`. `load_checkpoint(path)` restores them, tolerating an
  optimizer param-group mismatch (e.g. pretrain checkpoints that froze `f`).
- Pretraining scripts: `scripts/pretrain_dynamics.py` (SimSiam dynamics warm-start
  → `pretrain_dynamics.pt`) and `scripts/pretrain_on_mates.py` (mate-in-1 puzzles
  → `mate_pretrained.pt`, the default resume point — see [Baseline Scoring](baseline-scoring.md)).

## Diagnostics

`train_batch` emits (via `os.write` to fd 1, bypassing PyO3 stdout redirection)
per-step `[val_stats]`, `[reward_stats]`, `[policy_stats]`, and every 50 calls
`[sym_probe]`, `[tgt_hist]`, `[start_value]`, `[kqk_value]`, `[cm_count]`.
Watch the canonical-position probes: if `[start_value]`/`[kqk_value]` stay in
`[−0.1, +0.1]` for many steps, the value head has collapsed.

## Gotchas

1. **Policy = logits**: inference applies (masked) softmax; training uses raw logits + CE with `nan_to_num` to absorb `-inf` masked positions.
2. **Reward is per-step**: only the terminal step has a non-zero reward; ~99% of targets are 0, so the reward head is class-imbalanced.
3. **Reward loss divides by K, not K+1** (no reward at step 0).
4. **Gradient hook on g output**: `register_hook(lambda grad: grad * 0.5)` scales the chained K-step gradient.
5. **Loss weights at 1.0**: amplifying value weight destabilizes the closed-loop multi-head system. Prefer tuning β.
6. **`torch.load(weights_only=False)`** is intentional — checkpoints carry an `eval_metrics` dict alongside tensors.

## Related

- [Board Encoding](board-encoding.md) — observation planes, action space (4672)
- [MCTS](mcts.md) — how the evaluator drives search
- [Rust-Python Integration](rust-python-integration.md) — PyO3 bridge, batch contracts
- `python/hyzero/models/*.py` — network definitions
- `python/hyzero/training/trainer.py` — training loop
- `python/hyzero/inference/server.py` — batch inference
- `src/py/training.rs` — Rust-side batch assembly, target blending
