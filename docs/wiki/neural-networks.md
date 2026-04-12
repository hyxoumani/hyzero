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

## Key Gotchas

1. **Policy**: Network outputs logits. Inference server applies softmax; training uses raw logits + CE.
2. **Value**: Tanh [-1, 1]. Predicts advantage, not outcome directly.
3. **Reward**: Per-step (immediate), not cumulative. Real rewards come from trajectory.
4. **Action encoding**: 4096 = 64×64, queen-default promotion. Underpromotion (4672) unimplemented.
5. **Value not negated per ply** in backup — intentional, verify during training.
6. **Reward loss K not K+1**: Only K reward terms (steps 1..K), policy/value have K+1 (steps 0..K). Divide reward loss by K.
7. **Gradient hook on g output**: `register_hook(lambda grad: grad * 0.5)` on dynamics OUTPUT for correct chained K-step scaling (MuZero Appendix G).
8. **torch.load deprecation**: Use `weights_only=False` explicitly in PyTorch 2.x to avoid FutureWarning.

## Related Files

- `python/hyzero/models/*.py` — network definitions
- `python/hyzero/training/trainer.py` — training loop (Task 25)
- `python/hyzero/inference/server.py` — batch inference (Task 26)
- `src/data/encoding.rs` — board → observation encoding
- `docs/TASKS_PYTHON.md` — task specs (Tasks 24-26)
