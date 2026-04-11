# Research: Task 24 — Python Project Setup + Model Definitions

## Problem Statement

The hyzero MuZero engine needs a Python neural network layer. Task 24 creates the
package scaffold and all three core networks: RepresentationNetwork (h),
DynamicsNetwork (g), and PredictionNetwork (f), with a shared ResidualBlock. These
networks are consumed by Tasks 25 (training) and 26 (inference) before being
called from Rust via PyO3.

## Current State

No `python/` directory exists. The Rust side has full infrastructure for calling into
a neural network backend, but the `InferenceBackend` trait is currently fulfilled only
by `RandomBackend` (returns uniform policies and zero values). The Rust-Python boundary
is not yet implemented — that is Task 26's concern. Task 24 is pure Python.

The relevant Rust files are all read-only reference material for this task:
- `src/data/types.rs` — defines `BoardObservation`, `HiddenState`, data shapes
- `src/data/encoding.rs` — encodes board → 19 float planes, action → 3 spatial planes
- `src/mcts/evaluator.rs` — `Evaluator` trait: `root_setup` and `expand_leaf` signatures
- `src/selfplay/inference.rs` — `InferenceBackend`, `InferenceRequest` enum, `RandomBackend`

## Relevant Patterns

### Tensor shape contract (from Rust types and encoding.rs)

| Tensor | Shape | Notes |
|--------|-------|-------|
| Observation | `[B, 19, 8, 8]` | 19 planes, each 8x8; planes 0-5 white, 6-11 black, 12-15 castling, 16 EP, 17 side to move, 18 halfmove clock |
| Action spatial | `[B, 3, 8, 8]` | Plane 0: from-sq one-hot, plane 1: to-sq one-hot, plane 2: promotion flag constant |
| Hidden state | `[B, 64, 8, 8]` | 64 channels (configurable), spatial 8x8 |
| Dynamics input | `[B, 67, 8, 8]` | Hidden state (64) concatenated with action planes (3) |
| Policy logits | `[B, 4096]` | 64*64 action space, from_sq*64+to_sq |
| Value | `[B, 1]` | Bounded [-1, 1] via tanh |
| Reward | `[B, 1]` | Bounded [-1, 1] via tanh |

### Architecture (from TASKS_PYTHON.md)

**ResidualBlock**: Conv(C,C,3,pad=1) → BN → ReLU → Conv(C,C,3,pad=1) → BN, then add skip input (no activation on skip path, final ReLU outside or included — spec shows `+ skip` implying pre-activation is omitted, standard AlphaZero style where ReLU comes after the residual add).

**RepresentationNetwork (h)**: Conv(19,64,3,pad=1) → BN → ReLU → 4xResidualBlock → output `[B,64,8,8]`

**DynamicsNetwork (g)**:
- Stem: Conv(67,64,3,pad=1) → BN → ReLU → 4xResidualBlock → trunk `[B,64,8,8]`
- State head: identity (trunk IS the next hidden state)
- Reward head: Conv(64,1,1) → Flatten(64 elements) → Linear(64,1) → Tanh → `[B,1]`

**PredictionNetwork (f)**:
- Policy head: Conv(64,2,1) → BN → ReLU → Flatten(128 elements) → Linear(128,4096) → `[B,4096]` (raw logits)
- Value head: Conv(64,1,1) → BN → ReLU → Flatten(64 elements) → Linear(64,64) → ReLU → Linear(64,1) → Tanh → `[B,1]`

### ResidualBlock ReLU placement

The spec says `Conv → BN + skip`. Standard ResNet (He et al.) applies ReLU after the residual add. The spec does not show a final ReLU in the residual block, but this is the conventional placement. Use: BN → add skip → ReLU. This produces `F(x) + x` then ReLU, matching AlphaZero's implementation.

## Constraints

- Python is a library, not a standalone process. No `__main__` entry points in model files.
- Task 24 has zero PyO3 integration — that is Task 26. Do not add PyO3 bindings here.
- The `hidden_channels` config value is 64 and the `num_res_blocks` is 4. These must come from config, not be hardcoded, so Tasks 25/26 can override them.
- `pyproject.toml` is preferred over `setup.py` (modern packaging standard).
- No CUDA required for tests — tests must pass on CPU. Tests use random tensors, not real chess data.
- The `encode_action_spatial` function in Rust produces `[3*64]` flat — Python receives this reshaped to `[3, 8, 8]` per sample.

## Prior Art

No prior Python code in this repo. The task spec in `TASKS_PYTHON.md` is the definitive
reference for all network architectures and config defaults.

## Risks and Edge Cases

1. **Reward head flatten size**: DynamicsNetwork reward head does `Conv(64,1,1)` on `[B,64,8,8]` → `[B,1,8,8]` → Flatten → `[B,64]` → Linear(64,1). The flatten produces 64 elements, not 1. The spec says `Linear(64,1)` which confirms this.

2. **Policy head flatten size**: PredictionNetwork policy head does `Conv(64,2,1)` → `[B,2,8,8]` → Flatten → `[B,128]` → Linear(128,4096). Correct.

3. **Value head flatten size**: `Conv(64,1,1)` → `[B,1,8,8]` → Flatten → `[B,64]` → Linear(64,64) → ReLU → Linear(64,1) → Tanh. Correct.

4. **ResidualBlock skip connection**: Input and output channels are the same (C→C), so no projection is needed on the skip path. Confirm this in implementation.

5. **BatchNorm in eval mode**: For Task 26 inference, networks must be switched to `eval()` mode to get correct BatchNorm behavior (uses running stats). This is Task 26's responsibility; Task 24 must ensure `training=True` default.

6. **pyproject.toml dependencies**: Must list `torch` as a dependency. Torch version should not be pinned tightly to avoid conflicts — use `torch>=2.0`. Also list `numpy>=1.24` and `pytest` as dev dependency.

7. **Test device**: Tests should run on `cpu` explicitly to work in CI without GPU. The test file should instantiate networks with no device arg and call `.to("cpu")` or rely on default CPU placement.
