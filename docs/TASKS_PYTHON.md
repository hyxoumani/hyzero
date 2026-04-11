# Tasks: Python Neural Network Layer

## Overview

Build the Python neural network layer for MuZero: representation, dynamics, and prediction networks, plus the training loop and inference server entry points called from Rust via PyO3.

See `ARCHITECTURE.md` → "Python Neural Network Layer" for full design.

**Key constraint**: Python is a library called from Rust via PyO3. It does NOT run standalone. Rust owns the queues and calls Python methods with pre-packed numpy arrays.

---

## Phase 1: Foundation (Sequential)

### Task 24: Python Project Setup + Model Definitions

Create the Python package structure and implement all three neural networks with a shared ResidualBlock.

**New files:**
- `python/hyzero/__init__.py`
- `python/hyzero/config.py` — default hyperparameters
- `python/hyzero/models/__init__.py` — re-exports
- `python/hyzero/models/common.py` — `ResidualBlock`
- `python/hyzero/models/representation.py` — `RepresentationNetwork`
- `python/hyzero/models/dynamics.py` — `DynamicsNetwork`
- `python/hyzero/models/prediction.py` — `PredictionNetwork`
- `python/tests/test_models.py` — forward pass shape tests
- `python/setup.py` or `python/pyproject.toml`

**Config defaults:**
```python
DEFAULT_CONFIG = {
    "hidden_channels": 64,       # C
    "num_res_blocks": 4,         # residual blocks per network
    "input_planes": 19,          # observation planes
    "num_actions": 4096,         # 64 × 64 action space
    "action_planes": 3,          # spatial action encoding planes
    "lr": 1e-3,
    "weight_decay": 1e-4,
}
```

**ResidualBlock:**
- Conv2d(C, C, 3, padding=1) → BatchNorm2d → ReLU → Conv2d(C, C, 3, padding=1) → BatchNorm2d + skip

**RepresentationNetwork (h):**
- Input: `[B, 19, 8, 8]`
- Conv2d(19, 64, 3, padding=1) → BatchNorm2d → ReLU → 4 ResidualBlocks
- Output: `[B, 64, 8, 8]`

**DynamicsNetwork (g):**
- Input: `[B, 67, 8, 8]` (hidden state + 3 action planes concatenated)
- Conv2d(67, 64, 3, padding=1) → BatchNorm2d → ReLU → 4 ResidualBlocks
- State head: output `[B, 64, 8, 8]`
- Reward head: Conv2d(64, 1, 1) → Flatten → Linear(64, 1) → Tanh → output `[B, 1]`
- Returns: (next_hidden_state, reward)

**PredictionNetwork (f):**
- Input: `[B, 64, 8, 8]`
- Policy head: Conv2d(64, 2, 1) → BatchNorm2d → ReLU → Flatten → Linear(128, 4096) → output `[B, 4096]` (logits, not softmax)
- Value head: Conv2d(64, 1, 1) → BatchNorm2d → ReLU → Flatten → Linear(64, 64) → ReLU → Linear(64, 1) → Tanh → output `[B, 1]`
- Returns: (policy_logits, value)

**Tests (test_models.py):**
- Each network accepts correct input shape and produces correct output shape
- ResidualBlock preserves spatial dimensions
- DynamicsNetwork returns both hidden state and reward
- PredictionNetwork returns both policy logits and value
- Policy logits have 4096 outputs
- Value is bounded [-1, 1] (tanh)
- Reward is bounded [-1, 1] (tanh)

**Verify:** `cd python && pip install -e . && pytest tests/test_models.py -v`

---

## Phase 2: Training + Inference (Parallel after Phase 1)

### Task 25: Training Loop + Checkpointing `[PARALLEL]`

Implement the Trainer class with MuZero loss, K-step unrolling, checkpoint save/load, and weight serialization.

**New files:**
- `python/hyzero/training/__init__.py`
- `python/hyzero/training/trainer.py` — `Trainer` class
- `python/tests/test_training.py`

**Trainer class methods:**

`__init__(self, config: dict, device: str = "cuda")`:
- Instantiate all 3 networks in train mode
- Adam optimizer over all parameters
- Initialize `model_version = 0`

`train_batch(self, batch: dict) -> dict`:
- Input: dict with numpy arrays:
  - `"observations"`: `[B, 19, 8, 8]`
  - `"actions"`: `[B, K, 3, 8, 8]` — K actions for K-step unroll
  - `"target_policies"`: `[B, K+1, 4096]` — MCTS visit distributions
  - `"target_values"`: `[B, K+1]` — MCTS root values
  - `"target_rewards"`: `[B, K+1]` — actual rewards
- K-step unroll:
  - Step 0: `hidden = h(observations)`, `(policy, value) = f(hidden)`
  - Steps 1..K: `hidden, reward = g(hidden, actions[:, k-1])`, `(policy, value) = f(hidden)`
  - Scale dynamics gradient by 1/K for stability
- Loss: sum of policy cross-entropy + value MSE + reward MSE across all K+1 steps, divided by K+1
- Returns: `{"total_loss", "policy_loss", "value_loss", "reward_loss", "model_version"}`

`get_weights(self) -> bytes`:
- Serialize all 3 network state dicts via `torch.save()` to `io.BytesIO`
- Return bytes (Rust passes this opaquely to `InferenceServer.load_weights()`)

`save_checkpoint(self, path: str, eval_metrics: dict)`:
- Save: 3 network state dicts + optimizer state + model_version + eval_metrics
- `torch.save(...)` to path

`load_checkpoint(self, path: str) -> dict`:
- Load all state dicts + optimizer state + model_version
- Return eval_metrics dict

**Tests (test_training.py):**
- `train_batch()` with random data: returns loss dict with all keys, losses are finite
- Loss decreases over multiple training steps on fixed data
- `save_checkpoint()` + `load_checkpoint()` round-trip: model produces same output
- `get_weights()` returns non-empty bytes
- K-step unroll produces K+1 loss terms
- Gradient flows through all 3 networks (no detached tensors accidentally)

**Verify:** `pytest tests/test_training.py -v`

---

### Task 26: Inference Server `[PARALLEL]`

Implement the InferenceServer class with batch inference methods and weight loading.

**New files:**
- `python/hyzero/inference/__init__.py`
- `python/hyzero/inference/server.py` — `InferenceServer` class
- `python/tests/test_inference.py`

**InferenceServer class methods:**

`__init__(self, config: dict, device: str = "cuda")`:
- Instantiate all 3 networks in eval mode
- Store device

`root_setup_batch(self, observations: np.ndarray) -> tuple`:
- Input: `[B, 19, 8, 8]` float32 numpy
- Run `h(observations)` → hidden, then `f(hidden)` → (policy_logits, value)
- Apply softmax to policy_logits
- Return: `(hidden_states [B, 64, 8, 8], policies [B, 4096], values [B])` as numpy
- All computation under `torch.no_grad()`

`expand_leaf_batch(self, hidden_states: np.ndarray, actions: np.ndarray) -> tuple`:
- Input: `[B, 64, 8, 8]` hidden states + `[B, 3, 8, 8]` action planes, float32 numpy
- Concatenate → `[B, 67, 8, 8]`, run `g()` → (new_hidden, reward), then `f(new_hidden)` → (policy_logits, value)
- Apply softmax to policy_logits
- Return: `(new_hidden [B, 64, 8, 8], rewards [B], policies [B, 4096], values [B])` as numpy

`load_weights(self, state_dict_bytes: bytes)`:
- Deserialize bytes via `torch.load(io.BytesIO(state_dict_bytes))`
- Load into all 3 networks
- Keep networks in eval mode

**Tests (test_inference.py):**
- `root_setup_batch()`: correct output shapes for batch sizes 1, 8, 32
- `expand_leaf_batch()`: correct output shapes for batch sizes 1, 8, 32
- Policies sum to ~1.0 (softmax applied)
- Values bounded [-1, 1]
- Rewards bounded [-1, 1]
- `load_weights()` with bytes from `Trainer.get_weights()` — inference produces same results before/after sync
- All outputs are numpy arrays (not torch tensors)

**Verify:** `pytest tests/test_inference.py -v`

---

## Task Dependencies

```
Task 24 (project setup + models)
  │
  ├── Task 25 (training loop + checkpointing)
  │
  └── Task 26 (inference server)
```

**Execution:**
```
Task 24 → [Task 25 | Task 26] (parallel)
```

Python tasks can run **in parallel with Rust Tasks 17-20** since they share only the interface spec (tensor shapes), not code.

---

## Execution Strategy

Every task runs as a **subagent** with edit and bash permissions. After each task:
1. Run pytest to verify
2. Show full diff
3. Wait for user confirmation
4. Update CLAUDE.md and this doc with status

## Task Status

| Task | Status | Notes |
|------|--------|-------|
| 24. Python Project Setup + Models | DONE | Package scaffold, ResidualBlock, h/g/f networks, 9 tests passing |
| 25. Training Loop + Checkpointing | TODO | |
| 26. Inference Server | TODO | |
