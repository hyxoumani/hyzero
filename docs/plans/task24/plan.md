# Plan: Task 24 — Python Project Setup + Model Definitions

## Approach

Create the `python/` directory from scratch with a minimal `pyproject.toml`, a
`hyzero` package containing config and all three neural networks built on a shared
`ResidualBlock`, and a pytest test file that verifies every output shape. No PyO3
bindings, no training loop, no inference server — those are Tasks 25 and 26.

## Subtasks

### 1. Package scaffold

- **Files**:
  - `python/pyproject.toml` (new)
  - `python/hyzero/__init__.py` (new)
  - `python/hyzero/config.py` (new)
  - `python/hyzero/models/__init__.py` (new)
  - `python/tests/__init__.py` (new, empty)

- **Changes**:

  `python/pyproject.toml`:
  ```toml
  [build-system]
  requires = ["setuptools>=68"]
  build-backend = "setuptools.backends.legacy:build"

  [project]
  name = "hyzero"
  version = "0.1.0"
  requires-python = ">=3.10"
  dependencies = [
      "torch>=2.0",
      "numpy>=1.24",
  ]

  [project.optional-dependencies]
  dev = ["pytest>=7.0"]

  [tool.setuptools.packages.find]
  where = ["."]
  ```

  `python/hyzero/__init__.py`: empty or single docstring.

  `python/hyzero/config.py`:
  ```python
  DEFAULT_CONFIG = {
      "hidden_channels": 64,
      "num_res_blocks": 4,
      "input_planes": 19,
      "num_actions": 4096,
      "action_planes": 3,
      "lr": 1e-3,
      "weight_decay": 1e-4,
  }
  ```

  `python/hyzero/models/__init__.py`: re-export all four model classes:
  ```python
  from .common import ResidualBlock
  from .representation import RepresentationNetwork
  from .dynamics import DynamicsNetwork
  from .prediction import PredictionNetwork
  ```

- **Tests**: None directly, but needed by all later subtasks.
- **Dependencies**: none

---

### 2. ResidualBlock

- **Files**: `python/hyzero/models/common.py` (new)

- **Changes**:
  ```python
  import torch.nn as nn

  class ResidualBlock(nn.Module):
      def __init__(self, channels: int):
          super().__init__()
          self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
          self.bn1 = nn.BatchNorm2d(channels)
          self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
          self.bn2 = nn.BatchNorm2d(channels)
          self.relu = nn.ReLU(inplace=True)

      def forward(self, x):
          residual = x
          out = self.relu(self.bn1(self.conv1(x)))
          out = self.bn2(self.conv2(out))
          out = self.relu(out + residual)
          return out
  ```
  ReLU is applied after the residual add (standard ResNet He et al. style).

- **Tests**: `test_residual_block_preserves_shape` — input `[2, 64, 8, 8]`, output `[2, 64, 8, 8]`.
- **Dependencies**: Subtask 1

---

### 3. RepresentationNetwork

- **Files**: `python/hyzero/models/representation.py` (new)

- **Changes**:
  ```python
  import torch.nn as nn
  from .common import ResidualBlock

  class RepresentationNetwork(nn.Module):
      def __init__(self, config: dict):
          super().__init__()
          C = config["hidden_channels"]
          in_planes = config["input_planes"]
          n_blocks = config["num_res_blocks"]
          self.stem = nn.Sequential(
              nn.Conv2d(in_planes, C, 3, padding=1, bias=False),
              nn.BatchNorm2d(C),
              nn.ReLU(inplace=True),
          )
          self.res_blocks = nn.Sequential(*[ResidualBlock(C) for _ in range(n_blocks)])

      def forward(self, x):
          # x: [B, 19, 8, 8]
          return self.res_blocks(self.stem(x))
          # output: [B, 64, 8, 8]
  ```

- **Tests**: `test_representation_network_shape` — input `[2, 19, 8, 8]`, output `[2, 64, 8, 8]`.
- **Dependencies**: Subtask 2

---

### 4. DynamicsNetwork

- **Files**: `python/hyzero/models/dynamics.py` (new)

- **Changes**:
  ```python
  import torch.nn as nn
  from .common import ResidualBlock

  class DynamicsNetwork(nn.Module):
      def __init__(self, config: dict):
          super().__init__()
          C = config["hidden_channels"]
          action_planes = config["action_planes"]
          n_blocks = config["num_res_blocks"]
          in_channels = C + action_planes  # 67

          self.stem = nn.Sequential(
              nn.Conv2d(in_channels, C, 3, padding=1, bias=False),
              nn.BatchNorm2d(C),
              nn.ReLU(inplace=True),
          )
          self.res_blocks = nn.Sequential(*[ResidualBlock(C) for _ in range(n_blocks)])
          self.reward_head = nn.Sequential(
              nn.Conv2d(C, 1, 1, bias=False),
              nn.Flatten(),          # [B, 64]
              nn.Linear(64, 1),
              nn.Tanh(),
          )

      def forward(self, hidden_state, action_planes):
          # hidden_state: [B, 64, 8, 8]
          # action_planes: [B, 3, 8, 8]
          import torch
          x = torch.cat([hidden_state, action_planes], dim=1)  # [B, 67, 8, 8]
          trunk = self.res_blocks(self.stem(x))                 # [B, 64, 8, 8]
          reward = self.reward_head(trunk)                      # [B, 1]
          return trunk, reward
  ```

- **Tests**:
  - `test_dynamics_network_shapes` — inputs `[2,64,8,8]` and `[2,3,8,8]`, outputs `([2,64,8,8], [2,1])`.
  - `test_dynamics_reward_bounded` — all reward values in `[-1, 1]`.
- **Dependencies**: Subtask 2

---

### 5. PredictionNetwork

- **Files**: `python/hyzero/models/prediction.py` (new)

- **Changes**:
  ```python
  import torch.nn as nn
  from .common import ResidualBlock

  class PredictionNetwork(nn.Module):
      def __init__(self, config: dict):
          super().__init__()
          C = config["hidden_channels"]
          num_actions = config["num_actions"]

          self.policy_head = nn.Sequential(
              nn.Conv2d(C, 2, 1, bias=False),
              nn.BatchNorm2d(2),
              nn.ReLU(inplace=True),
              nn.Flatten(),             # [B, 128]
              nn.Linear(128, num_actions),
          )
          self.value_head = nn.Sequential(
              nn.Conv2d(C, 1, 1, bias=False),
              nn.BatchNorm2d(1),
              nn.ReLU(inplace=True),
              nn.Flatten(),             # [B, 64]
              nn.Linear(64, 64),
              nn.ReLU(inplace=True),
              nn.Linear(64, 1),
              nn.Tanh(),
          )

      def forward(self, hidden_state):
          # hidden_state: [B, 64, 8, 8]
          policy_logits = self.policy_head(hidden_state)  # [B, 4096]
          value = self.value_head(hidden_state)           # [B, 1]
          return policy_logits, value
  ```

- **Tests**:
  - `test_prediction_network_shapes` — input `[2,64,8,8]`, outputs `([2,4096], [2,1])`.
  - `test_prediction_value_bounded` — all value outputs in `[-1, 1]`.
  - `test_prediction_policy_is_logits` — policy outputs are NOT bounded (no softmax applied here).
- **Dependencies**: Subtask 2

---

### 6. Test file

- **Files**: `python/tests/test_models.py` (new)

- **Changes**: One test function per assertion. Use `DEFAULT_CONFIG` from `hyzero.config`.
  All tests run on CPU. Use batch size `B=2` as the default (small and fast).

  Tests to write:
  ```
  test_residual_block_preserves_shape          — [2,64,8,8] → [2,64,8,8]
  test_representation_network_shape            — [2,19,8,8] → [2,64,8,8]
  test_dynamics_network_shapes                 — hidden [2,64,8,8] + action [2,3,8,8] → ([2,64,8,8], [2,1])
  test_dynamics_reward_bounded                 — all(reward.abs() <= 1)
  test_prediction_network_shapes               — [2,64,8,8] → ([2,4096], [2,1])
  test_prediction_value_bounded                — all(value.abs() <= 1)
  test_prediction_policy_raw_logits            — policy has no bound (just check shape and not NaN)
  test_batch_size_1                            — all networks work with B=1
  test_batch_size_32                           — all networks work with B=32
  ```

  All tests use `torch.randn(...)` for inputs and `torch.no_grad()` context.

- **Tests**: This subtask IS the tests.
- **Dependencies**: Subtasks 3, 4, 5

---

## Testing Strategy

After all files are written:

```bash
cd /path/to/hyzero/python
pip install -e ".[dev]"
pytest tests/test_models.py -v
```

Every test must pass. No GPU required — all tests run on CPU with random tensors.
The test suite proves the network architecture matches the spec shapes exactly.

## Rollback

All files are new. Remove `python/` entirely to revert. No existing Rust code is
touched by this task.

## Implementation Order

Dependencies run top-to-bottom; subtasks 3, 4, 5 can be implemented in any order
(all depend only on subtask 2):

```
1. Package scaffold
2. ResidualBlock (common.py)
3. RepresentationNetwork  \
4. DynamicsNetwork         > any order
5. PredictionNetwork      /
6. Test file (write after 3, 4, 5 complete)
```

## Ambiguities Resolved

1. **ReLU after residual add**: Applied after `out + residual`, not before. Standard He et al. convention.
2. **DynamicsNetwork state head**: The trunk output IS the next hidden state — no separate projection layer needed.
3. **Policy output**: Raw logits, no softmax. Softmax is applied at inference time (Task 26) so that training uses `F.cross_entropy` directly on logits.
4. **`pyproject.toml` vs `setup.py`**: Use `pyproject.toml` (modern standard, pip >= 21.3).
5. **`bias=False` on convolutions before BatchNorm**: Standard practice — BN has its own learnable affine params that subsume the bias. Apply `bias=False` to all Conv2d layers that precede a BatchNorm.
