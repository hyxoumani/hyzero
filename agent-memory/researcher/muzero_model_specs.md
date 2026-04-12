# MuZero Chess Engine Model Specifications

## Summary
Complete model specification for hyzero MuZero chess engine. Three-network architecture (RepresentationNetwork, DynamicsNetwork, PredictionNetwork) with 19-plane board encoding and 4096-action space (64×64 from-to move encoding).

## Configuration (config.py)

| Parameter | Value | Description |
|-----------|-------|-------------|
| hidden_channels (C) | 64 | Channels in all hidden layers |
| num_res_blocks | 4 | Residual blocks per network |
| input_planes | 19 | Observation planes (board encoding) |
| num_actions | 4096 | 64 × 64 action space |
| action_planes | 3 | Spatial action encoding planes |
| learning_rate | 1e-3 | Adam optimizer learning rate |
| weight_decay | 1e-4 | L2 regularization |

## Board Encoding (19 planes)

**Total size**: 19 × 8 × 8 = 1216 floats per observation

| Planes | Content | Format |
|--------|---------|--------|
| 0-5 | White pieces (Pawn, Knight, Bishop, Rook, Queen, King) | One-hot binary (1.0 if piece present) |
| 6-11 | Black pieces (same order) | One-hot binary |
| 12-15 | Castling rights (White Kingside, White Queenside, Black Kingside, Black Queenside) | Constant plane (all 64 squares = 1.0 if right available, else 0.0) |
| 16 | En passant target square | One-hot binary (1.0 if target set, else all 0.0) |
| 17 | Side to move | Constant plane (all 1.0 if White, all 0.0 if Black) |
| 18 | Halfmove clock | Constant plane (all squares = clock / 100.0) |

## Action Space

- **Total actions**: 4096
- **Encoding**: action = from_square × 64 + to_square (from_sq ∈ [0,63], to_sq ∈ [0,63])
- **Action spatial encoding** (3 planes for dynamics network):
  - Plane 0: source square one-hot
  - Plane 1: destination square one-hot
  - Plane 2: promotion flag (all 1.0 if move to rank 0 or 7, else 0.0)

Note: Default promotion is Queen; underpromotion added later.

## RepresentationNetwork (h)

Maps raw board observation → hidden state

```
Input:  [B, 19, 8, 8]
Output: [B, 64, 8, 8]

Architecture:
  Stem:
    - Conv2d(19 → 64, k=3, p=1, bias=False)
    - BatchNorm2d(64)
    - ReLU (inplace=True)
  
  Residual Stack (4 blocks):
    - 4 × ResidualBlock(64)
```

**Parameters**: ~42K
- Conv1: 19 × 64 × 3 × 3 = 10,944 params + BN(64)
- ResBlocks: 4 × (2 × Conv2d(64→64, k=3) + 2 × BN(64)) = ~31,360 params

## DynamicsNetwork (g)

Maps (hidden_state, action) → (next_hidden_state, reward)

```
Inputs:
  hidden_state:  [B, 64, 8, 8]
  action_planes: [B, 3, 8, 8]
Output:
  next_hidden:   [B, 64, 8, 8]
  reward:        [B, 1]  (bounded in [-1, 1] via Tanh)

Architecture:
  Concat hidden_state + action_planes → [B, 67, 8, 8]
  
  Stem:
    - Conv2d(67 → 64, k=3, p=1, bias=False)
    - BatchNorm2d(64)
    - ReLU (inplace=True)
  
  Residual Stack (4 blocks):
    - 4 × ResidualBlock(64)
  
  Reward Head:
    - Conv2d(64 → 1, k=1, bias=False)  [B, 1, 8, 8]
    - Flatten  [B, 64]
    - Linear(64 → 1)  [B, 1]
    - Tanh  → bounded in [-1, 1]
```

**Parameters**: ~34K
- Stem Conv: 67 × 64 × 3 × 3 = 38,592 params + BN(64)
- ResBlocks: ~31,360 params
- Reward head: 1×64×1 + 64×1 = ~128 params

## PredictionNetwork (f)

Maps hidden_state → (policy_logits, value)

```
Input:  [B, 64, 8, 8]
Outputs:
  policy_logits: [B, 4096]  (raw logits, no softmax)
  value:         [B, 1]  (bounded in [-1, 1] via Tanh)

Architecture:
  Policy Head:
    - Conv2d(64 → 2, k=1, bias=False)  [B, 2, 8, 8]
    - BatchNorm2d(2)
    - ReLU (inplace=True)
    - Flatten  [B, 128]
    - Linear(128 → 4096)  [B, 4096]
  
  Value Head:
    - Conv2d(64 → 1, k=1, bias=False)  [B, 1, 8, 8]
    - BatchNorm2d(1)
    - ReLU (inplace=True)
    - Flatten  [B, 64]
    - Linear(64 → 64)  [B, 64]
    - ReLU (inplace=True)
    - Linear(64 → 1)  [B, 1]
    - Tanh  → bounded in [-1, 1]
```

**Parameters**: ~30K
- Policy conv: 64×2×1 + BN(2) = ~130 params
- Policy linear: 128×4096 = ~524K params
- Value conv: 64×1×1 + BN(1) = ~65 params
- Value linear stack: 64×64 + 64×1 = ~4160 params

## ResidualBlock (common.py)

Preserves spatial/channel dimensions [B, C, 8, 8] → [B, C, 8, 8]

```
Input: [B, C, 8, 8]

Path 1 (residual):
  x (unchanged)

Path 2 (processing):
  Conv2d(C → C, k=3, p=1, bias=False)
  BatchNorm2d(C)
  ReLU (functional)
  Conv2d(C → C, k=3, p=1, bias=False)
  BatchNorm2d(C)

Output:
  ReLU(path2 + path1)
```

**Per-block parameters**: ~2×C² × 3×3 + 2×C = ~1152 per block (C=64)
- Conv1: 64 × 64 × 3 × 3 = 36,864 params
- Conv2: 64 × 64 × 3 × 3 = 36,864 params
- BN2: 64 params
- Total: ~73.8K per block × 4 blocks

## Training Hyperparameters (trainer.py)

### Loss Components
- **Policy loss**: Cross-entropy (negative log-likelihood) with soft target distribution
  - Formula: `-sum(targets * log_softmax(logits)) / B`
- **Value loss**: MSE between predicted and target values
  - Formula: `MSE(value.squeeze(-1), target_values)`
- **Reward loss**: MSE between predicted and target rewards (K steps only, not step 0)
  - Formula: `MSE(reward.squeeze(-1), target_rewards)`
- **Total loss**: `policy_loss + value_loss + reward_loss`

### K-Step Unroll
- Input batch:
  - observations: [B, 19, 8, 8]
  - actions: [B, K, 3, 8, 8]
  - target_policies: [B, K+1, 4096]
  - target_values: [B, K+1]
  - target_rewards: [B, K+1]
- **K-step unroll depth**: Not explicitly set in code (driven by batch preparation)
- **Gradient scaling**: 0.5× gradient at dynamics boundary (Appendix G) to stabilize K-step unroll

### Optimizer
- **Type**: Adam
- **Learning rate**: 1e-3
- **Weight decay**: 1e-4
- **Shared**: Single optimizer over all three networks (h, g, f)
- **Checkpointing**: Full state persisted (weights + optimizer state + model_version)

## MCTS Configuration

### MCTSConfig (src/mcts/tree.rs)
| Parameter | Value | Description |
|-----------|-------|-------------|
| num_simulations | 800 | Simulations per move |
| exploration_constant (c) | 1.5 | PUCT exploration weight |

### PUCT Score Formula
```
score(a) = Q(s,a) + c * P(s,a) * sqrt(N_parent) / (1 + N(a))
```
- Q = average value per visit (total_value / visit_count, or 0.0 if unvisited)
- P = prior from policy network (normalized over legal actions)
- N_parent = visit count of parent node
- N(a) = visit count of child action
- c = exploration constant (1.5)

### Dirichlet Noise
- **Applied at**: Root node only
- **Exploration fraction**: ε = 0.25
- **Alpha parameter**: α = 0.03 (chess)
- **Formula**: `P(a) = (1 - ε) * P(a) + ε * η_a` where η ~ Dir(α, ..., α)
- **Implementation**: Marsaglia-Tsang + Box-Muller for Gamma(α, 1) sampling

### Temperature
- **Early moves** (0 to temperature_moves-1): temperature = 1.0 (higher exploration)
- **Late moves** (temperature_moves onward): temperature = 0.01 (greedy)
- **Default temperature_moves**: 30

## Self-Play Configuration (GameConfig, coordinator.rs)

### Game Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| num_simulations | 800 | MCTS simulations per move |
| exploration_constant | 1.5 | PUCT constant |
| temperature_moves | 30 | Moves with temperature=1.0 before switching to 0.01 |
| max_concurrent_games | 4 (coordinator) | Parallel game concurrency |
| max_game_length | 300 half-moves | Prevent runaway games |

### Trajectory Recording
Each game produces a `GameTrajectory`:
- **steps**: Vec of StepRecord
  - observation: 19-plane board encoding
  - action: selected move (u16)
  - visit_distribution: normalized visits across legal actions
  - root_value: value prediction from root
  - reward: immediate reward (0 for most moves, -1/+1/0 at game end)
  - legal_moves: available actions from this state
- **game_outcome**: Final result (-1.0, 0.0, or +1.0)
- **model_version**: Which model weights generated this trajectory

## Network Parameter Count Summary

| Network | Layers | Approx Params |
|---------|--------|---------------|
| RepresentationNetwork | Stem + 4 ResBlocks | ~42K + 295K = ~337K |
| DynamicsNetwork | Stem + 4 ResBlocks + Reward head | ~38K + 295K + 128 = ~333K |
| PredictionNetwork | Policy head + Value head | ~524K + 4.2K = ~528K |
| **Total** | **13 layers + heads** | **~1.2M parameters** |

Note: This is approximate due to rounding and dependency on batch norm params.

## Data Types (src/data/types.rs)

- **ActionIndex**: u16 (0-4095)
- **Policy**: Vec<f32> (4096 action logits or probabilities)
- **BoardObservation**: Vec<f32> of 19×64=1216 floats
- **HiddenState**: Vec<f32> of channels×64 floats
- **StepRecord**: Single trajectory step (observation, action, visit dist, value, reward, legal moves)
- **GameTrajectory**: Full game (steps, outcome, model_version)
