# Architecture

## Overview

hyzero is a MuZero-based chess engine. The goal is to learn chess from scratch via self-play, using Monte Carlo Tree Search (MCTS) guided by learned neural networks. The system has two main components:

- **Rust engine**: Game state management, move generation/validation (bitboards + magic bitboards), MCTS tree search, and multi-game self-play orchestration
- **Python layer**: Neural networks (representation, dynamics, prediction), training loop, and replay buffer

## Component Architecture

```
+---------------------------------------------------+
|                   Rust Engine                      |
|  +-----------+  +----------+  +--------------+    |
|  | Game State|  |   MCTS   |  |  Inference   |    |
|  | (board,   |  | (tree,   |  |  Coordinator |    |
|  |  moves,   |  |  select, |  | (batches     |    |
|  |  validate)|  |  expand, |  |  leaf nodes, |    |
|  |           |  |  backup) |  |  calls PyO3) |    |
|  +-----------+  +----------+  +------+-------+    |
|                                      | PyO3/FFI   |
+--------------------------------------+------------+
                                       |
+--------------------------------------+------------+
|                 Python Layer          |            |
|  +-------------+ +----------+ +------+--------+   |
|  |  Policy     | |  Value   | |  Dynamics     |   |
|  |  Network    | |  Network | |  Network      |   |
|  | (h -> p)    | | (h -> v) | | (s,a -> s',r) |   |
|  +-------------+ +----------+ +---------------+   |
|  +--------------------------------------------+   |
|  |         Replay Buffer + Trainer            |   |
|  | (sample trajectories, unroll K steps,      |   |
|  |  compute loss, update weights)             |   |
|  +--------------------------------------------+   |
+---------------------------------------------------+
```

## Architectural Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| MCTS ownership | Rust | Avoids Python GIL contention; co-located with game engine for fast node expansion |
| Rust-Python IPC | PyO3/FFI | Simpler build than shared memory; direct numpy array passing; can migrate to shared memory later if GIL becomes a bottleneck |
| Self-play concurrency | Multi-game parallel | N game threads + 1 inference thread for GPU batch efficiency from the start |
| Data storage | In-memory replay buffer + disk checkpoints | Fast random-access sampling for training; durable across restarts |
| MCTS persistence | Visit counts + root value only | All training needs; tree is transient working memory, discarded after each move |

## Threading Model (Self-Play)

```
Game Thread 1 --+
Game Thread 2 --+--> inference queue --> Inference Thread --> PyTorch (GIL) --> GPU
Game Thread 3 --+                   <-- results via per-thread channels
Game Thread N --+
```

- **N game threads**: Pure Rust, no GIL. Each runs its own MCTS + game state. When leaf nodes need neural net evaluation, they submit to a shared inference queue and block for results.
- **1 inference thread**: Collects requests until batch is full (or timeout), acquires GIL once, calls PyTorch, distributes results back via per-thread channels.
- **Result**: One GIL acquisition per batch, not per node or per game. GPU stays saturated with large batches.

## MCTS Per-Move Flow

1. Build a fresh tree rooted at the current position
2. Run N simulations (typically 800):
   - **Select**: Walk tree using UCB/PUCT to find a leaf
   - **Expand**: Use dynamics model to predict next hidden state + reward
   - **Evaluate**: Use prediction model to get policy + value at the leaf
   - **Backpropagate**: Update visit counts and values up the tree
3. Extract visit count distribution (improved policy) + root value estimate
4. Select action based on visit counts (temperature-based exploration)
5. Store visit counts + root value as training targets
6. **Discard entire tree**
7. Apply action to real game state, repeat from step 1

The tree is transient working memory. All hallucinated branches (nodes expanded by the dynamics model during simulations) are thrown away after extracting the visit distribution. The visit distribution summarizes what the search found, and that is all the training loop needs.

## Training Data

### Per Game Step

| Field | Description |
|-------|-------------|
| `observation` | Board state (input to representation network) |
| `action` | Move actually played |
| `visit_distribution` | Normalized MCTS visit counts (policy target) |
| `root_value` | MCTS root value estimate (value target) |
| `reward` | 0 during game, +1/-1 at terminal state |
| `legal_moves` | Legal move mask for policy head |

### Per Game (Metadata)

| Field | Description |
|-------|-------------|
| `game_outcome` | Winner (+1/-1) or draw (0) |
| `model_version` | Which neural net version generated this game |
| `temperature_schedule` | Exploration temperature used for move selection |

## Training Loop

1. Sample a position at step `t` from the replay buffer
2. Unroll K steps forward (typically K=5). For each step t+k, compute loss on:
   - **Policy loss**: predicted policy vs MCTS visit distribution at t+k
   - **Value loss**: predicted value vs MCTS root value at t+k
   - **Reward loss**: predicted reward vs actual reward at t+k
3. Backpropagate combined loss through all three networks
4. Periodically checkpoint weights and sync to self-play threads

## Replay Buffer

- In-memory ring buffer of recent game trajectories
- Supports random access: sample position t, then read targets for t, t+1, ..., t+K
- Periodic checkpoint to disk (protobuf/msgpack) for crash recovery
- Tracks model version per game for staleness management
- Capacity sized to hold the most recent ~100K-1M game steps

## Neural Networks (Python)

### Representation Network (h)
- Input: raw board observation (8x8 board state + auxiliary features)
- Output: hidden state (learned latent representation)
- Only used at the root of MCTS (real position), not during hallucination

### Dynamics Network (g)
- Input: (hidden_state, action)
- Output: (next_hidden_state, predicted_reward)
- Used during MCTS expansion to "hallucinate" future states in latent space

### Prediction Network (f)
- Input: hidden_state
- Output: (policy, value)
- Used at every MCTS leaf to evaluate positions

## Game Server (Human/External Play)

Separate from self-play. A Unix domain socket server for two-player networked chess:

- Listens on `/tmp/hyzero.sock`
- Newline-delimited text protocol
- Accepts 2 clients (White first, Black second), coordinates turns
- Stores move history with `W:` / `B:` prefixes and board snapshots after each move

### Protocol

**Server -> Client**: `COLOR white|black`, `YOUR_TURN`, `OPPONENT_MOVED <notation>`, `OK <notation>`, `INVALID <reason>`, `GAME_OVER <result>`

**Client -> Server**: `MOVE <notation>` (e.g., `MOVE e2e4`)

## Future: Python Directory Structure

```
python/
  models/
    representation.py   # h: observation -> hidden state
    dynamics.py          # g: (hidden_state, action) -> (next_hidden_state, reward)
    prediction.py        # f: hidden_state -> (policy, value)
  training/
    trainer.py           # training loop, loss computation
    replay_buffer.py     # ring buffer + disk checkpointing
  inference/
    server.py            # inference entry point called via PyO3
```
