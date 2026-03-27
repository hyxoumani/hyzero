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
| Async runtime | Tokio (not rayon) | Game tasks spend most time awaiting inference results — async fits better than work-stealing; tokio already in use for game server |
| Self-play mode | Continuous (Option A) | Games start immediately when slots open; no stop-the-world pauses; stale model data acceptable since replay buffer tracks version |
| Inference channel | Combined RootSetup/ExpandLeaf requests | Halves round-trips by combining h()+f() and g()+f() into single requests |
| Trajectory ownership | Move semantics, no Arc | Game task owns trajectory during play, moves it to replay buffer channel when done — no shared ownership needed |
| Inference/training separation | Separate channels | Different latency profiles: inference is latency-sensitive (800x per move), training is throughput-oriented (batch after games) |
| GIL strategy | Same-process, trait-abstracted | Simpler to build; trait boundary allows future process split if GIL contention becomes bottleneck |
| Action space | 4096 (64×64, queen default) | Simple encoding: `from*64+to`. Underpromotion added later by expanding to 4672 |
| Hidden state shape | `[C, 8, 8]` spatial, C=64 | Board has spatial structure; start small, increase C later |

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

### High-Level Steps

1. Build a fresh tree rooted at the current position
2. Run N simulations (typically 800)
3. Extract visit count distribution (improved policy) + root value estimate
4. Select action based on visit counts (temperature-based exploration)
5. Store visit counts + root value as training targets
6. **Discard entire tree**
7. Apply action to real game state, repeat from step 1

### Root Setup

At the start of each move, the representation network encodes the real board into latent space:

1. `h(real_board_observation)` → hidden state `s0`
2. `f(s0)` → (root_policy, root_value)
3. Root policy provides prior probabilities `P(s0, a)` for each legal move — used by PUCT to guide initial exploration

### Single Simulation Walkthrough

Each of the N simulations follows this path:

1. **Select**: Starting at the root, walk down the tree using PUCT scores to pick which child to visit at each node. At already-expanded nodes, PUCT balances the policy prior against the accumulated value evidence.

2. **Reach an unexpanded child**: PUCT selected action `a` from a node with hidden state `s_parent`, but that child doesn't exist yet.

3. **Expand via dynamics model**: `g(s_parent, a)` → `(s_new, reward)`. This produces the child's hidden state and predicted immediate reward. The dynamics model operates entirely in latent space — it never touches the real board.

4. **Evaluate via prediction model**: `f(s_new)` → `(policy, value)`. The policy tells MCTS how to prioritize future exploration from this node. The value estimates how good this position is.

5. **Store in tree**: Create the new node with `s_new`, `reward`, `policy` (as child priors), and `value`.

6. **Backpropagate**: Walk back up to the root, updating visit counts and adding `value` to each ancestor's total value.

### Deeper Traversal

When PUCT walks through already-expanded nodes before reaching a leaf, the dynamics model uses the **parent node's hidden state** — not the root's. This builds chains through latent space:

```
Root: s0 (from representation network h)
  └─ action a1 → s1 = g(s0, a1)     [simulation 1 expanded this]
       └─ action a3 → s2 = g(s1, a3) [simulation 5 expanded this]
            └─ action a7 → s3 = g(s2, a7) [simulation 12 expanded this]
```

Each node stores its own hidden state, so traversal doesn't require recomputing the chain from the root.

### Tree Discard

After all N simulations complete, extract the visit count distribution at the root and the root Q value. Then **discard the entire tree**, including all hidden state tensors stored in nodes. The tree is transient working memory — the visit distribution is its compressed summary, and that is all the training loop needs.

## PUCT Selection Formula

At each internal node during selection, MCTS picks the child action with the highest PUCT score:

```
score(a) = Q(s,a) + c * P(s,a) * sqrt(N_parent) / (1 + N(a))
```

| Term | Meaning |
|------|---------|
| `Q(s,a)` | Average backpropagated value for action `a` — exploitation signal from the value network |
| `P(s,a)` | Prior probability from the prediction network's policy output — exploration signal |
| `N(a)` | Visit count for action `a` |
| `N_parent` | Total visits to the parent node |
| `c` | Exploration constant (hyperparameter) |

**Behavior over time:**
- **Early in search** (small `N(a)`): The exploration term `P(s,a) * sqrt(N_parent) / (1 + N(a))` dominates → MCTS explores what the policy network suggests
- **Late in search** (large `N(a)`): The exploration term shrinks, `Q(s,a)` dominates → MCTS exploits positions the value network found to be strong
- **Correction mechanism**: A move with high prior `P` but poor `Q` (value estimates come back low after exploring its subtree) will stop being visited. This is how the value network corrects the policy during search.

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

### Training Data Scope

Only the **actually played trajectory** produces training data — one sample per real move. A 40-move game produces exactly 40 training samples, not thousands from hallucinated branches. All simulated tree branches served their purpose by informing the visit distribution and root value, which are the compressed summary of the entire search tree.

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

## Value Training & Self-Improvement Loop

### Three Training Losses

| Loss | What it trains | Target | Source |
|------|---------------|--------|--------|
| **Policy loss** | Policy head of `f` | MCTS visit distribution | Normalized visit counts at root after N simulations |
| **Value loss** | Value head of `f` | MCTS root Q value | Average backpropagated value at root after N simulations |
| **Reward loss** | Reward head of `g` | Actual reward | 0 during game, +1/-1 at terminal state |

The key insight: the value network is trained to predict the **MCTS root value**, which is a much more informed estimate than what the network alone would produce (because it incorporates the results of search). This is **distilling search into the network**.

```
Without search: value_network(position) → 0.2  (rough guess)
With 800 sims:  MCTS root Q value       → 0.35 (informed estimate)
Training target: make value_network(position) output 0.35
```

### Self-Improvement Cycle

1. **Weak networks** → noisy MCTS, but search is still better than raw network output → slightly informative visit distributions and root values
2. **Train on MCTS outputs** → networks improve slightly
3. **Better networks** → better leaf evaluations during MCTS → better Q values → sharper, more accurate visit distributions
4. **Train on better MCTS outputs** → networks improve further
5. **Repeat** → the value network eventually "internalizes" what search would find without needing to search, and the policy network learns to directly output the improved policy that MCTS would produce

Each component lifts the other: MCTS improves the training signal for the networks, and better networks improve the quality of MCTS.

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
- Used **once per real move** at the MCTS root to encode the actual board into latent space
- Never called during hallucination — only the dynamics network operates inside the search tree

### Dynamics Network (g)
- Input: (parent_hidden_state, action)
- Output: (child_hidden_state, predicted_reward)
- Used at **every MCTS expansion** to "hallucinate" the next state in latent space
- Operates entirely in latent space — never touches the real board representation
- Chains build depth: `s0 → s1 → s2 → ...` where each step is one dynamics call

### Prediction Network (f)
- Input: hidden_state (from either `h` or `g`)
- Output: (policy, value)
- Used at **every MCTS leaf** (including the root after `h`)
- **Policy output**: prior probabilities over legal moves, used by PUCT to guide exploration from this node
- **Value output**: position evaluation, backpropagated up the tree to update Q values

### Network Data Flow Summary

```
Real board → h() → s0 → f() → (root_policy, root_value)
                                     |
                              PUCT selects action a
                                     |
                    g(s0, a) → (s1, r1) → f(s1) → (policy1, value1)
                                                        |
                                                 PUCT selects action b
                                                        |
                                       g(s1, b) → (s2, r2) → f(s2) → (policy2, value2)
                                                                            |
                                                                     backpropagate value2 up
```

## MCTS Node Structure (Rust)

Preliminary design for the tree node:

```rust
struct MCTSNode {
    hidden_state: Tensor,             // from h() or g()
    visit_count: u32,                 // N(s)
    total_value: f32,                 // W(s) — sum of backpropagated values
    reward: f32,                      // r from dynamics model (0 at root)
    prior: f32,                       // P(s,a) from parent's policy output
    children: Vec<Option<MCTSNode>>,  // indexed by action, None = unexpanded
}
```

- `children` is sized to the number of legal moves at this position
- Each child starts as `None` and gets populated when PUCT first selects that action
- `Q(s,a)` is computed as `child.total_value / child.visit_count`
- The `hidden_state` tensor is the dominant memory cost per node

### Memory Estimate

With ~30 legal moves in a typical chess position and 800 simulations, the tree has a few hundred expanded nodes. If the hidden state is 256 floats:

```
~300 nodes × 256 floats × 4 bytes = ~300 KB per tree
× N parallel games = total memory
```

This is very manageable even with dozens of parallel games.

## Rust Self-Play Infrastructure

### Overview

The self-play system runs continuously: N game tasks execute in parallel on a tokio runtime, each playing a full game of chess using MCTS to select moves. Games produce training trajectories that flow to a replay buffer. Training runs concurrently — when a new model checkpoint is produced, new games pick it up automatically. Games in progress finish with their current model version.

```
+-----------------------------------------------------------------------+
|                        Tokio Runtime                                  |
|                                                                       |
|  Game Task 1 --+                                                      |
|  Game Task 2 --+--> [inference_tx] --> Inference Thread --> PyO3/GPU   |
|  Game Task 3 --+                   <-- oneshot results                |
|  ...           |                                                      |
|  Game Task N --+                                                      |
|       |                                                               |
|       +--> [trajectory_tx] --> Replay Buffer Thread                   |
|                                    |                                  |
|                                    +--> [training_tx] --> Training    |
|                                    |        Thread --> PyO3/GPU       |
|                                    |                                  |
|                                    +--> Periodic disk checkpoint      |
|                                                                       |
|  Coordinator: spawns new game task when one finishes                  |
+-----------------------------------------------------------------------+
```

### Inference Channel

Game tasks submit neural net evaluation requests through a shared `mpsc` channel. Each request includes a `oneshot::Sender` for the response, so the game task can `.await` the result without blocking other tasks.

```rust
enum InferenceRequest {
    /// At MCTS root: encode real board into latent space, then predict policy + value
    /// Combines h() + f() into one round-trip
    RootSetup {
        observation: BoardObservation,
        reply: oneshot::Sender<(HiddenState, Policy, f32)>,  // (s0, policy, value)
    },
    /// During MCTS expansion: predict next state + reward, then predict policy + value
    /// Combines g() + f() into one round-trip
    ExpandLeaf {
        hidden_state: HiddenState,
        action: ActionIndex,
        reply: oneshot::Sender<(HiddenState, f32, Policy, f32)>,  // (s_new, reward, policy, value)
    },
}
```

**Batching**: The inference thread collects requests until either (a) the batch is full (e.g., 32-128 requests) or (b) a timeout fires (e.g., 1ms). It then acquires the GIL once, runs all requests through PyTorch as a batch, and sends results back through each request's `oneshot` channel.

### Training Data Types

```rust
/// One step of a played game — produced after each real move
struct StepRecord {
    observation: BoardObservation,    // board state (input to h())
    action: ActionIndex,              // move that was actually played
    visit_distribution: Vec<f32>,     // normalized MCTS visit counts (policy target)
    root_value: f32,                  // MCTS root Q value (value target)
    reward: f32,                      // 0.0 during game, +1.0/-1.0 at terminal
    legal_moves: Vec<ActionIndex>,    // legal move mask for policy head
}

/// Complete trajectory of a played game — moved to replay buffer when game ends
struct GameTrajectory {
    steps: Vec<StepRecord>,
    game_outcome: f32,                // +1.0 (white wins), -1.0 (black wins), 0.0 (draw)
    model_version: u64,               // which model checkpoint generated this game
}
```

**Ownership flow**: The game task owns its `GameTrajectory` during play. When the game ends, the trajectory is **moved** (not cloned, not shared) into the trajectory channel. No `Arc` needed — the game task is done with it.

```
Game task builds GameTrajectory
  → moves into mpsc::channel<GameTrajectory>
  → Replay buffer thread receives and owns it
```

### Replay Buffer

The replay buffer lives on a dedicated thread (or tokio task) that:

1. Receives completed `GameTrajectory` objects from the trajectory channel
2. Stores them in an in-memory ring buffer (bounded capacity, oldest evicted first)
3. Supports random-access sampling: pick a random game, pick a random step `t`, return steps `t..t+K` for K-step unrolling
4. Periodically serializes to disk (bincode/msgpack) for crash recovery
5. Tracks model version per game for optional staleness weighting

```rust
struct ReplayBuffer {
    trajectories: VecDeque<GameTrajectory>,   // ring buffer, oldest at front
    max_trajectories: usize,                  // capacity limit
    total_steps: usize,                       // sum of all trajectory lengths
}

impl ReplayBuffer {
    fn add(&mut self, trajectory: GameTrajectory) { /* push back, evict front if full */ }
    fn sample_batch(&self, batch_size: usize, unroll_k: usize) -> Vec<TrainingSample> { /* random sampling */ }
    fn checkpoint_to_disk(&self, path: &Path) { /* serialize */ }
    fn load_from_disk(path: &Path) -> Self { /* deserialize */ }
}
```

### Training Channel

Separate from inference. The training thread:

1. Periodically samples a batch from the replay buffer (via a request/response channel or shared access)
2. Sends the batch to Python via PyO3 for forward pass, loss computation, and backpropagation
3. Receives updated model weights
4. Publishes the new model version (e.g., via `tokio::sync::watch`) so that game tasks starting new games can pick it up

```rust
/// Training thread sends this to Python
struct TrainingBatch {
    samples: Vec<TrainingSample>,
}

/// Per-sample data for K-step unrolling
struct TrainingSample {
    steps: Vec<StepRecord>,   // K+1 consecutive steps starting from sampled position
    game_outcome: f32,
}
```

### Self-Play Coordinator

The coordinator is the top-level orchestrator. It runs on the tokio runtime and manages the continuous self-play loop:

```rust
struct SelfPlayCoordinator {
    precomputed: Arc<PrecomputedItems>,
    inference_tx: mpsc::Sender<InferenceRequest>,
    trajectory_tx: mpsc::Sender<GameTrajectory>,
    model_version: watch::Receiver<u64>,
    max_concurrent_games: usize,
}
```

**Continuous game spawning (Option A)**:

```rust
impl SelfPlayCoordinator {
    async fn run(&self) {
        let semaphore = Arc::new(Semaphore::new(self.max_concurrent_games));
        loop {
            let permit = semaphore.clone().acquire_owned().await.unwrap();
            let game = self.spawn_game();
            tokio::spawn(async move {
                game.play().await;  // runs full game, sends trajectory when done
                drop(permit);       // releases slot, coordinator spawns next game
            });
        }
    }
}
```

Each game task:
1. Creates a fresh `GameBoard` with standard starting position
2. Loops: run MCTS for current position → select move → apply move → record `StepRecord`
3. When game ends (checkmate, stalemate, draw): build `GameTrajectory`, move it into the trajectory channel
4. Task exits, semaphore permit drops, coordinator spawns a new game

### Model Version Propagation

New games pick up the latest model version automatically. Games in progress continue with their current version — this is fine because:

- The replay buffer tracks `model_version` per trajectory
- Training can weight recent (newer model) games higher if desired
- A game using a slightly stale model still produces useful training data

The model version is propagated via `tokio::sync::watch`:

```
Training thread produces new weights → updates watch::Sender<u64>
Game tasks check watch::Receiver<u64> at game start → use latest version
```

### Board Observation Encoding

The `BoardObservation` encodes the board as 19 float planes (8×8 each) for the representation network:

| Planes | Content |
|--------|---------|
| 0-5 | White pieces (pawn, knight, bishop, rook, queen, king) — 1.0 where piece exists |
| 6-11 | Black pieces (same order) |
| 12-15 | Castling rights (WK, WQ, BK, BQ) — all 1.0 if available, all 0.0 if not |
| 16 | En passant target — 1.0 at EP square, 0.0 elsewhere |
| 17 | Side to move — all 1.0 if white, all 0.0 if black |
| 18 | Halfmove clock — all squares = clock/100.0 (normalized) |

```rust
struct BoardObservation {
    pub planes: [f32; 19 * 64],  // 1216 floats, maps to numpy [19, 8, 8]
}
```

Constructed from `GameBoard`'s existing fields: `player1.pieces_bb[6]` / `player2.pieces_bb[6]` → piece planes, castling booleans, `en_passant_target`, `halfmove_clock` → auxiliary planes.

### Hidden State

```rust
struct HiddenState {
    pub data: Vec<f32>,    // length = channels * 64, maps to numpy [C, 8, 8]
    pub channels: usize,   // C (default 64)
}
```

### Action Space Encoding

4096 actions: `from_square * 64 + to_square`. Default queen promotion (underpromotion added later).

```rust
type ActionIndex = u16;   // index into 4096 action space
type Policy = Vec<f32>;   // probability per action

fn move_to_action(mv: &Move) -> ActionIndex { mv.from as u16 * 64 + mv.to as u16 }
fn action_to_move(action: ActionIndex) -> Move { /* from = action/64, to = action%64 */ }
```

For the dynamics network, actions are also encoded as 3 spatial planes (source one-hot, dest one-hot, promotion flag):

```rust
fn encode_action_spatial(action: ActionIndex) -> [f32; 3 * 64] { /* ... */ }
```

### Training Config

```rust
struct TrainingConfig {
    min_samples_before_training: usize,  // e.g., 10_000 steps — don't train until buffer has this many
    train_batch_size: usize,             // e.g., 256
    unroll_k: usize,                     // e.g., 5
    weight_sync_interval: usize,         // sync to inference every M training batches
    checkpoint_interval: usize,          // save to disk every N training batches
    max_loss_threshold: f32,             // rollback if loss exceeds this
}

### Rust Infra Changes From Python Spec

Speccing the Python side surfaced several changes to the Rust infrastructure design:

**1. BoardObservation: flat struct → 19-plane array**
- Original: separate `piece_planes: [[f32; 64]; 12]` + `AuxiliaryFeatures` struct
- Changed to: `planes: [f32; 1216]` (19 × 64 contiguous floats)
- **Why**: Python expects a single numpy array `[19, 8, 8]`. A flat contiguous array maps directly to numpy without copying/reshaping. Splitting into struct fields would require the Rust batcher to manually interleave them into a contiguous buffer before every PyO3 call.

**2. HiddenState: `Vec<f32>` → struct with channels**
- Original: `type HiddenState = Vec<f32>` (opaque flat vector)
- Changed to: `struct HiddenState { data: Vec<f32>, channels: usize }`
- **Why**: The Python hidden state is `[C, 8, 8]` spatial. Rust needs to know `C` to pack batches correctly (`[B, C, 8, 8]`) — the batch dimension must be added before the channel dimension. Without `channels`, the batcher can't compute the stride.

**3. Action encoding: dual representation**
- Original: `ActionIndex` as an integer index for policy vector
- Added: `encode_action_spatial()` → `[f32; 192]` (3 planes × 64 squares)
- **Why**: Python uses `ActionIndex` for the policy head (index into 4096 logits), but the dynamics network needs the action as spatial planes concatenated with the hidden state `[C+3, 8, 8]`. Two different representations of the same action cross the PyO3 boundary at different points.

**4. Inference batcher: separate batching by request type**
- Original: single batch of mixed `InferenceRequest` variants
- Changed to: **RootSetup and ExpandLeaf batched separately**
- **Why**: They call different Python methods (`root_setup_batch` vs `expand_leaf_batch`) with different input shapes. RootSetup takes `[B, 19, 8, 8]` observations; ExpandLeaf takes `[B, C, 8, 8]` hidden states + `[B, 3, 8, 8]` action planes. Mixing them in one batch would require the batcher to split them anyway.

**5. Training thread: expanded with threshold + failover logic**
- Original: simple "sample batch, send to Python, get weights"
- Changed to: training thread owns replay buffer, checks `min_samples_before_training` threshold, monitors loss for divergence, rolls back to `best.pt` checkpoint if needed
- **Why**: Without a threshold, training would start with near-zero data and produce garbage updates. Without failover, a single bad training batch (NaN gradients, loss spike) would corrupt the model permanently.

**6. Training batch: `Vec<TrainingSample>` → `PackedTrainingBatch`**
- Original: send `Vec<TrainingSample>` to Python
- Changed to: `PackedTrainingBatch` with pre-packed contiguous float arrays
- **Why**: Python `train_batch()` expects numpy arrays shaped `[B, 19, 8, 8]`, `[B, K, 3, 8, 8]`, `[B, K+1, 4096]`, etc. Sending a Vec of Rust structs would require Python-side unpacking. Pre-packing on the Rust side means a single zero-copy buffer can be passed through PyO3 as numpy.

**7. Weight sync: opaque bytes**
- Not in original spec
- **Why**: `Trainer.get_weights()` returns `torch.save()` serialized bytes. Rust treats this as `Vec<u8>` — completely opaque. Passes it to `InferenceServer.load_weights(bytes)`. Rust never inspects model weights, only shuttles them.

### Planned Rust Module Structure

```
src/
  mcts/
    mod.rs          // MCTSTree, MCTSNode, run_simulations()
    puct.rs         // PUCT selection logic
  selfplay/
    mod.rs          // SelfPlayCoordinator
    game_task.rs    // single game loop (MCTS per move, build trajectory)
    inference.rs    // InferenceRequest enum, batching thread
    training.rs     // TrainingBatch, training thread, model version watch
  data/
    mod.rs          // StepRecord, GameTrajectory, BoardObservation, ActionIndex
    replay_buffer.rs // ReplayBuffer, sampling, disk checkpoint
    encoding.rs     // GameBoard → BoardObservation conversion, Move ↔ ActionIndex mapping
```

## Game Server (Human/External Play)

Separate from self-play. A Unix domain socket server for two-player networked chess:

- Listens on `/tmp/hyzero.sock`
- Newline-delimited text protocol
- Accepts 2 clients (White first, Black second), coordinates turns
- Stores move history with `W:` / `B:` prefixes and board snapshots after each move

### Protocol

**Server -> Client**: `COLOR white|black`, `YOUR_TURN`, `OPPONENT_MOVED <notation>`, `OK <notation>`, `INVALID <reason>`, `GAME_OVER <result>`

**Client -> Server**: `MOVE <notation>` (e.g., `MOVE e2e4`)

## Python Neural Network Layer

Python is a library called from Rust via PyO3 — it does NOT run as a standalone process. Rust owns both the inference and training queues and calls into Python when ready.

### Network Architectures

All networks use a shared `ResidualBlock` (conv-bn-relu-conv-bn + skip connection). Start with C=64 channels and 4 residual blocks per network. Tune later.

| Network | Input | Output | Architecture |
|---------|-------|--------|-------------|
| Representation (h) | `[B, 19, 8, 8]` | `[B, 64, 8, 8]` | Conv2d(19→64, 3×3) + 4 residual blocks |
| Dynamics (g) | `[B, 67, 8, 8]` | `([B, 64, 8, 8], [B, 1])` | Conv2d(67→64, 3×3) + 4 residual blocks + reward head (conv→flatten→fc→tanh) |
| Prediction (f) | `[B, 64, 8, 8]` | `([B, 4096], [B, 1])` | Policy head (conv→flatten→fc→4096 logits) + value head (conv→flatten→fc→tanh) |

### Input Observation (19 planes)

| Planes | Content |
|--------|---------|
| 0-5 | White pieces (pawn, knight, bishop, rook, queen, king) |
| 6-11 | Black pieces (same order) |
| 12-15 | Castling rights (WK, WQ, BK, BQ) — constant plane per right |
| 16 | En passant target square |
| 17 | Side to move (all 1s = white) |
| 18 | Halfmove clock (normalized by 100) |

### Action Encoding for Dynamics

The dynamics network needs the action as spatial planes concatenated with the hidden state:
- Plane 0: source square one-hot (8×8)
- Plane 1: destination square one-hot (8×8)
- Plane 2: promotion flag (all 1s if promotion, all 0s otherwise)

Input to dynamics: `cat(hidden_state, action_planes)` → `[B, C+3, 8, 8]` = `[B, 67, 8, 8]`

### PyO3 Interface

Two Rust threads call Python. They share the GIL but never run simultaneously. The training thread yields GIL between batches so inference can proceed.

#### InferenceServer (called from Rust inference thread)

```python
class InferenceServer:
    def __init__(self, config: dict, device: str = "cuda"):
        # Loads all 3 networks in eval mode

    def root_setup_batch(self, observations: np.ndarray) -> tuple:
        # observations: [B, 19, 8, 8] float32
        # returns: (hidden_states [B, 64, 8, 8], policies [B, 4096], values [B])
        # Runs h() then f() — representation + prediction

    def expand_leaf_batch(self, hidden_states: np.ndarray, actions: np.ndarray) -> tuple:
        # hidden_states: [B, 64, 8, 8], actions: [B, 3, 8, 8] float32
        # returns: (new_hidden [B, 64, 8, 8], rewards [B], policies [B, 4096], values [B])
        # Runs g() then f() — dynamics + prediction

    def load_weights(self, state_dict_bytes: bytes):
        # Deserializes and loads new weights from training
```

#### Trainer (called from Rust training thread)

```python
class Trainer:
    def __init__(self, config: dict, device: str = "cuda"):
        # Loads all 3 networks in train mode + Adam optimizer

    def train_batch(self, batch: dict) -> dict:
        # batch keys (numpy arrays):
        #   "observations": [B, 19, 8, 8]
        #   "actions": [B, K, 3, 8, 8]        — K actions for unrolling
        #   "target_policies": [B, K+1, 4096]  — MCTS visit distributions
        #   "target_values": [B, K+1]           — MCTS root values
        #   "target_rewards": [B, K+1]          — actual rewards
        # returns: {"total_loss", "policy_loss", "value_loss", "reward_loss", "model_version"}

    def get_weights(self) -> bytes:
        # Serializes current weights for inference server to load

    def save_checkpoint(self, path: str, eval_metrics: dict):
        # Saves weights + optimizer state + eval metrics to disk

    def load_checkpoint(self, path: str) -> dict:
        # Loads from disk, returns eval_metrics
```

### Training Trigger & Data Flow

Training does not run immediately. The Rust training thread manages the lifecycle:

1. Receive `GameTrajectory` objects from trajectory channel → add to replay buffer
2. Check: `replay_buffer.total_steps() >= min_samples_before_training`?
   - No → continue receiving, do not call Python
   - Yes → proceed to step 3
3. Sample a `PackedTrainingBatch` from replay buffer
4. Acquire GIL, call `trainer.train_batch(batch)`
5. Check returned loss — if NaN or exceeds `max_loss_threshold`, rollback to `best.pt`
6. Every M batches: call `trainer.get_weights()` → send to inference server via `load_weights()`
7. Every N batches: call `trainer.save_checkpoint()` with eval metrics
8. Release GIL, loop back to step 1

### Weight Storage & Failover

Checkpoints stored in `checkpoints/` directory:
- `checkpoints/latest.pt` — most recent (always overwritten)
- `checkpoints/v{version}.pt` — periodic snapshots
- `checkpoints/best.pt` — best model by evaluation metric

Each checkpoint contains: all 3 network state dicts, optimizer state, model version, eval metrics dict.

**Failover**: If training diverges (NaN loss or spike above threshold), Rust detects it from returned loss dict, calls `trainer.load_checkpoint("checkpoints/best.pt")`, and resumes.

### MuZero Loss (K-step unroll)

```
For each sample in batch:
  Step 0: hidden = h(observation), (policy, value) = f(hidden)
          loss += cross_entropy(policy, target_policy[0]) + mse(value, target_value[0])

  Steps 1..K: hidden, reward = g(hidden, action[k-1]), (policy, value) = f(hidden)
              loss += cross_entropy(policy, target_policy[k]) + mse(value, target_value[k]) + mse(reward, target_reward[k])
              scale dynamics gradient by 1/K for stability
```

### Python Directory Structure

```
python/
  hyzero/
    __init__.py
    config.py                # C=64, num_blocks=4, lr=1e-3, weight_decay=1e-4, etc.
    models/
      __init__.py
      common.py              # ResidualBlock
      representation.py      # h: [B, 19, 8, 8] → [B, 64, 8, 8]
      dynamics.py            # g: [B, 67, 8, 8] → ([B, 64, 8, 8], [B, 1])
      prediction.py          # f: [B, 64, 8, 8] → ([B, 4096], [B, 1])
    training/
      __init__.py
      trainer.py             # Trainer class: train_batch, K-step unroll, checkpoints
    inference/
      __init__.py
      server.py              # InferenceServer class: batch inference methods
  tests/
    test_models.py           # forward pass shape verification
    test_training.py         # loss computation correctness
    test_inference.py        # batch inference shape verification
  checkpoints/               # model weight storage
  setup.py
```

**Note**: No Python-side replay buffer. The replay buffer lives in Rust — Python only receives pre-packed numpy batches via PyO3.
