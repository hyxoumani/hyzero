# Tasks: MCTS & Self-Play Infrastructure

## Overview

Build the Rust infrastructure for MuZero self-play: MCTS tree search, inference batching, training data collection, replay buffer, and continuous self-play orchestration. All neural net calls are stubbed with trait-based interfaces until PyO3 integration.

See `ARCHITECTURE.md` → "Rust Self-Play Infrastructure" for full design.

---

## Phase 1: Foundation (Sequential)

### Task 17: Create Module Structure + Core Data Types

Create the new module directories and define all shared types that other components depend on.

**New files:**
- `src/data/mod.rs` — re-exports
- `src/data/types.rs` — `StepRecord`, `GameTrajectory`, `BoardObservation`, `AuxiliaryFeatures`, `HiddenState`, `Policy`, `ActionIndex`
- `src/data/encoding.rs` — `GameBoard → BoardObservation` conversion, `Move ↔ ActionIndex` mapping
- `src/mcts/mod.rs` — re-exports
- `src/selfplay/mod.rs` — re-exports

**Modify:**
- `src/lib.rs` — add `pub mod data;`, `pub mod mcts;`, `pub mod selfplay;`

**Types to define:**
```rust
pub type ActionIndex = u16;
pub type HiddenState = Vec<f32>;  // placeholder until PyO3 tensor type decided
pub type Policy = Vec<f32>;

pub struct BoardObservation {
    pub piece_planes: [[f32; 64]; 12],  // 6 piece types × 2 colors
    pub castling_rights: [f32; 4],       // WK, WQ, BK, BQ
    pub en_passant: [f32; 64],           // one-hot EP target square
    pub side_to_move: f32,               // 1.0 = white, 0.0 = black
    pub halfmove_clock: f32,             // normalized
}

pub struct StepRecord {
    pub observation: BoardObservation,
    pub action: ActionIndex,
    pub visit_distribution: Vec<f32>,
    pub root_value: f32,
    pub reward: f32,
    pub legal_moves: Vec<ActionIndex>,
}

pub struct GameTrajectory {
    pub steps: Vec<StepRecord>,
    pub game_outcome: f32,
    pub model_version: u64,
}
```

**Encoding functions:**
- `encode_board(game_board: &GameBoard, side_to_move: Color) -> BoardObservation` — convert bitboards to float planes
- `move_to_action(mv: &Move) -> ActionIndex` — encode Move as index into ~4672 action space
- `action_to_move(action: ActionIndex) -> Move` — decode index back to Move
- `num_actions() -> usize` — total action space size

**Verify:** `cargo check`

---

### Task 18: MCTS Tree Structure + PUCT Selection

Implement the transient MCTS tree used during search. This task builds the tree data structure and selection logic but does NOT connect to neural nets — it uses a trait for evaluation that will be stubbed.

**New files:**
- `src/mcts/node.rs` — `MCTSNode` struct
- `src/mcts/tree.rs` — `MCTSTree` struct with `run_simulations()`, `select()`, `expand()`, `backpropagate()`, `extract_policy()`
- `src/mcts/puct.rs` — PUCT score computation
- `src/mcts/evaluator.rs` — `Evaluator` trait (abstraction over neural net calls)

**MCTSNode:**
```rust
pub struct MCTSNode {
    pub hidden_state: HiddenState,
    pub visit_count: u32,
    pub total_value: f32,
    pub reward: f32,
    pub prior: f32,
    pub children: Vec<Option<Box<MCTSNode>>>,  // indexed by action
    pub legal_actions: Vec<ActionIndex>,
}
```

**MCTSTree:**
```rust
pub struct MCTSTree {
    root: MCTSNode,
    exploration_constant: f32,  // c in PUCT
    num_simulations: u32,       // typically 800
}

impl MCTSTree {
    pub fn new(root_hidden_state: HiddenState, root_policy: Policy, root_value: f32, legal_actions: Vec<ActionIndex>, config: MCTSConfig) -> Self;
    pub async fn run_simulations(&mut self, evaluator: &dyn Evaluator) -> ();
    pub fn extract_visit_distribution(&self) -> Vec<f32>;
    pub fn root_value(&self) -> f32;
    pub fn select_action(&self, temperature: f32) -> ActionIndex;
}
```

**Evaluator trait:**
```rust
#[async_trait]
pub trait Evaluator: Send + Sync {
    /// h() + f() — encode real board and predict
    async fn root_setup(&self, observation: &BoardObservation) -> (HiddenState, Policy, f32);
    /// g() + f() — dynamics + prediction in one call
    async fn expand_leaf(&self, hidden_state: &HiddenState, action: ActionIndex) -> (HiddenState, f32, Policy, f32);
}
```

**PUCT:**
```rust
pub fn puct_score(q_value: f32, prior: f32, parent_visits: u32, child_visits: u32, c: f32) -> f32;
pub fn select_child(node: &MCTSNode, c: f32) -> usize;  // returns index of best child
```

**Unit tests:**
- PUCT score with known values
- Tree with mock evaluator: verify visit counts increase, value backpropagation correct
- `extract_visit_distribution` sums to 1.0
- `select_action` with temperature=0 picks highest visit count

**Verify:** `cargo test`, `cargo check`

---

## Phase 2: Channels + Storage (Parallel after Phase 1)

### Task 19: Inference Channel + Batching `[PARALLEL]`

Build the inference request/response channel and the batching thread. Uses a stub evaluator implementation that returns random values (real PyO3 integration comes later).

**New files:**
- `src/selfplay/inference.rs` — `InferenceRequest` enum, `InferenceBatcher`, `ChannelEvaluator`

**InferenceRequest:**
```rust
pub enum InferenceRequest {
    RootSetup {
        observation: BoardObservation,
        reply: oneshot::Sender<(HiddenState, Policy, f32)>,
    },
    ExpandLeaf {
        hidden_state: HiddenState,
        action: ActionIndex,
        reply: oneshot::Sender<(HiddenState, f32, Policy, f32)>,
    },
}
```

**InferenceBatcher:**
- Receives from `mpsc::Receiver<InferenceRequest>`
- Collects requests until batch is full OR timeout (configurable)
- Calls an `InferenceBackend` trait (stubbed with random outputs for now)
- Sends results back through each request's `oneshot` channel

**ChannelEvaluator:**
- Implements `Evaluator` trait from Task 18
- Holds a `mpsc::Sender<InferenceRequest>`
- `root_setup()` sends `RootSetup` request, awaits `oneshot` reply
- `expand_leaf()` sends `ExpandLeaf` request, awaits `oneshot` reply

**InferenceBackend trait (for future PyO3):**
```rust
pub trait InferenceBackend: Send {
    fn evaluate_batch(&mut self, requests: Vec<InferenceRequest>);
}

pub struct RandomBackend;  // stub: returns random hidden states, uniform policies, 0.0 values
```

**Tests:**
- Send single request, get response back
- Batch collection respects max size and timeout
- ChannelEvaluator implements Evaluator correctly

**Verify:** `cargo test`, `cargo check`

---

### Task 20: Replay Buffer `[PARALLEL]`

Build the replay buffer that stores game trajectories and supports random sampling for training.

**New files:**
- `src/data/replay_buffer.rs` — `ReplayBuffer`, `TrainingSample`

**ReplayBuffer:**
```rust
pub struct ReplayBuffer {
    trajectories: VecDeque<GameTrajectory>,
    max_trajectories: usize,
    total_steps: usize,
}

impl ReplayBuffer {
    pub fn new(max_trajectories: usize) -> Self;
    pub fn add(&mut self, trajectory: GameTrajectory);
    pub fn sample_batch(&self, batch_size: usize, unroll_k: usize) -> Vec<TrainingSample>;
    pub fn len(&self) -> usize;
    pub fn total_steps(&self) -> usize;
    pub fn checkpoint_to_disk(&self, path: &Path) -> Result<(), io::Error>;
    pub fn load_from_disk(path: &Path) -> Result<Self, io::Error>;
}
```

**TrainingSample:**
```rust
pub struct TrainingSample {
    pub steps: Vec<StepRecord>,  // K+1 consecutive steps
    pub game_outcome: f32,
}
```

**Sampling logic:**
1. Pick random trajectory (weighted by length for uniform step sampling)
2. Pick random step index `t` where `t + unroll_k <= trajectory.len()`
3. Return steps `t..=t+unroll_k`

**Serialization:** Use `serde` + `bincode` for disk checkpoints. Add `serde` dependency.

**Tests:**
- Add trajectories, verify capacity eviction (oldest removed first)
- Sample batch returns correct number of samples
- Each sample has exactly K+1 steps
- Checkpoint to disk and reload matches original
- Empty buffer sampling returns empty vec

**Verify:** `cargo test`, `cargo check`

---

## Phase 3: Game Loop + Coordinator (Sequential, depends on Phase 2)

### Task 21: Self-Play Game Task

Build the game task that plays a single game using MCTS, producing a `GameTrajectory`.

**New files:**
- `src/selfplay/game_task.rs` — `play_game()` async function

**Function signature:**
```rust
pub async fn play_game(
    precomputed: Arc<PrecomputedItems>,
    evaluator: Arc<dyn Evaluator>,
    model_version: u64,
    config: GameConfig,
) -> GameTrajectory
```

**Game loop:**
1. Create fresh `GameBoard` with standard starting position
2. Loop until `game_result != Ongoing`:
   a. Encode current board → `BoardObservation`
   b. Get legal moves, convert to `Vec<ActionIndex>`
   c. Call `evaluator.root_setup(observation)` → (hidden_state, policy, value)
   d. Create `MCTSTree`, run simulations using evaluator
   e. Extract visit distribution + root value
   f. Select action (temperature-based: high early, low late)
   g. Convert action back to `Move`, apply via `process_move()`
   h. Record `StepRecord` in trajectory
3. Set terminal reward (+1/-1) on last step, set `game_outcome`
4. Return `GameTrajectory`

**GameConfig:**
```rust
pub struct GameConfig {
    pub num_simulations: u32,       // MCTS simulations per move (e.g., 800)
    pub exploration_constant: f32,  // PUCT c parameter
    pub temperature_moves: u32,     // use temperature=1.0 for first N moves, then near 0
}
```

**Tests:**
- Play a game with random evaluator, verify trajectory has expected structure
- Each step has observation, action, visit distribution, legal moves
- Game terminates (may take many moves with random play)
- Terminal step has non-zero reward

**Verify:** `cargo test`, `cargo check`

---

### Task 22: Self-Play Coordinator + Training Thread

Build the coordinator that spawns game tasks continuously and the training thread stub.

**New files:**
- `src/selfplay/coordinator.rs` — `SelfPlayCoordinator`
- `src/selfplay/training.rs` — `TrainingThread` (stub)

**SelfPlayCoordinator:**
```rust
pub struct SelfPlayCoordinator {
    precomputed: Arc<PrecomputedItems>,
    inference_tx: mpsc::Sender<InferenceRequest>,
    trajectory_tx: mpsc::Sender<GameTrajectory>,
    model_version: watch::Receiver<u64>,
    config: SelfPlayConfig,
}

pub struct SelfPlayConfig {
    pub max_concurrent_games: usize,
    pub game_config: GameConfig,
}
```

**Coordinator loop:**
- Uses `tokio::sync::Semaphore` to limit concurrent games
- Spawns game tasks continuously as slots open
- Each game task sends its `GameTrajectory` through `trajectory_tx` when done

**TrainingThread (stub):**
- Receives trajectories into replay buffer
- Periodically samples batches (logged but not sent to Python yet)
- Publishes model version increments via `watch::Sender<u64>`
- Disk checkpoints at configurable interval

**Integration binary:**
- `src/bin/selfplay.rs` — starts the full pipeline:
  1. Compute `PrecomputedItems`
  2. Create channels (inference, trajectory, model version)
  3. Spawn inference batcher thread
  4. Spawn replay buffer + training thread
  5. Run coordinator

**Tests:**
- Coordinator spawns up to max concurrent games
- Trajectory channel receives completed games
- Replay buffer accumulates trajectories
- Model version watch updates propagate

**Verify:** `cargo test`, `cargo check`, `cargo run --bin selfplay` runs without panic (plays games with random evaluator)

---

## Phase 4: Integration + Cleanup (Sequential)

### Task 23: End-to-End Integration Test + Cleanup

Wire everything together, run the full pipeline, verify data flow, clean up debug prints.

**Actions:**
- Run `cargo run --bin selfplay` — verify it spawns N games, plays them with random evaluator, collects trajectories, replay buffer grows
- Add integration test that runs 2 games to completion and verifies trajectory contents
- Remove `[DEBUG]` print from `server.rs` (leftover from earlier session)
- Verify all new modules have appropriate `pub` exports
- Run `cargo clippy` and fix warnings
- Update CLAUDE.md with final task status

**Verify:** `cargo test`, `cargo clippy`, `cargo run --bin selfplay` runs cleanly

---

## Task Dependencies

```
Task 17 (data types + modules)
  │
  ├── Task 18 (MCTS tree + PUCT)
  │     │
  │     ├── Task 19 (inference channel) ──┐
  │     │                                 │
  │     └─────────────────────────────────┼── Task 21 (game task)
  │                                       │        │
  ├── Task 20 (replay buffer) ────────────┘        │
  │                                                │
  └────────────────────────────────────────── Task 22 (coordinator + training)
                                                   │
                                              Task 23 (integration + cleanup)
```

**Execution:**
```
Task 17 → Task 18 → [Task 19 | Task 20] (parallel) → Task 21 → Task 22 → Task 23
```

## Task Status

| Task | Status | Notes |
|------|--------|-------|
| 17. Module Structure + Data Types | DONE | data/types.rs, data/encoding.rs, mcts/mod.rs, selfplay/mod.rs |
| 18. MCTS Tree + PUCT | DONE | MCTSNode, MCTSTree, PUCT selection, Evaluator trait |
| 19. Inference Channel + Batching | DONE | InferenceBatcher, ChannelEvaluator, RandomBackend stub |
| 20. Replay Buffer | DONE | VecDeque ring buffer, weighted sampling, bincode checkpoints |
| 21. Self-Play Game Task | DONE | play_game() async fn, MCTS per move, trajectory building |
| 22. Coordinator + Training Thread | DONE | SelfPlayCoordinator with semaphore, TrainingThread stub, selfplay binary |
| 23. Integration + Cleanup | DONE | Clippy fixes in new modules, removed DEBUG print, verified selfplay+tests |
