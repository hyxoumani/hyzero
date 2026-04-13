# MCTS & Self-Play Infrastructure

## MCTS Tree Search (Per Move)

**Per-move flow**:
1. Encode board → `BoardObservation` (19 planes: 6 piece types × 2 colors + castling + en passant + side + halfmove clock)
2. Call `evaluator.root_setup(observation)` → (hidden_state, policy, value)
3. Run N simulations (50 for CPU dev, 200+ for GPU production)
4. Extract visit count distribution, select action by visits (temperature scheduling)
5. Apply move, record StepRecord, **discard entire tree** (transient)

## Single MCTS Simulation

1. **SELECT**: Walk down tree using PUCT = Q(s,a) + P(s,a) * sqrt(N(s)) / (1 + N(s,a))
2. **EXPAND**: Call g(s, a) → (s_new, reward) in latent space
3. **EVALUATE**: Call f(s_new) → (policy, value) to initialize children
4. **BACKUP**: Propagate value to root with negation per ply, increment visit counts (two-player zero-sum)

## Inference Batching

Game threads send `InferenceRequest` (RootSetup or ExpandLeaf) to batcher. Batcher collects up to `batch_size` requests or times out (T_timeout), makes single PyO3 call to Python, distributes results via oneshot channels. Reduces GIL acquisitions from ~800/move to ~1/move.

## Replay Buffer

Ring buffer (`VecDeque<GameTrajectory>`) with K-step sampling: pick random trajectory (weighted by length), pick random index t where t+K ≤ len, return steps t..=t+K. Each trajectory tagged with model_version. Serialized with bincode for checkpoints.

## Self-Play Coordinator

Spawns N concurrent games (semaphore gated). Each game: root setup → MCTS → move selection → step record, repeating until terminal (or 300-move limit). Returns GameTrajectory with outcome and model_version. TrainingThread receives trajectories, adds to replay buffer, periodically samples batches for training and publishes model version updates.

### Root Noise for Exploration

Dirichlet(0.03) noise added to root policy before move selection. Implemented via Marsaglia-Tsang Gamma sampling (MuZero paper). **WARNING**: Slow in debug mode; use `--release` builds for end-to-end testing.

## Action Encoding

Action space: 4096 (64×64 from/to, queen default promotion). Encoding: `action = from_sq * 64 + to_sq`. Underpromotion (4672) planned later. Network learns to suppress illegal moves via loss.

## Key Gotchas

1. **Value negation**: Negated per ply during backprop (two-player zero-sum). See `MCTSTree::backpropagate()`: `child.total_value += -value`.
2. **Dirichlet noise overhead**: Marsaglia-Tsang Gamma sampling for Dir(0.03) is very slow in debug mode. For e2e testing, always use `--release` builds or run the binary directly (not via `cargo run`).
3. **Game length with Dirichlet**: Games run ~200 moves now (vs ~60 before); correct behavior for better play, but impacts iteration speed.
4. **Transient tree**: Discarded after each move. Fresh tree per move, no caching.
5. **Batch timeout tuning**: 10ms timeout is empirical. Few concurrent games → small batches → lower GPU utilization.
6. **Stale model data**: Old trajectories in buffer. Loss initially high (bootstrapping).
7. **Action space mismatch**: 4096 logits vs ~40 legal moves. Network learns suppression.
8. **Visit distribution sparsity**: replay buffer stores StepRecord with visit counts for each move. Array is sparse (length = num_visits). PyTrainingThread zero-pads to 4096 before passing to trainer.
9. **Stdout buffering in scripts**: `cargo run` buffers output. For log capture in shell scripts, run the binary directly: `target/release/selfplay` instead of `cargo run --bin selfplay`.
10. **action_to_move requires board context**: `action_to_move(action, board, color)` signature now requires the board state and active color to correctly reconstruct castling and en passant moves. The selfplay game_task.rs already handles this correctly.

## Related Files

- `src/mcts/{mod,node,tree,puct,evaluator}.rs` — tree operations, PUCT selection
- `src/selfplay/{mod,game_task,coordinator,inference,training}.rs` — orchestration
- `src/data/{types,encoding,replay_buffer}.rs` — board observation, action space, buffer
- `docs/TASKS_MCTS_SELFPLAY.md` — detailed task specs (Tasks 17-23)
