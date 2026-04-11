# MCTS & Self-Play Infrastructure

## MCTS Tree Search (Per Move)

**Per-move flow**:
1. Encode board → `BoardObservation` (19 planes: 6 piece types × 2 colors + castling + en passant + side + halfmove clock)
2. Call `evaluator.root_setup(observation)` → (hidden_state, policy, value)
3. Run N simulations (default 800)
4. Extract visit count distribution, select action by visits (temperature scheduling)
5. Apply move, record StepRecord, **discard entire tree** (transient)

## Single MCTS Simulation

1. **SELECT**: Walk down tree using PUCT = Q(s,a) + P(s,a) * sqrt(N(s)) / (1 + N(s,a))
2. **EXPAND**: Call g(s, a) → (s_new, reward) in latent space
3. **EVALUATE**: Call f(s_new) → (policy, value) to initialize children
4. **BACKUP**: Propagate value to root, increment visit counts (values NOT negated per ply)

## Inference Batching

Game threads send `InferenceRequest` (RootSetup or ExpandLeaf) to batcher. Batcher collects up to `batch_size` requests or times out (T_timeout), makes single PyO3 call to Python, distributes results via oneshot channels. Reduces GIL acquisitions from ~800/move to ~1/move.

## Replay Buffer

Ring buffer (`VecDeque<GameTrajectory>`) with K-step sampling: pick random trajectory (weighted by length), pick random index t where t+K ≤ len, return steps t..=t+K. Each trajectory tagged with model_version. Serialized with bincode for checkpoints.

## Self-Play Coordinator

Spawns N concurrent games (semaphore gated). Each game: root setup → MCTS → move selection → step record, repeating until terminal. Returns GameTrajectory with outcome and model_version. TrainingThread receives trajectories, adds to replay buffer, periodically samples batches for training and publishes model version updates.

## Action Encoding

Action space: 4096 (64×64 from/to, queen default promotion). Encoding: `action = from_sq * 64 + to_sq`. Underpromotion (4672) planned later. Network learns to suppress illegal moves via loss.

## Key Gotchas

1. **Value negation**: NOT negated per ply (unusual, intentional). Verify during training.
2. **Transient tree**: Discarded after each move. Fresh tree per move, no caching.
3. **Batch timeout**: Few concurrent games → small batches → lower GPU utilization.
4. **Stale model data**: Old trajectories in buffer. Loss initially high (bootstrapping).
5. **Action space mismatch**: 4096 logits vs ~40 legal moves. Network learns suppression.

## Related Files

- `src/mcts/{mod,node,tree,puct,evaluator}.rs` — tree operations, PUCT selection
- `src/selfplay/{mod,game_task,coordinator,inference,training}.rs` — orchestration
- `src/data/{types,encoding,replay_buffer}.rs` — board observation, action space, buffer
- `docs/TASKS_MCTS_SELFPLAY.md` — detailed task specs (Tasks 17-23)
