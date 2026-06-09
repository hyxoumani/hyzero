# Self-Play Coordinator

The self-play subsystem continuously generates game trajectories that feed the
trainer. Code lives in `src/selfplay/{coordinator,game_task,inference}.rs` and is
wired together in `src/bin/selfplay.rs`.

## Coordinator

`SelfPlayCoordinator` (`coordinator.rs`) spawns `max_concurrent_games`
**persistent** tokio tasks (no per-game spawn, no semaphore). Each task loops
forever:

```
loop {
    version = *model_version_rx.borrow();   // current weights version
    traj    = play_game(precomputed, evaluator, version, game_config).await;
    if trajectory_tx.send(traj).await.is_err() { break; }   // channel closed → stop
}
```

`run()` awaits all `JoinHandle`s; tasks exit when the trajectory channel closes.
The watch channel means every new game picks up the latest published weights.
`SelfPlayConfig` carries `max_concurrent_games` and a `GameConfig`.

The selfplay binary reserves one game slot for evaluation: with
`HYZERO_GAMES = N`, self-play concurrency is `N − 1` (min 1).

## Game Task

`play_game(precomputed, evaluator, model_version, config)` (`game_task.rs`) plays
one game and returns a `GameTrajectory { steps, game_outcome, model_version,
is_draw }`. Per ply it: encodes the board → `root_setup` → builds an `MCTSTree`
→ runs simulations → `select_action` (temperature schedule via
`temperature_moves`) → records a `StepRecord` → applies the move.

`GameConfig` fields: `num_simulations`, `exploration_constant` (1.5),
`temperature_moves`, `replay_dir` (Option — opt-in MCTS replay capture).

### Termination Paths

The loop ends on the first of:
1. **Terminal state** — `GameResult::Checkmate(color)` → `outcome = ±1`,
   `is_draw = false`.
2. **Non-checkmate terminal** — stalemate, threefold repetition, 50-move, the
   300-ply cap (`MAX_GAME_LENGTH = 300`), or insufficient material →
   `is_draw = true`.

**Outcome for non-checkmate games is `0.0` by default** (AlphaZero-style: only
real checkmates produce non-zero value targets). Material shaping is **opt-in**
via `HYZERO_MATERIAL_SHAPING=1`, which substitutes
`tanh(Δmaterial / HYZERO_MATERIAL_SHAPING_SCALE)`. Shaping is off by default
because it previously (a) mislabeled shaped draws in the PGN and (b) reinforced a
shuffle/passivity attractor (rewarding the material-leading side for drawing by
repetition). **Adjudication was removed** for the same passivity reason — games
play to checkmate, a non-checkmate terminal, or the 300-ply cap only.

The terminal reward is written onto the last step in last-step POV
(`reward = game_outcome · side_sign`); the trainer applies further ply-flipping
during batch assembly (see [Neural Networks](neural-networks.md)).

`play_game_dual(precomputed, white_eval, black_eval, config)` plays one game with
two distinct evaluators and returns a `DualGameOutcome { game_outcome, moves, ...}`
— used by the [Elo Ladder](elo-ladder-eval.md).

## Inference Batching

`InferenceBatcher` (`inference.rs`) collects `InferenceRequest`s (RootSetup or
ExpandLeaf) from many game tasks, batches up to `max_batch_size` or until
`batch_timeout_ms`, makes a single backend call, and distributes results via
oneshot channels. `BatcherConfig` defaults in the binary: `max_batch_size = 32`,
`batch_timeout_ms = 10`. This reduces PyO3/GIL acquisitions from ~per-node to
~per-batch. Backends: `RandomBackend` (test stub), `PyO3Backend` (real network),
and `SwappableBackend` (hot-swaps the underlying backend on promotion). The
binary runs **three** batchers: challenger/self-play, champion, and opponent
(Elo pool).

## Training Pipeline (Rust side)

`PyTrainingThread` (`src/py/training.rs`) owns the in-memory `ReplayBuffer` and
drives training:

- Receives `GameTrajectory`s, adds them to the buffer (`max_replay_trajectories
  = 10_000`).
- Per received game, runs `train_steps_per_game = 16` steps once
  `min_samples = 200` trajectories are buffered; each step samples a batch
  (`train_batch_size = 256`, env `HYZERO_TRAIN_BATCH_SIZE`), assembles arrays
  (`unroll_k = 5`), and calls the Python `Trainer.train_batch`.
- On each completed train batch it increments `model_version`, fetches weights,
  and publishes them via the watch channel (consumed by the inference servers).
- Every `checkpoint_interval_steps = 50` steps it saves
  `checkpoints/model_v{:06}.pt`, keeping the newest `checkpoint_keep_last = 5`
  (older ones pruned). `latest_checkpoint_path` is shared with the eval task so
  promotions can archive the right file.

`from_default_config(..., resume_checkpoint)` optionally resumes: it calls the
Python `load_checkpoint`, reads `model_version` back, and broadcasts the restored
weights so game loops use them immediately.

## Replay Buffer Sampling

`ReplayBuffer::sample_batch(batch_size, unroll_k)` does prioritized sampling:
decisive (non-draw) trajectories are oversampled to a fraction set by
`HYZERO_DECISIVE_SAMPLE_FRAC` (default 0.25). Within each pool, trajectories are
weighted by valid start positions (`steps.len() − unroll_k`). Falls back to
uniform when no decisive trajectories exist. The buffer is **memory-only**; the
only on-disk replay path is the opt-in capture in [Replay Subsystem](replay-subsystem.md).

## Related

- [MCTS](mcts.md) — the per-move tree search inside `play_game`
- [Neural Networks](neural-networks.md) — the trainer the pipeline drives
- [Elo Ladder Evaluation](elo-ladder-eval.md) — the eval task sharing a game slot
- [Replay Subsystem](replay-subsystem.md) — training buffer vs. `.replay` capture
- `src/selfplay/{coordinator,game_task,inference}.rs`, `src/bin/selfplay.rs`, `src/py/training.rs`
