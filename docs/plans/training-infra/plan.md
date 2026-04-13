# Plan: Training Infrastructure Scale-Up

## Approach

Replace the semaphore-gated coordinator with N independent long-lived game loops,
add a rolling-window checkpoint system with Python-side save/load, and add a
dedicated evaluation task that periodically benchmarks the current model against
RandomEvaluator. The three components are independent and can be built in any order;
only the coordinator change touches the binary entry point.

---

## SOTA Cross-Reference

### Self-Play Worker Management

**MuZero (DeepMind paper)**: 3,000 TPU actors each running independently, sending
trajectories to a central learner via a FIFO replay buffer. Each actor is stateless:
plays games, sends trajectories, reads the latest weights at the start of each game.
No semaphore or synchronization between actors.

**AlphaZero**: Similar pattern — fixed-N independent actors, no rendezvous between
games. Weight staleness of O(1000 training steps) is acceptable.

**muzero-general (open-source)**: Ray actors, one process per game. Each actor holds
its own copy of the model and periodically requests the latest weights from a shared
"SharedStorage" object. Identical to the watch channel pattern already in hyzero.

**LeelaZero / KataGo**: C++ worker threads (not async tasks), each plays one game
at a time, blocks on inference, restarts immediately. No concurrency within a single
worker thread; batching happens at the inference server.

**Mapping to hyzero**: Our `ChannelEvaluator` + `InferenceBatcher` already implements
the correct batching architecture. The coordinator's semaphore is unnecessary
complexity — replacing it with N persistent `tokio::spawn` tasks that each loop
`play_game → send → repeat` is the SOTA pattern. Weight reads via `watch::Receiver`
are already correct (staleness of one game is acceptable).

### Checkpoint Strategy

**MuZero**: Checkpoints every N training steps. Stores weights + optimizer + step
counter. No rolling window — all checkpoints kept (storage is cheap at Google scale).
Inference weights are a separate "fast" snapshot for distribution.

**KataGo**: Saves a checkpoint every 1,000 gradient steps. Keeps the last 3 for
rollback safety, plus permanent snapshots at logarithmic intervals (step 1K, 2K, 4K,
8K, ...) for ablation.

**muzero-general**: Saves `latest_model.checkpoint` (overwrite) plus `model_X.checkpoint`
every N steps. Rolling window of 20 snapshots before pruning.

**What to build**: A rolling window of M checkpoints (M=5 is reasonable). Each
checkpoint is a `.pt` file named by model_version. Python `Trainer` already has
`save_checkpoint()` / `load_checkpoint()`. The Rust side needs to trigger saves at
the right cadence and manage the window of filenames. The metrics dict in
`save_checkpoint()` is the right slot for attaching eval results.

### Evaluation Strategy

**AlphaZero**: Evaluation used only to decide whether to replace the "best" model:
400 games against the previous checkpoint; if win rate >= 55%, replace. Eval runs on
a dedicated set of workers, not on self-play workers.

**MuZero**: No explicit evaluation in the paper for self-play; Elo computed
post-hoc on logs. Practical implementations (muzero-general) run a periodic eval
task against a fixed policy (random or prior checkpoint).

**KataGo**: Uses a continuous Elo estimation via match play against all prior
checkpoints, averaged with Bradley-Terry. Very expensive; not appropriate for hyzero.

**EfficientZero**: Evaluation is integrated — a "test worker" runs in parallel,
plays against the current model, and logs episodic reward. Results written to a shared
metrics store.

**What to build for hyzero**: A lightweight Rust `EvaluationTask` that:
1. Every E games (configurable), grabs the current model version.
2. Plays E_games evaluation games using `play_game()` with a `ChannelEvaluator` for
   the current model vs `RandomBackend` for the opponent.
3. Logs win rate and average game length.
4. Optionally attaches metrics to the next checkpoint save.

The eval task does NOT block training. It uses a separate `ChannelEvaluator` connecting
to the same `InferenceBatcher`. "vs previous checkpoint" is deferred.

### Weight Distribution

**MuZero / AlphaZero**: Weights pushed to actors after every N gradient steps.
Actors discard the old model and use the new one at the start of the next game.
A staleness of 1 game (~200 moves) is irrelevant in practice.

**KataGo**: Each worker checks shared memory for a "current model" pointer before
starting a game. Memory-mapped weights for zero-copy distribution.

**Mapping to hyzero**: The `watch::Sender<u64>` (version) + `watch::Sender<Option<Vec<u8>>>` 
(weight bytes) + weight-loader task in `selfplay.rs` is the correct SOTA pattern.
Workers read `model_version` at game start from `watch::Receiver<u64>`. The weight
bytes are pushed to `InferenceServer.load_weights()` by a dedicated weight-loader
tokio task. This design matches AlphaZero/MuZero intent. No changes needed here.

### Training-to-Self-Play Ratio

**MuZero**: ~0.1 training steps per MCTS simulation (paper Table A1). At 800 sims/move
and ~150 moves/game, that is ~120,000 sims/game → ~12,000 gradient steps per game.
At batch size 2048, that is far more gradient steps than we can afford at toy scale.

**muzero-general default**: 1 training step per game received. Configurable.

**EfficientZero**: "reuse_factor" config controls how many training steps per game.
Default is 1-4.

**Mapping to hyzero**: `train_steps_per_game = 4` in `PyTrainingThread::from_default_config()`
(line 247 of `src/py/training.rs`) is already sensible. For scale-up, expose this as
a config parameter rather than hardcoded.

---

## Subtasks

### 1. Simplify Coordinator (N Independent Game Loops)

**Files**:
- `src/selfplay/coordinator.rs` — rewrite `SelfPlayCoordinator::run()`
- `src/selfplay/mod.rs` — no signature changes; `SelfPlayConfig` stays identical
- `src/bin/selfplay.rs` — no changes needed

**Current code (lines 55-83 of coordinator.rs)**:
```
loop {
    let permit = semaphore.acquire_owned().await;
    tokio::spawn(async move { play_game(...).await; send; drop(permit) });
}
```
This continuously spawns new tasks whenever a slot opens, relying on the semaphore
to cap total. The problem: each game is a short-lived task; spawning overhead
accumulates; model_version is captured at spawn time but games can queue behind the
semaphore.

**Target design (N persistent loops)**:
```
for _ in 0..config.max_concurrent_games {
    tokio::spawn(async move {
        loop {
            let version = *model_version_rx.borrow();
            let traj = play_game(..., version, ...).await;
            if trajectory_tx.send(traj).await.is_err() { break; }
        }
    });
}
```

**Changes**:
- Remove `Semaphore` import and usage
- `SelfPlayCoordinator::run()` spawns exactly `config.max_concurrent_games` tasks
- Each spawned task holds its own `watch::Receiver<u64>` clone and loops forever
- `run()` itself joins on a `JoinSet` or waits on any completion signal; simplest is
  to block on the trajectory sender: `loop { tokio::time::sleep(FOREVER).await }`
  until sender errors, then abort all handles
- `model_version_rx` must be cloned `N` times before spawning (use `watch::Receiver::clone()`)

**Why better**: No spawn overhead per game; model_version read at loop-top is always
fresh; no shared mutable state; idiomatic Tokio pattern.

**Tests**:
- Existing `test_coordinator_produces_trajectories` covers correctness; update it to
  verify N=2 tasks each complete at least one game
- Add `test_coordinator_reads_fresh_model_version`: send a version update mid-run,
  verify subsequent trajectories use the new version

**Dependencies**: None

---

### 2. Checkpoint System (Rolling Window, Rust Side)

**Files**:
- `src/py/training.rs` — add checkpoint window management to `PyTrainingThread`
- `src/py/training.rs` — add `CheckpointConfig` struct (or extend existing fields)

**Current state**:
`PyTrainingThread::run()` (lines 327-342 of `src/py/training.rs`) saves a checkpoint
every 50 training steps to `checkpoints/model_v{version}.pt`. There is no window
management — files accumulate indefinitely.

**Changes**:
1. Add `checkpoint_keep_last: usize` field to `PyTrainingThread` (recommend default 5).
2. Maintain `checkpoint_files: VecDeque<String>` tracking saved filenames.
3. After each checkpoint save, push the new filename and pop+delete the oldest if
   `checkpoint_files.len() > checkpoint_keep_last`.
4. Expose `checkpoint_interval_steps` as a config parameter (currently hardcoded 50);
   add it to `from_default_config()`.
5. Name files with zero-padded version for lexicographic sort:
   `checkpoints/model_v{:06}.pt`.

**What stays the same**:
- `Trainer.save_checkpoint()` / `load_checkpoint()` in Python — already complete
- The `save_checkpoint` PyO3 call at line 334 — just update the path format
- `weight_tx`/`version_tx` publish cadence — unchanged

**Tests**:
- `test_checkpoint_window_prunes_oldest`: mock `save_checkpoint` call, verify that
  after 6 saves with keep_last=5, the oldest file path is deleted
- Unit test can stay in `src/py/training.rs #[cfg(test)]`; use `tempdir` for paths

**Dependencies**: None

---

### 3. Checkpoint System (Python: load_checkpoint integration)

**Files**:
- `python/hyzero/training/trainer.py` — `load_checkpoint()` already exists (lines 188-204)
- No changes needed; method is complete and correct

**What is missing**: No Rust-side startup path that loads a checkpoint on resume.

**Changes**:
- Add `resume_checkpoint: Option<String>` to `PyTrainingThread::from_default_config()`
  (or a separate `from_checkpoint()` constructor).
- If `Some(path)`, call `trainer.load_checkpoint(path)` via PyO3 after construction
  and set `self.model_version` from the returned `model_version` field.
- Also push the loaded weights into `weight_tx` so `InferenceServer` starts with the
  restored weights, not random initialization.

**Tests**:
- `#[ignore = "requires hyzero Python package"]` integration test that saves a
  checkpoint, constructs a new `PyTrainingThread` with `from_checkpoint()`, verifies
  `model_version` matches what was saved.

**Dependencies**: Subtask 2 (checkpoint saving must work first)

---

### 4. Evaluation Task (Rust, vs RandomEvaluator)

**Files**:
- `src/selfplay/evaluation.rs` — new file
- `src/selfplay/mod.rs` — add `pub mod evaluation` and re-export
- `src/bin/selfplay.rs` — spawn eval task after coordinator

**What to build**:
An `EvaluationTask` struct with a `run()` method:

```
pub struct EvaluationConfig {
    pub eval_interval_games: usize,  // how many self-play games between eval runs
    pub eval_games: usize,           // games per eval run (recommend 10)
    pub num_simulations: u32,        // MCTS sims per move for eval (can be lower, e.g. 50)
}

pub struct EvaluationTask {
    precomputed: Arc<PrecomputedItems>,
    model_evaluator: Arc<dyn Evaluator>,   // ChannelEvaluator → InferenceBatcher
    games_played_rx: watch::Receiver<u64>, // updated by training thread
    config: EvaluationConfig,
}
```

**Data flow**:
```
TrainingThread (game counter)  →  watch::Sender<u64> (games_played)
                                         ↓
                              EvaluationTask::run()
                              every eval_interval_games:
                                ┌─ play_game(model_evaluator) ×E  (White = model)
                                │  play_game(random_evaluator) ×E (White = random)
                                └─ log: win_rate, avg_game_length, model_version
```

The simplest evaluation: play E games where one side uses the `ChannelEvaluator`
(backed by the current model) and the other side uses `RandomBackend` (zero value,
uniform policy). The model plays both colors (E/2 each) to avoid color bias.

**Implementation notes**:
- Reuse `play_game()` from `game_task.rs` — no changes needed
- `RandomBackend` already in `src/selfplay/inference.rs` — wrap it in an `Evaluator`
  that calls the backend directly (no channel needed for the opponent; it's synchronous
  and trivial). Simplest: create a `RandomEvaluator` struct implementing `Evaluator`
  (the same one that exists in the test modules, just make it public in a new module
  `src/selfplay/evaluation.rs`).
- `EvaluationTask` does not need to observe training-side channels except for the
  "games played" trigger. Alternatively, simpler trigger: sleep for a fixed interval
  (e.g., 60s between eval runs at toy scale) — avoids a new channel.

**Recommended trigger**: `tokio::time::interval(eval_interval_secs)` — no new channels.

**Metrics**: Print to stdout in structured form:
```
[eval] v42 win_rate=0.72 (13/18) avg_length=87.3 random_baseline
```

Future: write to a CSV file; add comparison against previous checkpoint.

**Tests**:
- `test_evaluation_completes` in `#[cfg(test)]` block: run 2 eval games with
  `num_simulations=2` and `RandomEvaluator` vs `RandomEvaluator`, verify win_rate is
  in [0.0, 1.0] and avg_length > 0

**Dependencies**: Subtask 1 (coordinator simplification) should be done first so
the `ChannelEvaluator` is stable, but technically the eval task compiles independently.

---

### 5. Config Consolidation

**Files**:
- `src/bin/selfplay.rs` — replace hardcoded values with a top-level config struct
- `src/selfplay/coordinator.rs` — no changes (config struct stays)
- `src/py/training.rs` — expose `checkpoint_interval_steps` and `checkpoint_keep_last`
  as constructor args (or a config struct)

**Current hardcoded values in `selfplay.rs`**:
- Line 56: `max_batch_size: 32`
- Line 57: `batch_timeout_ms: 10`
- Line 95-100: `max_concurrent_games: 4`, `num_simulations: 50`, `temperature_moves: 15`
- `from_default_config` defaults: `train_steps_per_game=4`, `min_samples=200`

**Changes**:
- Add a `SelfPlayRunConfig` struct in `src/bin/selfplay.rs` (binary-only; not in lib)
  collecting all these. Default impl with the current values. Parse from env vars or
  a simple JSON file for easy tuning without recompiling.

**Tests**: No new Rust tests (binary-only config). Verify by running
`cargo run --bin selfplay` and observing log output.

**Dependencies**: None; can be done at any time.

---

## Data Flow After Changes

```
┌─────────────────────────────────────────────────────────┐
│                    hyzero selfplay process               │
│                                                          │
│  N game loops (tokio tasks)                              │
│  ┌──────────────┐   ┌──────────────┐                    │
│  │ game_loop_1  │   │ game_loop_N  │  ...               │
│  │  loop:       │   │  loop:       │                     │
│  │  version=rx  │   │  version=rx  │                    │
│  │  traj=play() │   │  traj=play() │                    │
│  │  tx.send(t)  │   │  tx.send(t)  │                    │
│  └──────┬───────┘   └──────┬───────┘                    │
│         │                  │                            │
│         └──────────┬───────┘                            │
│                    │ mpsc::Sender<GameTrajectory>        │
│                    ▼                                     │
│         ┌──────────────────┐                            │
│         │ PyTrainingThread │                            │
│         │  recv trajectory │                            │
│         │  add to buffer   │                            │
│         │  train (4×/game) │                            │
│         │  every 50 steps: │                            │
│         │   save ckpt      │───→ checkpoints/model_v*.pt│
│         │   prune window   │    (keep last 5)           │
│         │  version_tx.send │                            │
│         │  weight_tx.send  │                            │
│         └──────────────────┘                            │
│              │           │                              │
│              │version    │weights                       │
│              │(watch)    │(watch)                       │
│              ▼           ▼                              │
│         game loops   weight-loader task                 │
│         (read at     (calls InferenceServer             │
│         loop-top)     .load_weights())                  │
│                           │                             │
│                           ▼                             │
│                  ┌─────────────────┐                    │
│                  │ InferenceBatcher│                    │
│                  │  (PyO3Backend)  │                    │
│                  └────────┬────────┘                    │
│                           │  inference requests         │
│                           │  (from game loops +         │
│                           │   eval task)                │
│                           ▼                             │
│                  Python InferenceServer                  │
│                                                         │
│  Evaluation task (tokio, every 60s)                     │
│  ┌─────────────────────────────────┐                    │
│  │  play E games: model vs random  │                    │
│  │  log: win_rate, avg_len, version│                    │
│  └─────────────────────────────────┘                    │
└─────────────────────────────────────────────────────────┘
```

---

## Config Parameters with Recommended Defaults

| Parameter | Location | Default | Notes |
|-----------|----------|---------|-------|
| `max_concurrent_games` | `SelfPlayConfig` | 4 | Increase to 8-16 for GPU runs |
| `num_simulations` | `GameConfig` | 800 (default), 50 (binary) | Use 200+ for GPU |
| `temperature_moves` | `GameConfig` | 30 (default), 15 (binary) | Moves before greedy |
| `max_batch_size` | `BatcherConfig` | 32 | Tune with GPU utilization |
| `batch_timeout_ms` | `BatcherConfig` | 10ms | Lower for more games |
| `train_steps_per_game` | `PyTrainingThread` | 4 | MuZero ratio ~10-100 for prod |
| `min_samples` | `PyTrainingThread` | 200 | Steps before training starts |
| `checkpoint_interval_steps` | `PyTrainingThread` | 50 | Training steps between saves |
| `checkpoint_keep_last` | `PyTrainingThread` | 5 | Rolling window size |
| `eval_interval_secs` | `EvaluationTask` | 60 | Seconds between eval runs |
| `eval_games` | `EvaluationTask` | 10 | Games per eval run |
| `eval_num_simulations` | `EvaluationTask` | 50 | Sims for eval (can be lower) |

---

## Testing Strategy

**Unit tests (fast, no Python)**:
- Coordinator: N game loops each produce at least 1 trajectory (existing test + update)
- Checkpoint window: mock save paths, verify pruning after window overflows
- Evaluation: 2 games with `RandomEvaluator` vs `RandomEvaluator`, verify metrics

**Integration tests (require Python)**:
- `#[ignore = "requires hyzero Python package"]`
- Checkpoint save+load roundtrip: train 1 batch, save, reload, verify model_version
- Weight distribution: save checkpoint, load into InferenceServer, verify inference runs

**End-to-end** (`bash scripts/e2e_test.sh`):
- Existing e2e covers: 5 games, 50 sims, loss decreases
- After this work: extend to verify checkpoint file appears in `checkpoints/` and
  eval output appears in stdout

---

## What to Defer

| Item | Why defer |
|------|-----------|
| Eval vs previous checkpoint | Requires running two `ChannelEvaluator` instances concurrently pointing to different InferenceServer weights — needs weight versioning infrastructure |
| Elo tracking (Bradley-Terry) | 10 eval games per run is too few for meaningful Elo; track win rate first |
| Stockfish integration | Adds a binary dependency; build internal baseline first |
| Separate eval worker process | Not needed until GIL contention is measured as a bottleneck |
| Convergence detection (loss plateau) | Requires loss smoothing over a window; defer until basic checkpointing is stable |
| LR scheduling | Adam without schedule is fine for early training; add cosine decay later |
| Distributed actors (multiple machines) | No infrastructure for this yet; single-process is the right starting point |

---

## Risks

1. **GIL contention with eval task**: eval games use `ChannelEvaluator`, which feeds
   into the same `InferenceBatcher` and therefore the same GIL acquisition. If eval
   runs N games concurrently with self-play, this increases GIL pressure. Mitigation:
   run eval sequentially (one game at a time), not in parallel with itself.

2. **Checkpoint file deletion race**: if the training thread is checkpointing while
   a restart is loading, a partial file could be deleted. Mitigation: write to a temp
   file then `rename` atomically; `std::fs::rename` is atomic on POSIX.

3. **watch channel weight broadcast race**: if the weight-loader task is slow
   (loading 50MB weights into InferenceServer), game workers may start games with
   stale weights for 1-2 extra games. This is acceptable by SOTA design (staleness
   of O(1 game) is irrelevant). No fix needed; document as expected behavior.

4. **game_loop tasks not cancelling cleanly**: if `trajectory_tx` is dropped (on
   shutdown), the `send().await` in each game loop will return `Err`, which triggers
   `break`. This is correct. Verify `play_game()` itself doesn't leak resources on
   early termination (it doesn't: it allocates only stack-local structures).

5. **Checkpoint path collision**: multiple runs writing to the same `checkpoints/`
   directory will overwrite files. Mitigation: add a `run_id` prefix (timestamp or
   UUID) to checkpoint filenames. Defer to later run if only one process at a time.
