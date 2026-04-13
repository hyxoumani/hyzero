# Rust-Python Integration (PyO3)

## Status

- **Done**: Rust engine, MCTS, self-play pipeline with `RandomBackend` (uniform policies, zero values)
- **Done**: PyO3 bindings, real neural network calls via PyO3Backend and PyTrainingThread (Tasks 24-28)

## Data Flow

```
N Game Threads → InferenceRequest → InferenceBatcher
                                        ├─ collect up to batch_size (or timeout_ms)
                                        └─ single PyO3 call
                                              ↓
                                    Python InferenceServer
                                        ├─ root_setup_batch([B,19,8,8])
                                        │   → hidden[B,64,8,8], policy[B,4096], value[B]
                                        └─ expand_leaf_batch(hidden[B,64,8,8], actions[B,3,8,8])
                                            → next_hidden, rewards, policy, value
                                              ↓
                                    Batcher distributes via oneshot channels
```

## Batching Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `batch_size` | 32–64 | Higher = better GPU utilization, higher latency |
| `timeout_ms` | 10–50 | Partial batch sent after timeout |
| `max_concurrent_games` | 8–16 | Semaphore gate |

## Data Contracts

All arrays `float32` numpy. Policies are post-softmax. Values tanh-bounded [-1, 1].

```
RootSetup:   observations[B,19,8,8] → hidden[B,64,8,8], policies[B,4096], values[B]
ExpandLeaf:  hidden[B,64,8,8] + actions[B,3,8,8] → next_hidden, rewards[B], policies[B,4096], values[B]
```

## Weight Sync

```
Trainer.train_batch() → Trainer.get_weights() → bytes (torch.save)
    ↓ (PyO3 / channel)
InferenceServer.load_weights(bytes) → deserialize → load state dicts
```

Model version tracked via `watch::Sender<u64>`. Brief stale-weight lag is acceptable.

## PyO3 Implementation (DONE)

1. Added `pyo3 0.28` (feature: full) and `numpy 0.28` to `Cargo.toml`
2. `src/py/mod.rs` — conversion utilities for `BoardObservation` ↔ numpy; imports of `InferenceServer` and `Trainer` classes
3. `src/py/inference_backend.rs` — `PyO3Backend` struct implementing `InferenceBackend` trait
   - Acquires GIL, calls Python `InferenceServer.root_setup_batch()` and `expand_leaf_batch()`
   - Converts `InferenceRequest` → numpy arrays, distributes results via oneshot channels
4. `src/py/training.rs` — `PyTrainingThread` replaces stub
   - Calls `trainer.train_batch(batch)` with zero-padded visit distributions
   - Syncs weights via `trainer.get_weights()` → `watch::Sender<bytes>`
   - Weight loading task calls `inference_server.load_weights(bytes)`

## GIL Strategy

Same-process via `InferenceBackend` trait. One GIL acquisition per batch (~32 requests), not per MCTS node (~800). If GIL becomes bottleneck, swap `PyO3Backend` for `ProcessBackend` (Unix socket subprocess) — no other Rust changes needed.

## Known Gotchas

1. **action_to_move signature changed**: Now requires `(action, board, color)` instead of just `action`. The board state and active color are necessary to correctly reconstruct castling and en passant moves. Callers in encoding.rs already updated.
2. **Visit distribution padding**: StepRecord visit_distribution is sparse (length = num_visits < 4096). PyTrainingThread zero-pads to 4096 for batch assembly. Trainer.train_batch() expects dense [B, K+1, 4096] array.
3. **PyO3 reference counting**: InferenceServer Py<PyAny> handle must be cloned with `clone_ref()` (not `clone()`) to share between PyO3Backend and weight loading task. Regular `clone()` fails.
4. **pyo3 0.28 vs 0.22**: Version 0.28 is cleaner — no abi3-py38 needed, works with Python 3.9+. abi3 mode adds complexity; unnecessary here.
5. **GIL per batch**: One GIL acquisition per 32 requests (~800/move), not per MCTS node. If bottleneck detected, swap for ProcessBackend (Unix socket subprocess, no Rust changes needed).

## Related Files

- `src/selfplay/inference.rs` — `InferenceBatcher`, `RandomBackend` (stub to replace)
- `python/hyzero/inference/server.py` — `InferenceServer` (Task 26)
- `python/hyzero/training/trainer.py` — `Trainer` (Task 25)
- `docs/TASKS_PYTHON.md` — task specs
