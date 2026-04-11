# Rust-Python Integration (PyO3)

## Status

- **Done**: Rust engine, MCTS, self-play pipeline with `RandomBackend` (uniform policies, zero values)
- **TODO**: PyO3 bindings, real neural network calls (Tasks 24-26)

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

## TODO: PyO3 Implementation

1. Add `pyo3` (feature: full) to `Cargo.toml`; add `build.rs` and `pyproject.toml`
2. `src/py/mod.rs` — convert `BoardObservation` ↔ numpy; expose `InferenceServer` and `Trainer` class methods
3. `PyO3Backend` in `src/selfplay/inference.rs` — acquire GIL, call batch methods, release GIL
4. `PyTrainerThread` — call `train_batch()`, sync weights via `get_weights()`

## GIL Strategy

Same-process via `InferenceBackend` trait. One GIL acquisition per batch (~32 requests), not per MCTS node (~800). If GIL becomes bottleneck, swap `PyO3Backend` for `ProcessBackend` (Unix socket subprocess) — no other Rust changes needed.

## Known Gotchas

1. **Action plane layout**: 3 planes for `g()` input — exact encoding unspecified. Verify `Move → action planes` tensor layout.
2. **PyO3 panic propagation**: If Python call panics, test whether it poisons the Rust async loop.
3. **Stale weights**: `watch` channel lag is acceptable but verify no race at model version rollover.

## Related Files

- `src/selfplay/inference.rs` — `InferenceBatcher`, `RandomBackend` (stub to replace)
- `python/hyzero/inference/server.py` — `InferenceServer` (Task 26)
- `python/hyzero/training/trainer.py` — `Trainer` (Task 25)
- `docs/TASKS_PYTHON.md` — task specs
