# Research: PyO3 Integration

## Problem Statement

The Rust self-play loop currently uses `RandomBackend` — it returns uniform policies and zero
values for every board position. To make the engine learn, the `InferenceBatcher` must call
the real Python `InferenceServer` for neural network evaluation, and the `TrainingThread` must
drive the Python `Trainer` to update weights from replay buffer samples. This task connects
those two sides through PyO3 in the same process.

## Current State

### Inference path (Rust side)

- `src/selfplay/inference.rs` — `InferenceBackend` trait (line 23-25):
  ```rust
  pub trait InferenceBackend: Send {
      fn evaluate_batch(&mut self, requests: Vec<InferenceRequest>);
  }
  ```
- `RandomBackend` is the only implementation. It replies with uniform policy and 0.0 value.
- `InferenceBatcher::run()` blocks on an mpsc channel, collects up to `max_batch_size` requests
  or fires a timeout, then calls `backend.evaluate_batch(batch)` synchronously.
- `ChannelEvaluator` is the Evaluator-trait glue that game tasks use (async send + oneshot recv).
- Batcher runs on a tokio task; `evaluate_batch` is called from within that task. Any blocking
  work inside `evaluate_batch` blocks the batcher task — acceptable since GIL acquisition is the
  only source of blocking here and we do it once per batch.

### Training path (Rust side)

- `src/selfplay/training.rs` — `TrainingThread::run()` receives `GameTrajectory` from mpsc,
  adds to `ReplayBuffer`, samples batches when `total_steps >= min_samples`, logs stats, and
  increments `model_version` on the `watch::Sender<u64>`. The Python call is a stub (TODO comment).
- Replay buffer sampling returns `Vec<KStepSample>` (from `src/data/replay_buffer.rs`). The
  samples need to be serialized into numpy arrays matching Python `Trainer.train_batch()` format.

### Python side

- `python/hyzero/inference/server.py` — `InferenceServer`:
  - `root_setup_batch(observations: np.ndarray[B,19,8,8]) -> (hidden[B,64,8,8], policies[B,4096], values[B])`
  - `expand_leaf_batch(hidden[B,64,8,8], actions[B,3,8,8]) -> (new_hidden, rewards[B], policies[B,4096], values[B])`
  - `load_weights(bytes)` — accepts bytes from `Trainer.get_weights()`
- `python/hyzero/training/trainer.py` — `Trainer`:
  - `train_batch(batch: dict) -> dict` — batch is numpy arrays, returns loss scalars
  - `get_weights() -> bytes` — torch.save of state_dicts

### Data encoding

`src/data/encoding.rs`:
- `encode_board()` → `BoardObservation { planes: Vec<f32> }` of length `19 * 64`
- `encode_action_spatial(ActionIndex) -> [f32; 3 * 64]`; plane layout: [src_sq, dst_sq, promotion_flag]
- Layout is flat `[plane, square]` row-major, which numpy will see as `(3, 8, 8)` after reshape.
  Rust produces planes in order 0, 1, 2 with 64 floats each — this is `C-contiguous [3,64]`,
  trivially reshaped to `[3,8,8]`. Must confirm Python `g()` forward expects `[B,3,8,8]` channel-first
  (it does — see `DynamicsNetwork` which takes `torch.cat([hidden, action], dim=1)` → `[B,67,8,8]`).

### Key data shape table

| Rust type | Flat size | numpy reshape |
|-----------|-----------|---------------|
| `BoardObservation.planes` | 1216 (19×64) | `[19, 8, 8]` |
| `HiddenState.data` | 4096 (64×64) | `[64, 8, 8]` |
| `Policy` (Vec<f32>) | 4096 | `[4096]` |
| `encode_action_spatial` result | 192 (3×64) | `[3, 8, 8]` |

### Batch assembly for train_batch

`KStepSample` (from `src/data/replay_buffer.rs`) contains the per-step fields needed for
training. Each sample maps to:
- `observations[b]` = `steps[0].observation.planes` reshaped to `[19,8,8]`
- `actions[b,k]` = `encode_action_spatial(steps[k].action)` reshaped to `[3,8,8]`
- `target_policies[b,k]` = `steps[k].visit_distribution` (already `Vec<f32>` of len 4096)
- `target_values[b,k]` = `steps[k].root_value`
- `target_rewards[b,k]` = `steps[k].reward`

## Relevant Patterns

1. **Trait objects for backend**: `Box<dyn InferenceBackend>` in `InferenceBatcher`. New
   `PyO3Backend` must implement the same trait with no change to the batcher.
2. **Cargo feature flags**: The project uses minimal dependencies in `Cargo.toml`. PyO3 should
   be a regular (non-optional) dependency since the binary always needs Python inference.
   If a `mock` feature is desired for CI without Python, it can be added later — not in scope.
3. **No `build.rs` needed for `pyo3` >= 0.20 with `auto-initialize` feature**: PyO3 can find
   the Python installation via environment variables (`PYO3_PYTHON` or `python3` on PATH).
   A `build.rs` is only needed for embedding a specific interpreter. We should NOT add one unless
   required.
4. **Blocking inside tokio task**: `evaluate_batch` is called synchronously inside a tokio task
   (the batcher). GIL acquisition is a blocking mutex. Since the batcher runs on the multi-thread
   scheduler, this blocks one tokio worker thread for the duration of the Python call (~10-50ms).
   Acceptable for now; use `tokio::task::spawn_blocking` if contention is observed.
5. **watch channel for model version**: Pattern already established in `TrainingThread` —
   `version_tx.send(new_version)` after each train step. PyTrainingThread follows same pattern.

## Constraints

- `Cargo.lock` is read-only (managed by cargo).
- `target/` is read-only (build artifacts).
- `InferenceBatcher` must not be modified (disjoint file constraint for parallelism).
- `ChannelEvaluator` must not be modified.
- Python packages must remain importable via `python3 -c "import hyzero"` (i.e., no changes to
  the Python package structure).
- `evaluate_batch` signature is `&mut self` — PyO3Backend can hold a `PyObject` (GIL-independent
  reference) and acquire the GIL only inside `evaluate_batch`.

## Prior Art

No existing PyO3 usage in this codebase. Reference: `pyo3` crate docs for numpy interop via
`pyo3-numpy` (or equivalently the `numpy` crate for pyo3). Standard pattern for calling Python
from Rust:

```rust
Python::with_gil(|py| {
    let module = PyModule::import(py, "hyzero.inference.server")?;
    // ...
    Ok(())
})
```

## Risks and Edge Cases

### R1 — GIL vs training contention
Training runs `train_batch()` which takes 100-500ms. Inference runs `root_setup_batch` /
`expand_leaf_batch` which takes 10-50ms. Both acquire the GIL. If the training call holds the
GIL while inference batches arrive, all game tasks stall. Mitigation: run training on a
`std::thread` (not a tokio task) so that `Python::with_gil` releases the GIL between calls
naturally (GIL releases when the `with_gil` closure exits). This is the standard PyO3 pattern.

### R2 — PyO3 exceptions propagating to async runtime
If a Python call raises an exception, `Python::with_gil` returns `Err(PyErr)`. Must convert to
a Rust error and log, not panic. Panic inside a tokio task causes the task to silently die.
Mitigation: `evaluate_batch` signature does not return `Result` — wrap PyO3 calls in a closure
that logs on error, replies to each request with a fallback (uniform policy, 0.0 value), and
continues.

### R3 — numpy array ownership and GIL lifetime
`pyo3-numpy` requires that numpy array references are only valid while the GIL is held. Strategy:
copy out of Python into Rust `Vec<f32>` before releasing the GIL. This is already the natural
flow (the return values from `root_setup_batch` are `.cpu().numpy()` copies).

### R4 — Batch array stacking
For a batch of B requests, we need to stack B individual `Vec<f32>` into a single numpy array.
Safest approach: assemble a `Vec<Vec<f32>>` in Rust, then pass a single contiguous `Vec<f32>` to
a helper that creates a 2D numpy array. No allocator tricks needed; B is small (≤64).

### R5 — Python interpreter initialization
PyO3 with `auto-initialize` feature starts the Python interpreter when the first GIL is acquired.
Must ensure the hyzero Python package is importable: either installed in the venv (`pip install -e .`)
or `PYTHONPATH` includes `python/`. The selfplay binary must document this requirement.

### R6 — Action encoding for expand_leaf batch
Each `ExpandLeaf` request carries a single `ActionIndex`. The batch packs B actions into numpy
`[B, 3, 8, 8]`. Rust must call `encode_action_spatial` per request and stack. This is already in
`src/data/encoding.rs` — no new encoding logic needed.

### R7 — KStepSample layout for training batch
`KStepSample` from the replay buffer needs to be checked to confirm the field names match what
was described. If `visit_distribution` length < 4096 (sparse representation), it must be zero-padded
to 4096 before passing to Python.

## Wiki Pages That Informed This Plan

- `docs/wiki/rust-python-integration.md` — data flow, GIL strategy, weight sync pattern
- `docs/wiki/mcts-selfplay.md` — batching parameters, replay buffer details
- `docs/wiki/neural-networks.md` — exact network shapes, inference API

## Stale / Contradictory Wiki Content

- `docs/wiki/rust-python-integration.md` lists "TODO: build.rs and pyproject.toml" under step 1.
  A `build.rs` is NOT needed for pyo3 >= 0.20 without embedding. The `pyproject.toml` is already
  present at `python/pyproject.toml`. This item should be removed from the wiki TODO.
- The wiki says "add pyo3 (feature: full)" but pyo3 `full` feature does not exist. The correct
  feature set is `["auto-initialize"]` plus the `numpy` crate (separate `pyo3-numpy` or via the
  `numpy` pyo3 helper crate). This should be corrected in the wiki.
