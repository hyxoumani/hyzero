# Rust-Python Integration (PyO3)

The Rust core (game logic, MCTS, self-play) calls the PyTorch network layer
in-process over PyO3 (`pyo3 = 0.28`, `numpy = 0.28`, `auto-initialize`). See
[Board Encoding](board-encoding.md) for plane definitions and
[Neural Networks](neural-networks.md) for full network shapes. Code lives in
`src/py/{mod,inference_backend,training}.rs`; the Python side is the `hyzero`
package under `python/hyzero/`.

## Data Flow

```
N game tasks → InferenceRequest → InferenceBatcher
                                     ├─ collect up to max_batch_size (or batch_timeout_ms)
                                     └─ single PyO3 call (one GIL acquisition)
                                           ↓
                                 Python InferenceServer
                                     ├─ root_setup_batch([B,102,8,8], legal_masks[B,4672]|None)
                                     │     → hidden[B,128,8,8], policy[B,4672], value[B]
                                     └─ expand_leaf_batch(hidden[B,128,8,8], actions[B,3,8,8])
                                           → new_hidden, rewards[B], policy[B,4672], value[B]
                                           ↓
                                 Batcher distributes via oneshot channels
```

All arrays are `float32` numpy. Policies are post-softmax (root_setup masks
illegal logits to `-inf` first when `legal_masks` is provided). Values are
tanh-bounded `[-1, 1]`.

## Inference Backend (`src/py/inference_backend.rs`)

`PyO3Backend` implements the `InferenceBackend` trait (`src/selfplay/inference.rs`).
It holds a `Py<PyAny>` handle to a Python `InferenceServer` plus `hidden_channels`.
Per batch it:
1. Acquires the GIL (`Python::attach`).
2. Converts `InferenceRequest`s (RootSetup / ExpandLeaf) into numpy arrays
   (observations, legal masks, or hidden states + action planes).
3. Calls `root_setup_batch` / `expand_leaf_batch` on the server.
4. Reshapes outputs back into `HiddenState` / `Policy` / value and replies on each
   request's oneshot channel.

`from_config` reads `hidden_channels` out of `hyzero.config.DEFAULT_CONFIG`.

## Training Bridge (`src/py/training.rs`)

`PyTrainingThread` owns a `Py<PyAny>` handle to the Python `Trainer` plus the
in-memory `ReplayBuffer`. It:
- Receives `GameTrajectory`s, buffers them, and once `min_samples` are present
  runs `train_steps_per_game` steps per game.
- Assembles batch numpy arrays in Rust (`assemble_batch_arrays`: observations,
  actions, target policies/values/rewards, legal masks, with color augmentation),
  then calls `trainer.train_batch(batch)`.
- After each batch increments `model_version`, calls `trainer.get_weights()`
  (bytes), and publishes them through `watch::Sender<Option<Vec<u8>>>`.
- Periodically calls `trainer.save_checkpoint(path)` (`model_v{:06}.pt`).

See [Self-Play Coordinator](selfplay-coordinator.md) and
[Neural Networks](neural-networks.md) for the training-step details.

## Weight Sync & Servers

The selfplay binary (`src/bin/selfplay.rs`) constructs **three** Python
`InferenceServer`s, each behind its own batcher:
- **challenger / self-play** — receives fresh weights whenever the trainer
  publishes a new version (a weight-loader task calls `load_weights(bytes)` on it).
- **champion** — frozen weights of the current champion (`best.pt`), hot-swapped on
  promotion via `SwappableBackend`.
- **opponent** — reloaded from `best_v{NNN}.pt` once per pool member per Elo cycle
  (the `EvaluationTask` calls `load_weights` directly on its held `Py<PyAny>`).

`trainer.get_weights()` and `inference_server.load_weights(bytes)` are the
contract: a `torch.save` dict of `{h, g, f}` state dicts, deserialized with
`weights_only=False`.

## GIL Strategy

One GIL acquisition per **batch** (~32 requests), not per MCTS node. Same-process
via the `InferenceBackend` trait. If the GIL ever becomes the bottleneck, the
`PyO3Backend` could be swapped for a subprocess backend without touching the rest
of the Rust pipeline.

## Gotchas

1. **`action_to_move` signature**: `(action, board, color)` — needs the board and
   active color to reconstruct castling/en-passant moves. Selfplay/inference
   callers already pass them.
2. **Visit distribution is sparse**: `StepRecord.visit_distribution` has length =
   number of legal moves; the batch assembler scatters it into a dense
   `[B, K+1, 4672]` target by action index (not by slot).
3. **`Py<PyAny>` sharing**: clone the handle with `clone_ref(py)` (not `clone()`)
   when sharing a server between the backend and the weight-loader task.
4. **Three servers, not one**: challenger, champion, and opponent are independent
   `InferenceServer` instances on separate batchers so they can run concurrently.
5. **`weights_only=False`** is intentional on every `torch.load` (payloads carry
   dicts alongside tensors).

## Related Files

- `src/py/inference_backend.rs` — `PyO3Backend`
- `src/py/training.rs` — `PyTrainingThread`, batch assembly
- `src/selfplay/inference.rs` — `InferenceBatcher`, `InferenceBackend`, `RandomBackend`, `SwappableBackend`
- `python/hyzero/inference/server.py` — `InferenceServer`
- `python/hyzero/training/trainer.py` — `Trainer`
