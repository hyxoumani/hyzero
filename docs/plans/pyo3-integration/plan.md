# Plan: PyO3 Integration

## Approach

Add PyO3 + numpy as Cargo dependencies, then implement three things in isolation: a
`PyO3Backend` that converts batched Rust inference requests into numpy arrays and calls
`InferenceServer`, a training batch assembly function that converts `TrainingSample` lists
into the numpy dict `Trainer.train_batch()` expects, and wire both into `selfplay.rs`. Each
subtask touches disjoint files, so they can be developed in parallel worktrees.

## Subtasks

### 1. Cargo dependency setup
- **Files**: `Cargo.toml`
- **Changes**:
  Add to `[dependencies]`:
  ```toml
  pyo3 = { version = "0.22", features = ["auto-initialize"] }
  numpy = "0.22"
  ```
  `pyo3 = 0.22` is the current stable release. `numpy` is the `pyo3-numpy` integration crate
  (published as `numpy` on crates.io). Versions must match each other — check `pyo3 = 0.22`
  and `numpy = 0.22` are compatible before committing.
  No `build.rs` is needed. PyO3 with `auto-initialize` locates the interpreter via
  `PYO3_PYTHON` env var or `python3` on PATH at runtime.
- **Tests**: `cargo check` must pass. No behavior test at this step.
- **Dependencies**: none

### 2. PyO3Backend — inference bridge
- **Files**: `src/py/mod.rs` (new), `src/py/inference_backend.rs` (new), `src/lib.rs` (add `pub mod py;`)
- **Changes**:

  `src/py/mod.rs`:
  ```rust
  pub mod inference_backend;
  pub use inference_backend::PyO3Backend;
  ```

  `src/py/inference_backend.rs` — implement `InferenceBackend` for `PyO3Backend`:

  ```rust
  use pyo3::prelude::*;
  use numpy::PyArray;
  use crate::selfplay::inference::{InferenceBackend, InferenceRequest};
  use crate::data::{HiddenState, Policy, NUM_ACTIONS, encode_action_spatial};

  pub struct PyO3Backend {
      server: PyObject,   // hyzero.inference.server.InferenceServer instance
  }

  impl PyO3Backend {
      pub fn new(server: PyObject) -> Self { Self { server } }
  }

  impl InferenceBackend for PyO3Backend {
      fn evaluate_batch(&mut self, requests: Vec<InferenceRequest>) { ... }
  }
  ```

  Inside `evaluate_batch`:
  1. Separate requests into two groups: `RootSetup` and `ExpandLeaf`. Preserve (reply sender,
     batch index) for each.
  2. If either group is non-empty, acquire GIL once (`Python::with_gil(|py| { ... })`).
  3. **RootSetup group**: stack observations into a contiguous `Vec<f32>` of shape `[B,19,8,8]`
     (B rows × 1216 floats each). Create numpy array via `numpy::PyArray2::from_vec(py, ...)`,
     reshape to `[B,19,8,8]`. Call `server.call_method1(py, "root_setup_batch", (obs_np,))`.
     Unpack the returned Python tuple: `(hidden_np, policies_np, values_np)`. Copy out to
     `Vec<f32>` with `.readonly().as_slice()?.to_vec()`. Distribute via `reply.send(...)`.
  4. **ExpandLeaf group**: stack hidden states `[B,64,8,8]` and action planes `[B,3,8,8]`
     similarly. Call `server.call_method1(py, "expand_leaf_batch", (hidden_np, actions_np))`.
     Unpack and distribute.
  5. On any `PyErr`: log the error (do not panic), send a uniform-policy fallback reply to each
     waiting sender in the failed batch.

  Fallback reply helper (inline, not a separate function):
  ```rust
  let uniform: Policy = vec![1.0 / NUM_ACTIONS as f32; NUM_ACTIONS];
  ```

  Key implementation detail for numpy array construction: `numpy` crate's `PyArray1::from_slice`
  then `reshape` is simplest. Alternatively, assemble a flat `Vec<f32>` and use
  `PyArray1::from_vec(py, flat).reshape([b, c, 8, 8])`. Both are correct; prefer the latter for
  clarity.

  `src/lib.rs` — append at end of module declarations:
  ```rust
  pub mod py;
  ```

- **Tests** (in `src/py/inference_backend.rs` `#[cfg(test)] mod tests`):
  Write a test that constructs a `PyO3Backend` using a real (in-process) `InferenceServer`
  with default config, fires a single `RootSetup` request through the backend, and asserts:
  - Hidden state length = 64 * 64 = 4096
  - Policy length = 4096
  - Policy sums to approximately 1.0
  - Value is in [-1, 1]
  And a single `ExpandLeaf` request with zero hidden state and action 0, same shape assertions.
  Mark tests `#[cfg(feature = "python-tests")]` or simply `#[ignore]` so they only run in an
  environment with hyzero Python package installed. Concrete annotation:
  ```rust
  #[test]
  #[ignore = "requires hyzero Python package (pip install -e python/)"]
  fn test_root_setup_batch() { ... }
  ```
- **Dependencies**: Subtask 1 must complete first (dependency on pyo3/numpy crates).

### 3. Training batch assembly + PyTrainingThread
- **Files**: `src/py/training.rs` (new), `src/py/mod.rs` (add `pub mod training;`)
- **Changes**:

  `src/py/training.rs` — a function that converts `Vec<TrainingSample>` into numpy arrays
  and calls `Trainer.train_batch()`, plus a replacement `TrainingThread` that uses it:

  ```rust
  pub fn train_batch_python(
      py: Python<'_>,
      trainer: &PyObject,
      samples: &[TrainingSample],
      unroll_k: usize,
  ) -> PyResult<f64>
  ```

  This function:
  1. Assembles batch numpy arrays from samples. For each sample:
     - `observations[b]` = `steps[0].observation.planes` (1216 floats → `[19,8,8]`)
     - `actions[b, k]` = `encode_action_spatial(steps[k+1].action)` (192 floats → `[3,8,8]`)
       for k in 0..unroll_k (K action steps, from step 1 to step K)
     - `target_policies[b, k]` = `steps[k].visit_distribution` zero-padded to 4096
       for k in 0..=unroll_k (K+1 policy targets)
     - `target_values[b, k]` = `steps[k].root_value`
     - `target_rewards[b, k]` = `steps[k].reward`
  2. Creates numpy arrays for each field, passes as a Python dict to `trainer.call_method1`.
  3. Returns `total_loss` as `f64` (extracted from the returned dict).

  **Important edge case**: `visit_distribution` in `StepRecord` may be shorter than 4096 if
  stored sparsely (tests use len-1 distributions). The assembler must zero-pad to `NUM_ACTIONS`
  before packing. Do NOT assume the vec is already length 4096.

  Also add `PyTrainingThread` — a struct that owns a `PyObject` trainer, receives trajectories
  on the existing `mpsc::Receiver<GameTrajectory>`, and replaces the stub training logic:
  ```rust
  pub struct PyTrainingThread { ... }
  impl PyTrainingThread {
      pub async fn run(&mut self) { ... }
  }
  ```
  The `run` loop is identical to the existing `TrainingThread::run` except:
  - After `sample_batch`, acquire GIL and call `train_batch_python`.
  - After training, call `trainer.call_method0(py, "get_weights")` to get `bytes`.
  - Send weights bytes over a `watch::Sender<Option<Vec<u8>>>` to a separate task that calls
    `inference_server.load_weights(bytes)`.

  Weight sync channel: `watch::Sender<Option<Vec<u8>>>` — initial value `None`. Inference
  server loader task watches this channel; when it receives `Some(bytes)`, it acquires GIL and
  calls `server.load_weights(bytes)`. This keeps weight loading off the training GIL-hold path.

- **Tests** (in `src/py/training.rs`):
  ```rust
  #[test]
  #[ignore = "requires hyzero Python package"]
  fn test_train_batch_python_returns_loss() { ... }
  ```
  Construct a minimal `TrainingSample` (6 steps, K=5), call `train_batch_python`, assert
  `total_loss` is finite and > 0.

  Separate unit test (no Python required) for the batch assembly logic:
  ```rust
  #[test]
  fn test_batch_assembly_shapes() { ... }
  ```
  Call an extracted helper `assemble_batch_arrays(samples, unroll_k) -> BatchArrays` and assert
  vec lengths: `observations` has B×1216 elements, `actions` has B×K×192, `target_policies`
  has B×(K+1)×4096, etc.

- **Dependencies**: Subtask 1 (pyo3/numpy crates). Can run in parallel with Subtask 2.

### 4. Wire PyO3Backend and PyTrainingThread into selfplay binary
- **Files**: `src/bin/selfplay.rs`, `src/selfplay/training.rs` (minor: keep stub, not deleted)
- **Changes**:

  `src/bin/selfplay.rs` — replace the `RandomBackend` construction block with:
  ```rust
  // Initialize Python interpreter (auto-initialize feature does this on first GIL acquire)
  let server_obj: PyObject = Python::with_gil(|py| {
      let module = PyModule::import(py, "hyzero.inference.server")?;
      let cls = module.getattr("InferenceServer")?;
      Ok::<_, PyErr>(cls.call0()?.into())
  }).expect("Failed to construct InferenceServer — is hyzero Python package installed?");

  let backend = Box::new(PyO3Backend::new(server_obj));
  ```

  Replace `TrainingThread::new(...)` with `PyTrainingThread::new(...)`, threading the weight
  sync watch channel between `PyTrainingThread` and a small `load_weights_task` that holds the
  `server_obj` clone and calls `server.load_weights(bytes)` on update.

  The `server_obj` is a `PyObject` (ref-counted, `Send + Sync`), so it can be cloned and sent
  across threads.

  Keep `RandomBackend` and `TrainingThread` in place — they are used by existing tests.
  `selfplay.rs` is the only place wired to use the new path.

- **Tests**: `cargo test` full suite must pass (existing tests use `RandomBackend`; `PyO3Backend`
  is only activated via the binary). Manual integration test: run `cargo run --bin selfplay`
  with hyzero Python package installed and confirm log lines include loss values.
- **Dependencies**: Subtasks 2 and 3 must complete first.

## Testing Strategy

**Unit tests (no Python required):**
- `test_batch_assembly_shapes` in `src/py/training.rs` — verifies array shape arithmetic without
  importing Python. Run as part of `cargo test`.

**Integration tests (Python required, `#[ignore]`):**
- `test_root_setup_batch` and `test_expand_leaf_batch` in `src/py/inference_backend.rs`
- `test_train_batch_python_returns_loss` in `src/py/training.rs`
- Run with: `cargo test -- --ignored` in an environment with `pip install -e python/`

**End-to-end:**
- `cargo run --bin selfplay` — observe that log lines show training loss values (not just
  "Sampled batch: X samples") and model version increments.
- Confirm `InferenceServer.root_setup_batch` is called by watching for PyTorch logs or adding
  a temporary print in the Python method.

**Existing test regression:**
- `cargo test` without `--ignored` must continue to pass. No existing test touches `src/py/`.
- The existing `TrainingThread` and `RandomBackend` tests are unaffected (files not modified
  by subtasks 2 and 3).

## Rollback

1. If PyO3 or numpy crate compilation fails (e.g., Python interpreter not found at build time):
   remove the pyo3/numpy lines from `Cargo.toml` and revert `src/lib.rs`. The existing
   `RandomBackend` path is untouched.
2. If GIL contention causes unacceptable latency: swap `PyO3Backend` back to `RandomBackend`
   in `selfplay.rs` — one line change. The `PyO3Backend` code stays for future use.
3. If training causes instability: revert `selfplay.rs` to use `TrainingThread` (stub) instead
   of `PyTrainingThread` — again a one-line swap.
4. All rollback paths are single-file, one-function changes. No database migrations, no
   protocol changes, no API breaks.

## Parallelization Notes

| Subtask | Can parallelize with |
|---------|----------------------|
| 1 (Cargo deps) | Must complete first |
| 2 (PyO3Backend) | Parallel with 3 |
| 3 (PyTrainingThread) | Parallel with 2 |
| 4 (wire selfplay.rs) | After 2 and 3 |

Subtasks 2 and 3 touch strictly disjoint files (`src/py/inference_backend.rs` vs
`src/py/training.rs`) and can be developed in separate worktrees simultaneously.
