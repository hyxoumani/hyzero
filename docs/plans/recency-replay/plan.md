# Plan: Recency-Weighted Replay Sampling

## Approach

Replace the current length-only weighted sampling in `ReplayBuffer::sample_batch` with
recency-weighted sampling: multiply each trajectory's length-weight by an exponential
decay factor `exp(-lambda * (current_version - traj.model_version))`. This biases
gradient steps toward recent, on-policy games while retaining the diversity of older
games. No search changes, no Python changes, no schema or shape breaks — the
`model_version` tag already exists on every `GameTrajectory`.

The decay constant `lambda` is passed as a new field on `PyTrainingThread` (default
`0.1`), making it configurable via `HYZERO_REPLAY_DECAY` env var without touching
`src/selfplay/game_task.rs` or any Python code. The replay buffer itself receives the
current `model_version` as a parameter at sample time so it remains stateless.

---

## 1. Context Summary

### Current state

`ReplayBuffer::sample_batch` (line 48, `src/data/replay_buffer.rs`) weights each
trajectory by `steps.len() - unroll_k` (number of valid start positions). This gives
equal per-step probability regardless of when the trajectory was generated.

After 70 games in the 900s baseline, approximately the first 10-15 games come from a
random or near-random model (version 1-8). Those early trajectories contribute ~15-20%
of the buffer's steps but carry almost zero signal quality. Training on them in equal
proportion dilutes each batch.

`GameTrajectory.model_version: u64` is already set at trajectory creation in
`src/selfplay/game_task.rs` (pulled from the watch channel). No schema change needed.

### Key file:line references

| Location | Current state | Change |
|---|---|---|
| `src/data/replay_buffer.rs:48-98` | `sample_batch(batch_size, unroll_k)` — length weights only | Add `model_version: u64, decay: f64` params; multiply weight by recency factor |
| `src/data/replay_buffer.rs:55-63` | weight = `steps.len() - unroll_k` | weight = `(steps.len() - unroll_k) * exp(-decay * (current_version - traj.model_version))` |
| `src/py/training.rs:283-295` | `from_default_config` — hardcoded `unroll_k=5` etc | Add `replay_decay: f64 = 0.1`, pass through to `PyTrainingThread::new` |
| `src/py/training.rs:187-230` | `PyTrainingThread` struct fields | Add `replay_decay: f64` field |
| `src/py/training.rs:330-380` | `run()` training loop — calls `replay_buffer.sample_batch(...)` | Pass `self.model_version, self.replay_decay` |
| `src/bin/selfplay.rs:54-87` | Env-var parse block | Add `HYZERO_REPLAY_DECAY` parsing into `RunConfig` |
| `src/bin/selfplay.rs:34-47` | `RunConfig::default()` | Add `replay_decay: f64 = 0.1` |
| `src/bin/selfplay.rs:136-139` | `PyTrainingThread::from_default_config(...)` call | Thread `config.replay_decay` through |

### Why lambda=0.1 as default

At lambda=0.1 with a 50-version window, the weight ratio between the newest game and
a game 20 versions old is exp(-0.1 * 20) ≈ 0.135. Games more than 50 versions old
decay to < 1% of the newest game's weight — effectively zero, but still present as
diversity. This is a conservative, bounded decay that won't exclude trajectories
entirely (avoids replay starvation) while still strongly preferring recent data.

Steeper values (0.2-0.5) can be tried in follow-up experiments.

---

## Subtasks

### 1. Update `ReplayBuffer::sample_batch` signature and weight logic

- **Files**: `src/data/replay_buffer.rs`
- **Changes**:
  - Change signature to `sample_batch(&self, batch_size: usize, unroll_k: usize, current_version: u64, decay: f64) -> Vec<TrainingSample>`
  - Replace the weight computation at lines 55-63:
    ```rust
    let age = current_version.saturating_sub(t.model_version) as f64;
    let recency = (-decay * age).exp();
    // Multiply length-weight by recency factor; convert to integer weight via scaling
    // Use floating-point weights: collect Vec<(usize, f64)> instead of Vec<(usize, usize)>
    Some((i, (t.steps.len() - unroll_k) as f64 * recency))
    ```
  - Update the weighted random selection loop to use `f64` weights (sample a uniform
    float in `[0, total_weight)` and walk the prefix sum)
  - When `decay == 0.0`, behavior is identical to current (all recency factors = 1.0)
- **Tests**: Update all 6 existing `sample_batch` call sites in `#[cfg(test)]` to pass
  `current_version=1, decay=0.0` (preserves existing semantics). Add 2 new tests:
  - `test_recency_biases_toward_newer`: buffer with 2 trajectories, versions 1 and 10;
    sample 1000 times with decay=0.5; verify trajectory 10 is picked with >90% frequency
  - `test_decay_zero_is_uniform`: buffer with versions 1 and 10; decay=0.0; verify
    selection is uniform (within tolerance)
- **Dependencies**: none

### 2. Thread `model_version` and `replay_decay` through `PyTrainingThread`

- **Files**: `src/py/training.rs`
- **Changes**:
  - Add `replay_decay: f64` field to `PyTrainingThread` struct (alongside `unroll_k` etc.)
  - Update `PyTrainingThread::new(...)` to accept `replay_decay: f64` parameter
  - Update `from_default_config` to pass `replay_decay: f64 = 0.1` (hardcoded default for
    now; wired to env var in subtask 3)
  - In `run()`, update the `sample_batch` call to:
    `self.replay_buffer.sample_batch(self.train_batch_size, self.unroll_k, self.model_version, self.replay_decay)`
- **Tests**: No new unit tests (integration tested via subtask 1 replay buffer tests and
  the e2e baseline run)
- **Dependencies**: Subtask 1 must complete first (signature change)

### 3. Add `HYZERO_REPLAY_DECAY` env var in selfplay binary

- **Files**: `src/bin/selfplay.rs`
- **Changes**:
  - Add `replay_decay: f64` to `RunConfig` struct
  - Set default `replay_decay: 0.1` in `RunConfig::default()`
  - Parse `HYZERO_REPLAY_DECAY` env var in the config block (same pattern as other vars)
  - Pass `config.replay_decay` into `PyTrainingThread::from_default_config` (update
    the `from_default_config` call at line 138 to include the new param)
- **Tests**: No new tests (env var parsing follows existing validated pattern)
- **Dependencies**: Subtask 2 must complete first

---

## Testing Strategy

1. **Unit tests** (`cargo test`): All 6 existing replay buffer tests must still pass
   (backward compat via `decay=0.0`). The 2 new recency tests confirm the bias works.
2. **Compile check** (`cargo check`): Verify no type errors from `f64` weight refactor
   and new function signatures.
3. **Clippy** (`cargo clippy`): Catch any `.exp()` float precision issues or range
   warnings.
4. **Baseline run** (`bash scripts/run_baseline.sh 900`): Compare against 5.7646.

---

## Expected Score Delta

The mechanism: each of the 552 training steps in the baseline samples from a pool that
includes early random-model games. With recency weighting (lambda=0.1), those games
get ~5-13x less weight by the end of the run. Effective batch quality improves, which
should accelerate policy loss descent (lower `final_policy_loss`) without touching
`decisive_ratio` or `avg_game_length` directly (search is unchanged).

Conservative estimate: `final_policy_loss` drops by 0.1-0.2 more in 900s, contributing
+0.1 to +0.2 to the score. Secondary: marginally better policy may shorten games
slightly (-0.5 on avg_game_length / 100 = +0.005). Expected delta: **+0.1 to +0.25**.

Stretch: if early random games are strongly suppressing value head learning, decisive
ratio could improve by 0.05 → +0.5 on score.

---

## Why This Is Safer Than Tree Reuse

Tree reuse required architectural understanding of the MuZero latent space
(hidden state grounding, Q-value injection into PUCT). It failed because the latent
children produced by `g(s, a)` during early training were too noisy to warm-start PUCT
selections. The entire benefit relied on latent-space invariants that don't hold yet.

This experiment touches only the replay buffer's sampling distribution. The training
loop, MCTS, inference pipeline, and Python model are all unchanged. The worst-case
outcome is the score stays flat (if the recency signal is weak at 70 games). There is
no structural risk analogous to the latent-space grounding issue, and the feature can
be disabled entirely by setting `HYZERO_REPLAY_DECAY=0.0`.

---

## Fallback

If recency weighting shows no improvement or regression at 900s:
- Try a higher decay (`HYZERO_REPLAY_DECAY=0.3`) since 70 games may be too few for
  lambda=0.1 to matter much (the maximum version gap at baseline end is ~552 / 8 ≈ 69
  model versions, so a trajectory from model v1 has weight exp(-0.1 * 69) ≈ 0.001 —
  already near-zero; 0.1 may already be the right value)
- Next candidate: increase `HYZERO_SIMS` from 50 → 80 (partial doubling to balance
  search quality vs. throughput)
- Next candidate: increase `hidden_channels` from 64 → 96 in `python/hyzero/config.py`
  (larger model, more capacity to absorb the richer 103-plane input)
