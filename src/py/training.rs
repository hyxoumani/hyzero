use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use numpy::{IntoPyArray, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use tokio::sync::{mpsc, watch};

use crate::data::{
    encode_action_spatial_for_color, flip_action, flip_obs_planes, GameTrajectory, ReplayBuffer,
    TrainingSample, NUM_ACTIONS, NUM_OBS_PLANES,
};

/// Flat arrays assembled from a batch of `TrainingSample` for Python training.
///
/// Array shapes:
///   observations:    [B, K+1, 102, 8, 8]  stored flat as B * (K+1) * NUM_OBS_PLANES * 64 (f32)
///                    All K+1 steps are included for EfficientZero consistency loss.
///   actions:         [B, K, 3, 8, 8] stored flat as B * K * 192 (f32)
///   target_policies: [B, K+1, 4672]  stored flat as B * (K+1) * NUM_ACTIONS (f32)
///   target_values:   [B, K+1]        stored flat as B * (K+1) (f32)
///   target_rewards:  [B, K+1]        stored flat as B * (K+1) (f32)
///   legal_masks:     [B, 4672]        stored flat as B * NUM_ACTIONS (bool)
pub struct BatchArrays {
    pub observations: Vec<f32>,
    pub actions: Vec<f32>,
    pub target_policies: Vec<f32>,
    pub target_values: Vec<f32>,
    pub target_rewards: Vec<f32>,
    /// Boolean mask derived from `steps[0].legal_moves`; shape [B, NUM_ACTIONS].
    pub legal_masks: Vec<bool>,
    pub batch_size: usize,
    pub unroll_k: usize,
}

/// Return whether color augmentation (obs/action/target flipping) is disabled via
/// `HYZERO_DISABLE_COLOR_AUG`. Any non-empty value that is not "0" / "false" (case-
/// insensitive) disables the augmentation. Default: enabled (returns false).
///
/// Intended for isolation experiments: if disabling augmentation removes the
/// observed color asymmetry, the flip branch is the culprit.
fn color_aug_disabled() -> bool {
    match std::env::var("HYZERO_DISABLE_COLOR_AUG") {
        Ok(v) => {
            let s = v.trim().to_ascii_lowercase();
            !(s.is_empty() || s == "0" || s == "false" || s == "no")
        }
        Err(_) => false,
    }
}

/// Return the outcome blend coefficient β from the `HYZERO_VALUE_OUTCOME_BETA` env var.
///
/// Reads the env var once per call; callers should cache the result across samples in
/// a batch. Accepts values in [0.0, 1.0]; clamps silently if out of range.
/// Defaults to 0.1 when the variable is absent or unparseable.
fn outcome_blend_beta() -> f32 {
    std::env::var("HYZERO_VALUE_OUTCOME_BETA")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .map(|v| v.clamp(0.0, 1.0))
        .unwrap_or(0.1)
}

/// True when HYZERO_CONDITIONAL_BETA is set to any truthy value.
/// When enabled, decisive games (checkmates, is_draw==false) use β=1.0
/// while drawn games use the configured beta. See [2026-04-20 bug-hunt].
fn conditional_beta_enabled() -> bool {
    match std::env::var("HYZERO_CONDITIONAL_BETA") {
        Ok(v) => {
            let s = v.trim().to_ascii_lowercase();
            !(s.is_empty() || s == "0" || s == "false" || s == "no")
        }
        Err(_) => false,
    }
}

/// Return the reward outcome blend coefficient γ from `HYZERO_REWARD_OUTCOME_GAMMA`.
/// Default 0.0 (no blending, preserves raw step.reward for backward compat).
/// Clamped to [0.0, 1.0]. On parse failure, default 0.0 with a stderr warning.
fn reward_blend_gamma() -> f32 {
    std::env::var("HYZERO_REWARD_OUTCOME_GAMMA")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .map(|v| v.clamp(0.0, 1.0))
        .unwrap_or(0.0)
}

/// Assemble flat f32 arrays from a slice of `TrainingSample`.
///
/// Each `TrainingSample` contains `K+1` `StepRecord`s. Steps are:
///   - `steps[0]`: initial observation and root MCTS stats
///   - `steps[1..=K]`: subsequent unroll steps
///
/// The action taken AT step k (transitioning s_k → s_{k+1}) is stored in `steps[k].action`
/// — this is the convention used in `game_task::play_game`, which pushes each StepRecord
/// with `action: selected_action` BEFORE applying the move.
///
/// MuZero dynamics unroll: `g(hidden_k, actions[k])` must feed the action that TRANSITIONS
/// s_k → s_{k+1}, so `actions[bi, k] = encode_action_spatial(steps[k].action)` — NOT
/// `steps[k+1].action` (which is the action at s_{k+1}, unrelated to the s_k → s_{k+1} step).
pub fn assemble_batch_arrays(samples: &[TrainingSample], unroll_k: usize) -> BatchArrays {
    let b = samples.len();
    let kp1 = unroll_k + 1; // K+1

    let obs_stride = NUM_OBS_PLANES * 64; // 102 * 64 = 6528
    let act_stride = 3 * 64;             // 192
    let pol_stride = NUM_ACTIONS;         // 4672

    // Read β and γ once for the whole batch; env-var overhead amortized.
    let beta = outcome_blend_beta();
    let gamma = reward_blend_gamma();

    // observations now includes all K+1 steps for EfficientZero consistency loss.
    // Shape: [B, K+1, NUM_OBS_PLANES, 8, 8] stored flat as B * (K+1) * obs_stride.
    let mut observations = vec![0.0f32; b * kp1 * obs_stride];
    let mut actions = vec![0.0f32; b * unroll_k * act_stride];
    let mut target_policies = vec![0.0f32; b * kp1 * pol_stride];
    let mut target_values = vec![0.0f32; b * kp1];
    let mut target_rewards = vec![0.0f32; b * kp1];
    let mut legal_masks = vec![false; b * pol_stride];

    for (bi, sample) in samples.iter().enumerate() {
        let steps = &sample.steps;
        debug_assert!(
            steps.len() > unroll_k,
            "TrainingSample has {} steps but needs at least {} for unroll_k={}",
            steps.len(),
            unroll_k + 1,
            unroll_k
        );

        // Color augmentation: randomly flip board perspective for 50% of samples.
        // The POV flip is applied to both observation and training targets; see the
        // per-step block below for the value/outcome sign convention.
        // Gated by HYZERO_DISABLE_COLOR_AUG — set to 1 to force apply_flip=false.
        let apply_flip: bool = if color_aug_disabled() { false } else { rand::random() };

        // observations[bi, k] = steps[k].observation.planes (optionally color-flipped)
        // for k in 0..=unroll_k. All K+1 steps are stored for consistency loss.
        for (k, step_k) in steps.iter().enumerate().take(unroll_k + 1) {
            let obs_base = (bi * kp1 + k) * obs_stride;
            if apply_flip {
                let flipped = flip_obs_planes(&step_k.observation.planes[..obs_stride]);
                observations[obs_base..obs_base + obs_stride].copy_from_slice(&flipped);
            } else {
                observations[obs_base..obs_base + obs_stride]
                    .copy_from_slice(&step_k.observation.planes[..obs_stride]);
            }
        }

        // actions[bi, k] = encode_action_spatial_for_color(steps[k].action, pov) for k in 0..K
        // (action taken AT step k, transitioning s_k → s_{k+1}). See doc-comment above.
        //
        // Under color augmentation (apply_flip=true), the observation is flipped to the
        // OPPOSITE player's POV, so the action spatial encoding must also use the flipped
        // color. We pass `!step.white_to_move` when apply_flip is true so that
        // underpromotion ranks match the POV the network will see.
        for k in 0..unroll_k {
            let step = &steps[k];
            let pov_white = if apply_flip {
                !step.white_to_move
            } else {
                step.white_to_move
            };
            let encoded = encode_action_spatial_for_color(step.action, pov_white);
            let act_base = (bi * unroll_k + k) * act_stride;
            actions[act_base..act_base + act_stride].copy_from_slice(&encoded);
        }

        // Color-augmentation POV convention.
        //
        // Under apply_flip, `flip_obs_planes` mirrors the board and swaps my/opp slots,
        // so the network sees the position from the OPPOSITE player's POV. Both the
        // stored `step.root_value` (originally in step-k-side POV) and any outcome
        // derived from `game_outcome` must therefore be NEGATED to match the flipped
        // observation's POV. The ORIGINAL `steps[0].white_to_move` drives the sign —
        // not a "flipped white_to_move" — because the outcome is computed from the
        // pre-flip trajectory's POV and then negated once globally.
        //
        // Regression test: `test_value_target_sign_under_flip_matches_observation_pov`.
        let flip_sign: f32 = if apply_flip { -1.0 } else { 1.0 };
        let original_root_side_sign: f32 = if steps[0].white_to_move { 1.0 } else { -1.0 };

        // target_policies, target_values, target_rewards for k in 0..=K
        for k in 0..kp1 {
            let step = &steps[k];

            // Map visit_distribution entries to their action indices.
            // visit_distribution[i] corresponds to legal_moves[i], so write to
            // target_policies[pol_base + legal_moves[i]] rather than pol_base + i.
            // When apply_flip, flip each action index before scattering.
            let pol_base = (bi * kp1 + k) * pol_stride;
            for (slot, &prob) in step.visit_distribution.iter().enumerate() {
                if let Some(&action) = step.legal_moves.get(slot) {
                    let idx = if apply_flip {
                        flip_action(action as usize)
                    } else {
                        action as usize
                    };
                    if idx < pol_stride {
                        target_policies[pol_base + idx] = prob;
                    }
                }
            }
            // Entries for actions not in legal_moves stay 0.0 from initialization

            // At step k, side alternates each ply. Convert game_outcome (White-absolute)
            // to the perspective of whoever is to move at step k, then negate under flip
            // so the POV matches the flipped observation.
            let ply_flip: f32 = if k % 2 == 0 { 1.0 } else { -1.0 };
            let outcome_in_step_perspective =
                flip_sign * sample.game_outcome * original_root_side_sign * ply_flip;

            // step.root_value is in the ORIGINAL step-k-side POV; flip_sign negates it
            // when the observation is flipped.
            let root_value_target = flip_sign * step.root_value;
            // Conditional β: checkmate games use pure outcome (β=1.0) so the value head
            // sees the full ±1 signal; non-checkmate games keep the configured β (default
            // 0.3) to bootstrap through root_value. Rationale: with material shaping OFF
            // (the default), non-checkmate outcomes are 0.0 — a flat β=1.0 would collapse
            // every drawn game to target=0, wasting the information that root_value
            // carries. Gate with HYZERO_CONDITIONAL_BETA=1 (default false to preserve
            // historical behavior).
            let effective_beta: f32 = if conditional_beta_enabled() && !sample.is_draw {
                1.0
            } else {
                beta
            };
            target_values[bi * kp1 + k] =
                (1.0 - effective_beta) * root_value_target + effective_beta * outcome_in_step_perspective;
            // step.reward is only non-zero on the trajectory's last step; apply the same
            // POV flip so the reward head sees a consistent sign convention.
            let reward_target = flip_sign * step.reward;
            target_rewards[bi * kp1 + k] =
                (1.0 - gamma) * reward_target + gamma * outcome_in_step_perspective;
        }

        // legal_masks[bi]: derive from steps[0].legal_moves (root position only)
        // When apply_flip, flip each legal move index before marking.
        let mask_base = bi * pol_stride;
        for &action in &steps[0].legal_moves {
            let idx = if apply_flip {
                flip_action(action as usize)
            } else {
                action as usize
            };
            if idx < pol_stride {
                legal_masks[mask_base + idx] = true;
            }
        }
    }

    BatchArrays {
        observations,
        actions,
        target_policies,
        target_values,
        target_rewards,
        legal_masks,
        batch_size: b,
        unroll_k,
    }
}

/// All loss components returned by one training step.
pub struct TrainResult {
    pub total_loss: f64,
    pub policy_loss: f64,
    pub value_loss: f64,
    pub reward_loss: f64,
    pub consistency_loss: f64,
}

/// Call `trainer.train_batch(batch_dict)` through the GIL and return all loss components.
///
/// Converts flat Rust `Vec<f32>` arrays into shaped numpy arrays:
///   observations: `[B, K+1, 102, 8, 8]` — all K+1 steps for EfficientZero consistency loss
///   policies: `[B, K+1, 4672]`
/// Builds the Python dict, calls `train_batch`, and extracts all five loss values.
pub fn train_batch_python(
    py: Python<'_>,
    trainer: &Py<PyAny>,
    samples: &[TrainingSample],
    unroll_k: usize,
) -> PyResult<TrainResult> {
    let arrays = assemble_batch_arrays(samples, unroll_k);
    let b = arrays.batch_size;
    let k = arrays.unroll_k;
    let kp1 = k + 1;

    // Build shaped numpy arrays
    let obs_arr = arrays.observations.into_pyarray(py);
    let obs_np = obs_arr.reshape([b, kp1, NUM_OBS_PLANES, 8, 8])?;

    let act_arr = arrays.actions.into_pyarray(py);
    let actions_np = act_arr.reshape([b, k, 3, 8, 8])?;

    let pol_arr = arrays.target_policies.into_pyarray(py);
    let policies_np = pol_arr.reshape([b, kp1, NUM_ACTIONS])?;

    let val_arr = arrays.target_values.into_pyarray(py);
    let values_np = val_arr.reshape([b, kp1])?;

    let rew_arr = arrays.target_rewards.into_pyarray(py);
    let rewards_np = rew_arr.reshape([b, kp1])?;

    let mask_arr = arrays.legal_masks.into_pyarray(py);
    let masks_np = mask_arr.reshape([b, NUM_ACTIONS])?;

    // Build batch dict
    let batch_dict = PyDict::new(py);
    batch_dict.set_item("observations", obs_np)?;
    batch_dict.set_item("actions", actions_np)?;
    batch_dict.set_item("target_policies", policies_np)?;
    batch_dict.set_item("target_values", values_np)?;
    batch_dict.set_item("target_rewards", rewards_np)?;
    batch_dict.set_item("legal_masks", masks_np)?;

    // Call train_batch
    let result_dict = trainer.call_method1(py, "train_batch", (batch_dict,))?;
    let bound = result_dict.bind(py);

    Ok(TrainResult {
        total_loss: bound.get_item("total_loss")?.extract()?,
        policy_loss: bound.get_item("policy_loss")?.extract()?,
        value_loss: bound.get_item("value_loss")?.extract()?,
        reward_loss: bound.get_item("reward_loss")?.extract()?,
        consistency_loss: bound.get_item("consistency_loss")?.extract()?,
    })
}

/// Training thread that connects the Rust replay buffer to the Python `Trainer`.
///
/// Receives `GameTrajectory` values from game tasks, adds them to the replay
/// buffer, and — once enough data is available — calls `train_batch` through
/// PyO3, then syncs model weights via `watch` channels.
pub struct PyTrainingThread {
    trainer: Py<PyAny>,
    replay_buffer: ReplayBuffer,
    trajectory_rx: mpsc::Receiver<GameTrajectory>,
    version_tx: watch::Sender<u64>,
    weight_tx: watch::Sender<Option<Vec<u8>>>,
    model_version: u64,
    train_batch_size: usize,
    unroll_k: usize,
    min_samples: usize,
    train_steps_per_game: usize,
    total_train_steps: u64,
    checkpoint_interval_steps: usize,
    checkpoint_keep_last: usize,
    checkpoint_files: VecDeque<PathBuf>,
    /// Shared pointer to the latest **completed** checkpoint path.
    /// Written by the training thread after a successful `save_checkpoint` + fsync.
    /// Read by the eval task to know which checkpoint to promote.
    pub latest_checkpoint_path: Arc<Mutex<Option<PathBuf>>>,
}

impl PyTrainingThread {
    /// Create a new `PyTrainingThread`.
    ///
    /// # Arguments
    /// * `trainer` - Python `Trainer` object (already constructed).
    /// * `trajectory_rx` - Channel receiving completed game trajectories.
    /// * `version_tx` - Watch channel to publish the current model version.
    /// * `weight_tx` - Watch channel to publish serialized model weights.
    /// * `max_replay_trajectories` - Capacity of the ring buffer.
    /// * `train_batch_size` - Number of samples per training step.
    /// * `unroll_k` - MuZero unroll depth (K).
    /// * `min_samples` - Minimum total steps before training starts.
    /// * `train_steps_per_game` - Number of training steps to run per game received.
    /// * `checkpoint_interval_steps` - Save a checkpoint every N training steps.
    /// * `checkpoint_keep_last` - Rolling window: keep at most this many checkpoint files.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        trainer: Py<PyAny>,
        trajectory_rx: mpsc::Receiver<GameTrajectory>,
        version_tx: watch::Sender<u64>,
        weight_tx: watch::Sender<Option<Vec<u8>>>,
        max_replay_trajectories: usize,
        train_batch_size: usize,
        unroll_k: usize,
        min_samples: usize,
        train_steps_per_game: usize,
        checkpoint_interval_steps: usize,
        checkpoint_keep_last: usize,
    ) -> Self {
        Self {
            trainer,
            replay_buffer: ReplayBuffer::new(max_replay_trajectories),
            trajectory_rx,
            version_tx,
            weight_tx,
            model_version: 1,
            train_batch_size,
            unroll_k,
            min_samples,
            train_steps_per_game,
            total_train_steps: 0,
            checkpoint_interval_steps,
            checkpoint_keep_last,
            checkpoint_files: VecDeque::new(),
            latest_checkpoint_path: Arc::new(Mutex::new(None)),
        }
    }

    /// Construct a `PyTrainingThread` using `DEFAULT_CONFIG` and a fresh Python `Trainer`.
    ///
    /// Imports `hyzero.config.DEFAULT_CONFIG` and `hyzero.training.trainer.Trainer`,
    /// then calls `Trainer(config, device)`.
    ///
    /// If `resume_checkpoint` is `Some(path)`, loads that checkpoint before returning,
    /// restoring `model_version` and pushing the loaded weights into `weight_tx` so
    /// the `InferenceServer` starts with restored weights instead of random initialization.
    pub fn from_default_config(
        device: &str,
        trajectory_rx: mpsc::Receiver<GameTrajectory>,
        version_tx: watch::Sender<u64>,
        weight_tx: watch::Sender<Option<Vec<u8>>>,
        resume_checkpoint: Option<&str>,
    ) -> PyResult<Self> {
        let trainer = Python::attach(|py| {
            let config = PyModule::import(py, "hyzero.config")?
                .getattr("DEFAULT_CONFIG")?
                .into_pyobject(py)?
                .unbind();
            let cls =
                PyModule::import(py, "hyzero.training.trainer")?.getattr("Trainer")?;
            let trainer: Py<PyAny> = cls.call1((config, device))?.unbind();
            Ok::<Py<PyAny>, PyErr>(trainer)
        })?;

        let train_batch_size: usize = std::env::var("HYZERO_TRAIN_BATCH_SIZE")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n: &usize| n >= 1)
            .unwrap_or(256);

        let mut thread = Self::new(
            trainer,
            trajectory_rx,
            version_tx,
            weight_tx,
            10_000,           // max_replay_trajectories
            train_batch_size, // train_batch_size (env: HYZERO_TRAIN_BATCH_SIZE, default 256)
            5,                // unroll_k
            200,              // min_samples
            16,               // train_steps_per_game
            50,               // checkpoint_interval_steps
            5,                // checkpoint_keep_last
        );

        if let Some(path) = resume_checkpoint {
            thread.load_checkpoint(path)?;
        }

        Ok(thread)
    }

    /// Load a checkpoint from `path` via PyO3, updating `model_version` and
    /// broadcasting the restored weights so game loops use them immediately.
    pub fn load_checkpoint(&mut self, path: &str) -> PyResult<()> {
        let (model_version, weights) = Python::attach(|py| -> PyResult<(u64, Vec<u8>)> {
            // Call load_checkpoint (ignore the return value — it's just eval_metrics)
            self.trainer.call_method1(py, "load_checkpoint", (path,))?;
            // Read model_version from the trainer object attribute (Python sets self.model_version during load)
            let version: u64 = self.trainer.getattr(py, "model_version")?.extract(py)?;
            let raw = self.trainer.call_method0(py, "get_weights")?;
            let bytes: Vec<u8> = raw.bind(py).extract()?;
            Ok((version, bytes))
        })?;

        self.model_version = model_version;
        let _ = self.weight_tx.send(Some(weights));
        let _ = self.version_tx.send(self.model_version);
        println!("[py_training] Resumed from checkpoint: {path} (model v{})", self.model_version);
        Ok(())
    }

    /// Run the training loop until the trajectory channel closes.
    ///
    /// Steps:
    /// 1. Receive trajectories and add to the replay buffer.
    /// 2. When the buffer has at least `min_samples` total steps:
    ///    a. For each of `train_steps_per_game` steps, sample a batch and call `train_batch`.
    ///    b. On the last step, increment model_version, fetch weights, publish via channels.
    ///    c. Checkpoint to disk every 50 total training steps.
    pub async fn run(&mut self) {
        while let Some(trajectory) = self.trajectory_rx.recv().await {
            let num_steps = trajectory.steps.len();
            let game_outcome = trajectory.game_outcome;
            let is_draw = trajectory.is_draw;
            self.replay_buffer.add(trajectory);

            // Notify the Python trainer so it can maintain the checkmate counter.
            // Called after add() so replay_buffer.len() is updated for the log line below.
            let notify_result = Python::attach(|py| -> PyResult<()> {
                self.trainer.call_method1(
                    py,
                    "notify_trajectory",
                    (game_outcome as f64, is_draw),
                )?;
                Ok(())
            });
            if let Err(e) = notify_result {
                eprintln!("[py_training] notify_trajectory error: {e}");
            }

            println!(
                "[py_training] Game received: {} steps, buffer: {} games / {} total steps, model v{}",
                num_steps,
                self.replay_buffer.len(),
                self.replay_buffer.total_steps(),
                self.model_version,
            );

            if self.replay_buffer.total_steps() >= self.min_samples {
                for step_i in 0..self.train_steps_per_game {
                    let batch = self
                        .replay_buffer
                        .sample_batch(self.train_batch_size, self.unroll_k);

                    if batch.is_empty() {
                        break;
                    }

                    let train_result = Python::attach(|py| {
                        train_batch_python(py, &self.trainer, &batch, self.unroll_k)
                    });

                    match train_result {
                        Ok(result) => {
                            self.total_train_steps += 1;

                            // Only sync weights and bump version on the last step of this batch
                            if step_i == self.train_steps_per_game - 1 {
                                self.model_version += 1;
                                let weights_result =
                                    Python::attach(|py| -> PyResult<Vec<u8>> {
                                        let raw =
                                            self.trainer.call_method0(py, "get_weights")?;
                                        let bytes: Vec<u8> = raw.bind(py).extract()?;
                                        Ok(bytes)
                                    });
                                match weights_result {
                                    Ok(weights) => {
                                        let _ = self.weight_tx.send(Some(weights));
                                        let _ = self.version_tx.send(self.model_version);
                                    }
                                    Err(e) => {
                                        eprintln!("[py_training] get_weights error: {e}");
                                    }
                                }
                            }

                            println!(
                                "[py_training] step {}: total={:.4} policy={:.4} value={:.4} reward={:.4} consistency={:.4} (v{})",
                                self.total_train_steps,
                                result.total_loss,
                                result.policy_loss,
                                result.value_loss,
                                result.reward_loss,
                                result.consistency_loss,
                                self.model_version,
                            );
                        }
                        Err(e) => {
                            eprintln!("[py_training] train_batch error: {e}");
                            break;
                        }
                    }

                    // Checkpoint every `checkpoint_interval_steps` training steps
                    if self.checkpoint_interval_steps > 0
                        && self.total_train_steps.is_multiple_of(self.checkpoint_interval_steps as u64)
                    {
                        let _ = std::fs::create_dir_all("checkpoints");
                        let path_str = format!(
                            "checkpoints/model_v{:06}.pt",
                            self.model_version
                        );
                        let path = PathBuf::from(&path_str);
                        let ckpt_result = Python::attach(|py| -> PyResult<()> {
                            let metrics = pyo3::types::PyDict::new(py);
                            self.trainer
                                .call_method1(py, "save_checkpoint", (&path_str, metrics))?;
                            Ok(())
                        });
                        match ckpt_result {
                            Ok(()) => {
                                println!("[py_training] Checkpoint saved: {path_str}");
                                // Publish path to eval task before pruning old files.
                                if let Ok(mut guard) =
                                    self.latest_checkpoint_path.lock()
                                {
                                    *guard = Some(path.clone());
                                }
                                self.checkpoint_files.push_back(path);
                                // Prune oldest if window exceeded
                                if self.checkpoint_files.len() > self.checkpoint_keep_last {
                                    if let Some(oldest) = self.checkpoint_files.pop_front() {
                                        if let Err(e) = std::fs::remove_file(&oldest) {
                                            eprintln!(
                                                "[py_training] Failed to delete old checkpoint {}: {e}",
                                                oldest.display()
                                            );
                                        } else {
                                            println!(
                                                "[py_training] Pruned old checkpoint: {}",
                                                oldest.display()
                                            );
                                        }
                                    }
                                }
                            }
                            Err(e) => eprintln!("[py_training] Checkpoint error: {e}"),
                        }
                    }
                }
            }
        }

        println!("[py_training] Trajectory channel closed, shutting down");
    }

    /// Access the replay buffer (e.g., for testing or checkpointing).
    pub fn replay_buffer(&self) -> &ReplayBuffer {
        &self.replay_buffer
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{BoardObservation, StepRecord};

    fn make_step_with_dist(dist: Vec<f32>) -> StepRecord {
        StepRecord {
            observation: BoardObservation::default(),
            action: 42,
            visit_distribution: dist,
            root_value: 0.5,
            reward: 0.1,
            legal_moves: vec![42],
            white_to_move: true,
        }
    }

    fn make_sample(num_steps: usize) -> TrainingSample {
        TrainingSample {
            steps: (0..num_steps)
                .map(|_| make_step_with_dist(vec![1.0]))
                .collect(),
            game_outcome: 1.0,
            is_draw: false,
        }
    }

    #[test]
    fn test_batch_assembly_shapes() {
        let b = 4usize;
        let k = 3usize;
        let kp1 = k + 1;

        // Each sample needs K+1 steps
        let samples: Vec<TrainingSample> = (0..b).map(|_| make_sample(kp1)).collect();
        let arrays = assemble_batch_arrays(&samples, k);

        assert_eq!(
            arrays.observations.len(),
            b * kp1 * NUM_OBS_PLANES * 64,
            "observations length should be B * (K+1) * NUM_OBS_PLANES * 64 = {}",
            b * kp1 * NUM_OBS_PLANES * 64
        );
        assert_eq!(
            arrays.actions.len(),
            b * k * 3 * 64,
            "actions length should be B * K * 3 * 64 = {}",
            b * k * 3 * 64
        );
        assert_eq!(
            arrays.target_policies.len(),
            b * kp1 * NUM_ACTIONS,
            "target_policies length should be B * (K+1) * NUM_ACTIONS = {}",
            b * kp1 * NUM_ACTIONS
        );
        assert_eq!(
            arrays.target_values.len(),
            b * kp1,
            "target_values length should be B * (K+1) = {}",
            b * kp1
        );
        assert_eq!(
            arrays.target_rewards.len(),
            b * kp1,
            "target_rewards length should be B * (K+1) = {}",
            b * kp1
        );
        assert_eq!(arrays.batch_size, b);
        assert_eq!(arrays.unroll_k, k);
    }


    #[test]
    fn test_dynamics_action_uses_step_k_not_step_kplus1() {
        // REGRESSION TEST (2026-04-18): off-by-one bug in assemble_batch_arrays previously
        // stored steps[k+1].action at actions[bi, k], feeding the dynamics network the
        // action AT s_{k+1} instead of the action that TRANSITIONS s_k → s_{k+1}.
        //
        // StepRecord convention (see src/selfplay/game_task.rs: push is BEFORE apply_move):
        //   steps[t].observation = obs(s_t)
        //   steps[t].action      = a_t (action taken AT s_t, producing s_{t+1})
        //
        // MuZero unroll:
        //   hidden_0 = h(obs_0), and for k in 0..K we compute
        //   (hidden_{k+1}, reward) = g(hidden_k, actions[k])
        // So actions[k] must be a_k = steps[k].action.

        use crate::data::encoding::encode_action_spatial;

        // Two steps, with *different* source squares to make the off-by-one visible:
        //   steps[0].action: from_sq=0 (rank 0, file 0), to_sq=1  (action index 0*64+1 = 1)
        //   steps[1].action: from_sq=7 (rank 0, file 7), to_sq=15 (action index 7*64+15 = 463)
        let action_at_s0: u16 = 0 * 64 + 1;
        let action_at_s1: u16 = 7 * 64 + 15;

        let step_at_s0 = StepRecord {
            observation: BoardObservation::default(),
            action: action_at_s0,
            visit_distribution: vec![1.0],
            root_value: 0.0,
            reward: 0.0,
            legal_moves: vec![action_at_s0],
            white_to_move: true,
        };
        let step_at_s1 = StepRecord {
            observation: BoardObservation::default(),
            action: action_at_s1,
            visit_distribution: vec![1.0],
            root_value: 0.0,
            reward: 0.0,
            legal_moves: vec![action_at_s1],
            white_to_move: false, // opposite colour — still insensitive to apply_flip
        };

        // unroll_k = 1, so we have one dynamics step (actions[0]) and two StepRecords.
        let k = 1usize;
        let sample = TrainingSample {
            steps: vec![step_at_s0, step_at_s1],
            game_outcome: 0.0,
            is_draw: false,
        };

        // Run many trials to cover both apply_flip paths (50/50 random) — the invariant
        // (actions[0] corresponds to steps[0].action, possibly flipped) must hold in both.
        use crate::data::encoding::flip_action_planes;
        let encoded_a0 = encode_action_spatial(action_at_s0);
        let encoded_a0_flipped = flip_action_planes(&encoded_a0);
        let encoded_a1 = encode_action_spatial(action_at_s1);
        let encoded_a1_flipped = flip_action_planes(&encoded_a1);

        let mut hits_a0 = 0usize;
        let mut hits_a1 = 0usize;
        let trials = 100usize;
        for _ in 0..trials {
            let arrays = assemble_batch_arrays(&[sample.clone()], k);
            let act_stride = 3 * 64;
            let act0 = &arrays.actions[..act_stride];

            let matches_a0 = act0 == encoded_a0.as_slice() || act0 == encoded_a0_flipped.as_slice();
            let matches_a1 = act0 == encoded_a1.as_slice() || act0 == encoded_a1_flipped.as_slice();

            if matches_a0 {
                hits_a0 += 1;
            }
            if matches_a1 {
                hits_a1 += 1;
            }
        }

        assert_eq!(
            hits_a0, trials,
            "actions[0] must always encode steps[0].action (a_0), possibly flipped. Hits a_0={}, a_1={}",
            hits_a0, hits_a1
        );
        assert_eq!(
            hits_a1, 0,
            "actions[0] must NEVER encode steps[1].action (a_1). Hits a_1={} indicates off-by-one bug.",
            hits_a1
        );
    }

    #[test]
    fn test_batch_assembly_maps_visit_dist_to_action_indices() {
        use crate::data::encoding::flip_action;

        let k = 2usize;
        let kp1 = k + 1;

        // Step with 3 legal moves at action indices 10, 42, 100
        // and visit distribution [0.2, 0.5, 0.3]
        let legal_moves = vec![10u16, 42u16, 100u16];
        let visit_dist = vec![0.2f32, 0.5f32, 0.3f32];

        let step = StepRecord {
            observation: BoardObservation::default(),
            action: 42,
            visit_distribution: visit_dist.clone(),
            root_value: 0.5,
            reward: 0.1,
            legal_moves: legal_moves.clone(),
            white_to_move: true,
        };

        let sample = TrainingSample {
            steps: (0..kp1).map(|_| step.clone()).collect(),
            game_outcome: 1.0,
            is_draw: false,
        };

        let arrays = assemble_batch_arrays(&[sample], k);

        // Total policy entries: B=1 * (K+1) * NUM_ACTIONS
        assert_eq!(arrays.target_policies.len(), kp1 * NUM_ACTIONS);

        // Color augmentation may or may not have been applied (random).
        // Determine which case by checking where 0.5 probability (action 42) landed.
        // Under no-flip: idx=42. Under flip: idx=flip_action(42).
        let flipped_42 = flip_action(42);
        let did_flip = (arrays.target_policies[42] - 0.5).abs() > 1e-6;

        if did_flip {
            // Flipped path: each action index was mirrored
            let idx_10 = flip_action(10);
            let idx_42 = flipped_42;
            let idx_100 = flip_action(100);
            assert!((arrays.target_policies[idx_10] - 0.2).abs() < 1e-6,
                "flipped: action flip(10)={idx_10} should be 0.2");
            assert!((arrays.target_policies[idx_42] - 0.5).abs() < 1e-6,
                "flipped: action flip(42)={idx_42} should be 0.5");
            assert!((arrays.target_policies[idx_100] - 0.3).abs() < 1e-6,
                "flipped: action flip(100)={idx_100} should be 0.3");
            // Legal masks at flipped positions
            assert!(arrays.legal_masks[idx_10], "legal_masks[flip(10)] should be true");
            assert!(arrays.legal_masks[idx_42], "legal_masks[flip(42)] should be true");
            assert!(arrays.legal_masks[idx_100], "legal_masks[flip(100)] should be true");
        } else {
            // Non-flipped path: original indices
            assert!((arrays.target_policies[10] - 0.2).abs() < 1e-6, "action 10 should be 0.2");
            assert!((arrays.target_policies[42] - 0.5).abs() < 1e-6, "action 42 should be 0.5");
            assert!((arrays.target_policies[100] - 0.3).abs() < 1e-6, "action 100 should be 0.3");
            // Legal masks at original positions
            assert!(arrays.legal_masks[10], "legal_masks[10] should be true");
            assert!(arrays.legal_masks[42], "legal_masks[42] should be true");
            assert!(arrays.legal_masks[100], "legal_masks[100] should be true");
            assert!(!arrays.legal_masks[0], "legal_masks[0] should be false");
            assert!(!arrays.legal_masks[50], "legal_masks[50] should be false");
        }

        // In both cases: exactly 3 non-zero entries in policy for step 0, and total mass ~ 1.0
        let step0_policy = &arrays.target_policies[..NUM_ACTIONS];
        let nonzero_count = step0_policy.iter().filter(|&&v| v > 0.0).count();
        assert_eq!(nonzero_count, 3, "exactly 3 non-zero policy entries expected");
        let total_mass: f32 = step0_policy.iter().sum();
        assert!((total_mass - 1.0).abs() < 1e-5, "policy mass should sum to 1.0, got {total_mass}");
    }

    /// Simulate the checkpoint window pruning logic without needing Python.
    ///
    /// Creates empty files in a temp directory, then drives the same
    /// `checkpoint_files` / `checkpoint_keep_last` logic used in `run()` directly,
    /// and verifies that after 6 saves with keep_last=5 the oldest file is deleted
    /// and only the 5 most-recent files remain on disk.
    #[test]
    fn test_checkpoint_window_prunes_oldest() {
        let dir = tempfile::tempdir().expect("failed to create tempdir");
        let base = dir.path();

        let keep_last: usize = 5;
        let mut files: VecDeque<PathBuf> = VecDeque::new();

        // Simulate 6 checkpoint saves
        for i in 1usize..=6 {
            let path = base.join(format!("model_v{:06}.pt", i));
            // "Save" by creating an empty file
            std::fs::write(&path, b"").expect("write failed");

            files.push_back(path.clone());

            if files.len() > keep_last {
                if let Some(oldest) = files.pop_front() {
                    std::fs::remove_file(&oldest).ok();
                }
            }
        }

        // After 6 saves with keep_last=5 the window should contain saves 2..=6
        assert_eq!(files.len(), 5, "expected 5 files in window");

        // File for version 1 must have been deleted
        assert!(
            !base.join("model_v000001.pt").exists(),
            "oldest checkpoint should have been pruned"
        );

        // Files 2..=6 must still exist
        for i in 2usize..=6 {
            assert!(
                base.join(format!("model_v{:06}.pt", i)).exists(),
                "checkpoint {} should still exist",
                i
            );
        }
    }

    #[test]
    #[ignore = "requires hyzero Python package"]
    fn test_resume_checkpoint_restores_model_version() {
        // Save a checkpoint with a known model_version, then construct a new
        // PyTrainingThread with resume_checkpoint and verify the version is restored.
        use tokio::sync::{mpsc, watch};

        let dir = tempfile::tempdir().expect("failed to create tempdir");
        let ckpt_path = dir.path().join("model_v000042.pt").to_string_lossy().to_string();

        // Create an initial trainer, call train_batch 42 times to advance model_version to 42,
        // then save a checkpoint. Production code does NOT insert model_version into metrics —
        // save_checkpoint writes self.model_version from the trainer object directly.
        let trainer_py = Python::attach(|py| -> PyResult<Py<PyAny>> {
            let config = PyModule::import(py, "hyzero.config")?
                .getattr("DEFAULT_CONFIG")?
                .into_pyobject(py)?
                .unbind();
            let cls = PyModule::import(py, "hyzero.training.trainer")?.getattr("Trainer")?;
            let trainer: Py<PyAny> = cls.call1((config, "cpu"))?.unbind();

            // Set model_version=42 directly on the Python object (simulates 42 training steps)
            trainer.setattr(py, "model_version", 42u64)?;

            // save_checkpoint(path, metrics) — pass empty metrics dict (production behavior)
            let metrics = PyDict::new(py);
            trainer.call_method1(py, "save_checkpoint", (&ckpt_path, metrics))?;
            Ok(trainer)
        })
        .expect("failed to create Trainer / save checkpoint");
        drop(trainer_py);

        // Now construct a new PyTrainingThread with resume_checkpoint
        let (traj_tx, traj_rx) = mpsc::channel::<crate::data::GameTrajectory>(8);
        let (version_tx, version_rx) = watch::channel(1u64);
        let (weight_tx, _weight_rx) = watch::channel(None::<Vec<u8>>);

        let thread = PyTrainingThread::from_default_config(
            "cpu",
            traj_rx,
            version_tx,
            weight_tx,
            Some(&ckpt_path),
        )
        .expect("from_default_config with resume failed");

        // version_rx should now reflect the restored model_version from the checkpoint
        assert_eq!(
            *version_rx.borrow(),
            thread.model_version,
            "version channel should match restored model_version"
        );
        // model_version must be > 1 (the default starting value)
        assert!(
            thread.model_version > 1,
            "model_version should be restored from checkpoint, got {}",
            thread.model_version
        );

        drop(traj_tx);
    }

    /// Build a minimal StepRecord with the given side-to-move and root value.
    ///
    /// `white_to_move` is stored directly on the record (plane 101 removed in Phase 3b).
    /// `root_value` is the MCTS Q-estimate for this step.
    fn make_step_with_side(white_to_move: bool, root_value: f32) -> StepRecord {
        StepRecord {
            observation: BoardObservation::default(),
            action: 0,
            visit_distribution: vec![1.0],
            root_value,
            reward: 0.0,
            legal_moves: vec![0],
            white_to_move,
        }
    }

    /// Build a TrainingSample where root is White to move, root_value=0.0 for all steps.
    /// game_outcome=1.0 (White wins). Used to test outcome blend in isolation.
    #[test]
    fn test_value_target_outcome_blend_white_root() {
        // Ensure env var is at the default 0.1 for this test.
        // Remove it in case a parent test set it to something else.
        std::env::remove_var("HYZERO_VALUE_OUTCOME_BETA");

        let k = 3usize;

        // Root is White to move; root_value=0.0 so blend term vanishes
        let root_step = make_step_with_side(true, 0.0);
        let other_step = make_step_with_side(false, 0.0); // subsequent steps alternate

        let sample = TrainingSample {
            steps: vec![
                root_step.clone(),   // k=0 White to move
                other_step.clone(),  // k=1 Black to move
                root_step.clone(),   // k=2 White to move
                other_step.clone(),  // k=3 Black to move
            ],
            game_outcome: 1.0, // White wins (checkmate)
            is_draw: false,
        };

        // β=0.1; root_value=0.0; game_outcome=1.0; root_side_sign=+1.
        // Unflipped expected targets [k=0..=3]: [+0.1, -0.1, +0.1, -0.1].
        // Under color augmentation (apply_flip), all signs flip uniformly.
        // Check invariant over many trials: shape matches either unflipped or flipped.
        let expected_unflipped = [0.1f32, -0.1, 0.1, -0.1];
        for _ in 0..64 {
            let arrays = assemble_batch_arrays(&[sample.clone()], k);
            let flip_sign = if arrays.target_values[0] > 0.0 { 1.0 } else { -1.0 };
            for (k_idx, &exp) in expected_unflipped.iter().enumerate() {
                let want = flip_sign * exp;
                let got = arrays.target_values[k_idx];
                assert!(
                    (got - want).abs() < 1e-6,
                    "k={k_idx} expected {want}, got {got} (flip_sign={flip_sign})",
                );
            }
        }
    }

    /// Same as above but with root Black to move and Black winning.
    /// From Black's perspective at k=0: outcome is positive (+0.1).
    #[test]
    fn test_value_target_outcome_blend_black_root() {
        std::env::remove_var("HYZERO_VALUE_OUTCOME_BETA");

        let k = 3usize;

        // Root is Black to move; root_value=0.0 so blend term vanishes
        let root_step = make_step_with_side(false, 0.0); // k=0 Black to move
        let other_step = make_step_with_side(true, 0.0); // k=1 White to move

        let sample = TrainingSample {
            steps: vec![
                root_step.clone(),   // k=0 Black to move
                other_step.clone(),  // k=1 White to move
                root_step.clone(),   // k=2 Black to move
                other_step.clone(),  // k=3 White to move
            ],
            game_outcome: -1.0, // Black wins (checkmate)
            is_draw: false,
        };

        // β=0.1; root_value=0.0; game_outcome=-1.0; root_side_sign=-1.
        // Unflipped expected [k=0..=3]: [+0.1, -0.1, +0.1, -0.1] (Black POV at k=0).
        // Under color augmentation all signs flip uniformly.
        let expected_unflipped = [0.1f32, -0.1, 0.1, -0.1];
        for _ in 0..64 {
            let arrays = assemble_batch_arrays(&[sample.clone()], k);
            let flip_sign = if arrays.target_values[0] > 0.0 { 1.0 } else { -1.0 };
            for (k_idx, &exp) in expected_unflipped.iter().enumerate() {
                let want = flip_sign * exp;
                let got = arrays.target_values[k_idx];
                assert!(
                    (got - want).abs() < 1e-6,
                    "k={k_idx} expected {want}, got {got} (flip_sign={flip_sign})",
                );
            }
        }
    }

    /// Confirm that the root_value signal is preserved with weight 0.9 when outcome is 0 (draw).
    /// No draw penalty is applied — draws just produce outcome_in_step_perspective = 0.
    #[test]
    fn test_value_target_outcome_blend_root_value_preserved() {
        std::env::remove_var("HYZERO_VALUE_OUTCOME_BETA");

        let k = 3usize;
        let kp1 = k + 1;

        // root_value=0.5 for all steps; game_outcome=0.0 (draw)
        let step = make_step_with_side(true, 0.5);

        let sample = TrainingSample {
            steps: (0..kp1).map(|_| step.clone()).collect(),
            game_outcome: 0.0,
            is_draw: true,
        };

        // β=0.1; root_value=0.5; game_outcome=0.0 (draw); outcome term = 0.
        // Unflipped target = 0.9 * 0.5 = 0.45 for all k.
        // With color augmentation, root_value is negated along with outcome; since
        // outcome is zero, net effect is the whole target vector flips sign uniformly.
        for _ in 0..64 {
            let arrays = assemble_batch_arrays(&[sample.clone()], k);
            let flip_sign = if arrays.target_values[0] > 0.0 { 1.0 } else { -1.0 };
            let want = flip_sign * 0.45;
            for k_idx in 0..kp1 {
                assert!(
                    (arrays.target_values[k_idx] - want).abs() < 1e-5,
                    "k={k_idx} expected {want}, got {}",
                    arrays.target_values[k_idx]
                );
            }
        }
    }

    /// Serialize tests that mutate env vars read during batch assembly.
    ///
    /// Rust tests run in parallel by default; env-var mutations are process-global.
    /// Any test that reads or writes HYZERO_REWARD_OUTCOME_GAMMA,
    /// HYZERO_VALUE_OUTCOME_BETA, or HYZERO_DISABLE_COLOR_AUG MUST hold this lock
    /// for its full duration to prevent races with other env-mutating tests.
    fn reward_gamma_env_lock() -> &'static std::sync::Mutex<()> {
        static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
        LOCK.get_or_init(|| std::sync::Mutex::new(()))
    }

    /// γ=0.0 (default, no env var set) → reward targets equal raw step.reward.
    ///
    /// Verifies backward compatibility: when HYZERO_REWARD_OUTCOME_GAMMA is absent,
    /// the blend is a no-op and step.reward passes through unchanged.
    #[test]
    fn test_reward_target_blend_default() {
        let _guard = reward_gamma_env_lock().lock().unwrap();
        std::env::remove_var("HYZERO_REWARD_OUTCOME_GAMMA");
        std::env::remove_var("HYZERO_DISABLE_COLOR_AUG");

        let k = 4usize;

        // Build steps with distinct reward values to verify identity pass-through.
        // rewards: -1, 0, 0, 0, 1 across 5 steps (k+1 = 5).
        let rewards: [f32; 5] = [-1.0, 0.0, 0.0, 0.0, 1.0];
        let steps: Vec<StepRecord> = rewards
            .iter()
            .map(|&r| {
                let mut s = make_step_with_side(true, 0.0);
                s.reward = r;
                s
            })
            .collect();

        let sample = TrainingSample {
            steps,
            game_outcome: 1.0, // White wins (checkmate), but γ=0 so outcome shouldn't matter
            is_draw: false,
        };

        // With γ=0.0, target_rewards[k] = step.reward for all k (unflipped), or
        // -step.reward under color augmentation (apply_flip negates step.reward's POV).
        // We detect the branch from the sign of target_rewards[0] vs rewards[0].
        for _ in 0..64 {
            let arrays = assemble_batch_arrays(&[sample.clone()], k);
            // First non-zero slot determines flip_sign.
            let flip_sign = {
                let mut probe = 1.0f32;
                for (i, &r) in rewards.iter().enumerate() {
                    if r.abs() > 1e-6 {
                        probe = if arrays.target_rewards[i].signum() == r.signum() { 1.0 } else { -1.0 };
                        break;
                    }
                }
                probe
            };
            for (k_idx, &expected) in rewards.iter().enumerate() {
                let want = flip_sign * expected;
                assert!(
                    (arrays.target_rewards[k_idx] - want).abs() < 1e-6,
                    "k={k_idx} expected {want}, got {} (flip_sign={flip_sign})",
                    arrays.target_rewards[k_idx]
                );
            }
        }
    }

    /// γ=0.5 → reward targets are 0.5 * outcome_in_step_perspective when step.reward=0.
    ///
    /// Uses a White-root sample with game_outcome=1.0. At each step k:
    ///   outcome_in_step_perspective = 1.0 * (+1) * ply_flip
    /// So target_reward = 0.5 * ply_flip (positive at even k, negative at odd k).
    #[test]
    fn test_reward_target_blend_with_outcome() {
        let _guard = reward_gamma_env_lock().lock().unwrap();
        std::env::remove_var("HYZERO_DISABLE_COLOR_AUG");
        // SAFETY: protected by reward_gamma_env_lock(); no concurrent env-var access.
        unsafe {
            std::env::set_var("HYZERO_REWARD_OUTCOME_GAMMA", "0.5");
        }

        let k = 3usize;

        // All steps have reward=0; root is White to move; game_outcome=1.0 (White wins).
        let root_step = make_step_with_side(true, 0.0); // k=0 White to move
        let other_step = make_step_with_side(false, 0.0); // k=1 Black to move

        let sample = TrainingSample {
            steps: vec![
                root_step.clone(),   // k=0 White to move
                other_step.clone(),  // k=1 Black to move
                root_step.clone(),   // k=2 White to move
                other_step.clone(),  // k=3 Black to move
            ],
            game_outcome: 1.0, // White wins (checkmate)
            is_draw: false,
        };

        // γ=0.5; step.reward=0.0; game_outcome=1.0; root_side_sign=+1.
        // Unflipped expected [k=0..=3]: [+0.5, -0.5, +0.5, -0.5].
        // Color augmentation negates all entries uniformly.
        let expected_unflipped = [0.5f32, -0.5, 0.5, -0.5];
        for _ in 0..64 {
            let arrays = assemble_batch_arrays(&[sample.clone()], k);
            let flip_sign = if arrays.target_rewards[0] > 0.0 { 1.0 } else { -1.0 };
            for (k_idx, &exp) in expected_unflipped.iter().enumerate() {
                let want = flip_sign * exp;
                assert!(
                    (arrays.target_rewards[k_idx] - want).abs() < 1e-6,
                    "k={k_idx} expected {want}, got {} (flip_sign={flip_sign})",
                    arrays.target_rewards[k_idx]
                );
            }
        }

        std::env::remove_var("HYZERO_REWARD_OUTCOME_GAMMA");
    }

    /// Verify that HYZERO_CONDITIONAL_BETA=1 produces target_value=1.0 for a decisive
    /// sample with is_draw=false, game_outcome=1.0, root_value=0.0 at k=0 (no flip).
    ///
    /// Without conditional β the target would be (1-0.3)*0.0 + 0.3*1.0 = 0.3.
    /// With conditional β=1.0 the target must be exactly 1.0.
    #[test]
    fn test_conditional_beta_decisive_uses_pure_outcome() {
        let _guard = reward_gamma_env_lock().lock().unwrap();
        // Disable color aug so we get deterministic values (no flip).
        // SAFETY: protected by reward_gamma_env_lock(); no concurrent env-var access.
        unsafe {
            std::env::set_var("HYZERO_CONDITIONAL_BETA", "1");
            std::env::set_var("HYZERO_DISABLE_COLOR_AUG", "1");
            std::env::set_var("HYZERO_VALUE_OUTCOME_BETA", "0.3");
        }

        let k = 1usize;

        // Decisive game: White wins (is_draw=false, game_outcome=1.0, root_value=0.0).
        let step = make_step_with_side(true, 0.0);
        let sample = TrainingSample {
            steps: vec![step.clone(), step.clone()],
            game_outcome: 1.0,
            is_draw: false,
        };

        let arrays = assemble_batch_arrays(&[sample], k);

        // k=0: outcome_in_step_perspective = 1.0 (White wins, White to move, ply 0).
        // With conditional β=1.0: target = (1-1.0)*0.0 + 1.0*1.0 = 1.0 exactly.
        assert!(
            (arrays.target_values[0] - 1.0).abs() < 1e-6,
            "expected target_value[0]=1.0 under conditional β, got {}",
            arrays.target_values[0]
        );

        // Also verify drawn games still use the configured β=0.3.
        let draw_step = make_step_with_side(true, 0.0);
        let draw_sample = TrainingSample {
            steps: vec![draw_step.clone(), draw_step.clone()],
            game_outcome: 0.0,
            is_draw: true,
        };

        let draw_arrays = assemble_batch_arrays(&[draw_sample], k);

        // Draw: outcome=0.0, root_value=0.0 → target = (1-0.3)*0.0 + 0.3*0.0 = 0.0.
        assert!(
            draw_arrays.target_values[0].abs() < 1e-6,
            "expected target_value[0]=0.0 for draw, got {}",
            draw_arrays.target_values[0]
        );

        // Clean up.
        std::env::remove_var("HYZERO_CONDITIONAL_BETA");
        std::env::remove_var("HYZERO_DISABLE_COLOR_AUG");
        std::env::remove_var("HYZERO_VALUE_OUTCOME_BETA");
    }

    #[test]
    #[ignore = "requires hyzero Python package"]
    fn test_train_batch_python_returns_loss() {
        let trainer = Python::attach(|py| -> PyResult<Py<PyAny>> {
            let config = PyModule::import(py, "hyzero.config")?
                .getattr("DEFAULT_CONFIG")?
                .into_pyobject(py)?
                .unbind();
            let cls = PyModule::import(py, "hyzero.training.trainer")?.getattr("Trainer")?;
            let trainer: Py<PyAny> = cls.call1((config, "cpu"))?.unbind();
            Ok(trainer)
        })
        .expect("failed to create Trainer");

        let k = 2usize;
        let kp1 = k + 1;
        let samples: Vec<TrainingSample> = (0..4).map(|_| make_sample(kp1)).collect();

        let result = Python::attach(|py| train_batch_python(py, &trainer, &samples, k))
            .expect("train_batch_python failed");

        assert!(
            result.total_loss.is_finite(),
            "total_loss should be finite, got {}",
            result.total_loss
        );
        assert!(
            result.total_loss > 0.0,
            "total_loss should be positive, got {}",
            result.total_loss
        );
        assert!(
            result.policy_loss.is_finite(),
            "policy_loss should be finite, got {}",
            result.policy_loss
        );
        assert!(
            result.value_loss.is_finite(),
            "value_loss should be finite, got {}",
            result.value_loss
        );
        assert!(
            result.reward_loss.is_finite(),
            "reward_loss should be finite, got {}",
            result.reward_loss
        );
    }

    /// Regression test for the terminal reward POV convention.
    ///
    /// game_outcome is white-absolute (-1.0 = Black wins). The terminal step has
    /// white_to_move=false and reward=+1.0 (Black's POV of Black winning = +1.0),
    /// matching the fixed convention in game_task.rs. With γ=0.0 and aug disabled,
    /// assemble_batch_arrays must pass through +1.0 unchanged (not invert to -1.0).
    #[test]
    fn test_terminal_reward_in_step_pov_not_white_absolute() {
        let _guard = reward_gamma_env_lock().lock().unwrap();
        std::env::remove_var("HYZERO_REWARD_OUTCOME_GAMMA");
        // SAFETY: protected by reward_gamma_env_lock(); no concurrent env-var access.
        unsafe {
            std::env::set_var("HYZERO_DISABLE_COLOR_AUG", "1");
        }

        let k = 3usize;

        // 4-step trajectory: Black delivers mate at step 3.
        // steps[3].white_to_move=false, reward=+1.0 (Black's POV of Black winning = +1.0).
        // game_outcome=-1.0 (white-absolute: Black wins).
        let steps = vec![
            StepRecord {
                observation: BoardObservation::default(),
                action: 0,
                visit_distribution: vec![1.0],
                root_value: 0.0,
                reward: 0.0,
                legal_moves: vec![0],
                white_to_move: true,  // step 0: White to move
            },
            StepRecord {
                observation: BoardObservation::default(),
                action: 0,
                visit_distribution: vec![1.0],
                root_value: 0.0,
                reward: 0.0,
                legal_moves: vec![0],
                white_to_move: false, // step 1: Black to move
            },
            StepRecord {
                observation: BoardObservation::default(),
                action: 0,
                visit_distribution: vec![1.0],
                root_value: 0.0,
                reward: 0.0,
                legal_moves: vec![0],
                white_to_move: true,  // step 2: White to move
            },
            StepRecord {
                observation: BoardObservation::default(),
                action: 0,
                visit_distribution: vec![1.0],
                root_value: 0.0,
                reward: 1.0,          // +1.0 from Black's POV (Black wins = +1.0 in step POV)
                legal_moves: vec![0],
                white_to_move: false, // step 3: Black to move (delivered mate)
            },
        ];

        let sample = TrainingSample {
            steps,
            game_outcome: -1.0, // white-absolute: Black wins
            is_draw: false,
        };

        // With γ=0.0 and aug disabled, target_rewards[k] = flip_sign * step.reward.
        // flip_sign=1.0 (aug disabled). So target_rewards[3] must equal +1.0.
        // Before the game_task.rs fix, last.reward would be stored as game_outcome=-1.0,
        // and target_rewards[3] would be -1.0 (wrong sign).
        let arrays = assemble_batch_arrays(&[sample], k);
        assert!(
            (arrays.target_rewards[3] - 1.0).abs() < 1e-6,
            "target_rewards[3] should be +1.0 (Black's POV of Black winning), got {}",
            arrays.target_rewards[3]
        );
        // Intermediate steps should have reward 0.0 (pass-through)
        assert!((arrays.target_rewards[0]).abs() < 1e-6, "step 0 reward should be 0.0");
        assert!((arrays.target_rewards[1]).abs() < 1e-6, "step 1 reward should be 0.0");
        assert!((arrays.target_rewards[2]).abs() < 1e-6, "step 2 reward should be 0.0");

        std::env::remove_var("HYZERO_DISABLE_COLOR_AUG");
    }

    /// Regression test for the apply_flip value-target POV bug.
    ///
    /// When color augmentation flips the observation, both the stored root_value
    /// and the outcome-derived target must be NEGATED so the training target
    /// matches the POV of the flipped observation. Before the fix, the outcome
    /// target was invariant under flip (the two sign-flips in effective_outcome
    /// and root_side_sign cancelled), which made 50% of samples carry wrong-sign
    /// value targets — driving value loss to collapse toward zero.
    ///
    /// We detect whether a given sample was flipped by inspecting the policy slot
    /// that got populated: we construct a root with a SINGLE legal action = 1
    /// (from_sq=0 → to_sq=1). Under no flip, slot 1 is populated. Under flip,
    /// flip_action(1) = flip_sq(0)*64 + flip_sq(1) = 56*64 + 57 = 3641 is populated.
    ///
    /// Then we verify that the value target has the correct sign for the POV
    /// implied by which slot was populated.
    #[test]
    fn test_value_target_sign_under_flip_matches_observation_pov() {
        let _guard = reward_gamma_env_lock().lock().unwrap();
        std::env::remove_var("HYZERO_VALUE_OUTCOME_BETA");
        std::env::remove_var("HYZERO_REWARD_OUTCOME_GAMMA");
        // Test requires both flip branches to fire; ensure aug is not disabled.
        std::env::remove_var("HYZERO_DISABLE_COLOR_AUG");

        // Distinct root_value so we can see sign flip in the target.
        let root_value: f32 = 0.7;
        let game_outcome: f32 = 1.0; // White wins

        // Root is White to move at step 0; step 1 (unused by this test) is Black to move.
        let mut root_step = make_step_with_side(true, root_value);
        root_step.action = 1; // from_sq=0 → to_sq=1
        root_step.legal_moves = vec![1];
        root_step.visit_distribution = vec![1.0];
        let mut next_step = make_step_with_side(false, root_value);
        next_step.action = 1;
        next_step.legal_moves = vec![1];
        next_step.visit_distribution = vec![1.0];

        // β defaults to 0.1, so target_values[0] = 0.9*root_value + 0.1*outcome_in_step_perspective.
        // For K=1 we only need 2 steps.
        let k = 1usize;

        // Policy slot indices: unflipped = 1; flipped = 3641.
        let flipped_slot = super::flip_action(1);
        assert_eq!(flipped_slot, 3641, "flip_action(1) expected 3641");

        // Run many trials to cover both flip branches.
        let trials = 200;
        let mut saw_unflipped = 0;
        let mut saw_flipped = 0;
        for _ in 0..trials {
            let sample = TrainingSample {
                steps: vec![root_step.clone(), next_step.clone()],
                game_outcome,
                is_draw: false,
            };
            let arrays = assemble_batch_arrays(&[sample], k);

            let pol_base = 0; // bi=0, k=0
            let unflipped_populated = arrays.target_policies[pol_base + 1] > 0.5;
            let flipped_populated = arrays.target_policies[pol_base + flipped_slot] > 0.5;
            assert!(
                unflipped_populated ^ flipped_populated,
                "exactly one policy slot should be populated"
            );

            // Expected value target at step 0 (White to move):
            //   unflipped: 0.9 * 0.7 + 0.1 * (+1) = 0.73
            //   flipped:   0.9 * (-0.7) + 0.1 * (-1) = -0.73  (POV reversed)
            let got = arrays.target_values[0];
            if unflipped_populated {
                saw_unflipped += 1;
                assert!(
                    (got - 0.73).abs() < 1e-5,
                    "unflipped expected 0.73, got {got}"
                );
            } else {
                saw_flipped += 1;
                assert!(
                    (got - (-0.73)).abs() < 1e-5,
                    "flipped expected -0.73, got {got}"
                );
            }

            // Step k=1 (other player's turn in unflipped POV).
            // unflipped (step 1 = Black to move): 0.9*0.7 + 0.1*(1*+1*-1) = 0.63 - 0.1 = 0.53
            //   Wait: root_value is the SAME 0.7 on step 1 (both steps built with root_value=0.7).
            //   From step 1's side POV, root_value=0.7 (stored as-is).
            //   outcome in step 1 POV: game_outcome * root_side_sign * ply_flip
            //     = +1 * +1 (root white) * -1 = -1
            //   target = 0.9*0.7 + 0.1*(-1) = 0.63 - 0.1 = 0.53
            // flipped: 0.9*(-0.7) + 0.1*(+1) = -0.53
            let got1 = arrays.target_values[1];
            if unflipped_populated {
                assert!(
                    (got1 - 0.53).abs() < 1e-5,
                    "unflipped step-1 expected 0.53, got {got1}"
                );
            } else {
                assert!(
                    (got1 - (-0.53)).abs() < 1e-5,
                    "flipped step-1 expected -0.53, got {got1}"
                );
            }
        }
        // Both branches should be exercised (≈100 each over 200 trials).
        assert!(saw_unflipped > 50, "unflipped trials too rare: {saw_unflipped}");
        assert!(saw_flipped > 50, "flipped trials too rare: {saw_flipped}");
    }

    /// Mirror-trajectory target-construction symmetry regression.
    ///
    /// Constructs two 6-step trajectories (unroll_k=5) that are POV-mirror images of
    /// each other: W-trajectory starts white-to-move, B-trajectory starts black-to-move.
    /// By the current-player POV encoding convention, mirror-equivalent positions look
    /// byte-identical: white's e2e4 from white's POV uses the same squares as black's
    /// e7e5 from black's POV (both map to from_sq=12, to_sq=28, action=796).
    ///
    /// With apply_flip=false (HYZERO_DISABLE_COLOR_AUG=1), the training pipeline must
    /// scatter visit distributions to the same action indices for both trajectories.
    /// Any divergence points to a bug in target-construction that is sensitive to the
    /// `white_to_move` flag when `apply_flip=false` (where it should be irrelevant for
    /// target_policies).
    #[test]
    #[ignore = "mirror-trajectory symmetry regression — expensive; run with --ignored"]
    fn test_mirror_trajectory_targets_are_symmetric() {
        use crate::data::encoding::encode_board;
        use crate::game::{GameBoard, Player};
        use crate::{Color, PrecomputedItems};
        use std::sync::Arc;

        let _guard = reward_gamma_env_lock().lock().unwrap();
        // Force apply_flip=false for both samples so the invariant is pure.
        std::env::set_var("HYZERO_DISABLE_COLOR_AUG", "1");
        // Reset outcome blending to defaults so value targets are comparable.
        std::env::remove_var("HYZERO_VALUE_OUTCOME_BETA");
        std::env::remove_var("HYZERO_REWARD_OUTCOME_GAMMA");

        // 5 POV-symmetric action IDs for a plausible opening (e2e4, e7e5, g1f3, b8c6, f1c4):
        //   e2e4: white from_sq=12, to_sq=28  → action 796
        //   e7e5: black from_sq=52→flip=12, to_sq=36→flip=28 → action 796
        //   g1f3: white from_sq=6, to_sq=21   → action 405
        //   b8c6: black from_sq=57→1, to_sq=42→18 → action 82
        //   f1c4: white from_sq=5, to_sq=26   → action 346
        //   f8c5: black from_sq=61→5, to_sq=34→26 → action 346
        //
        // Note: plies 1 and 3 differ in action ID: 796 vs 82. This is because for ply 1
        // (e7e5 / e2e4 mirror), both map to 796; for ply 3 (b8c6 / b1c3 mirror), both map
        // to 82. All POV-encoded.
        let action_seq: [u16; 5] = [796, 796, 405, 82, 346];

        // Visit distribution for each step: 100% on single chosen action.
        // Two extra legal moves to make the distribution non-trivial.
        // White trajectory uses absolute-board action IDs.
        // Black trajectory uses the SAME action IDs from black's POV — they are identical.
        let make_legal = |chosen: u16| -> Vec<u16> {
            // Include a few other plausible moves so the mask has multiple entries.
            let mut v = vec![chosen, (chosen + 1) % 4096, (chosen + 2) % 4096];
            v.sort_unstable();
            v.dedup();
            v
        };
        let make_dist = |legal: &[u16], chosen: u16| -> Vec<f32> {
            legal.iter().map(|&a| if a == chosen { 1.0 } else { 0.0 }).collect()
        };

        let precomputed = Arc::new(PrecomputedItems::begin_precomputing());

        // Build actual board observations at mirror positions.
        // White trajectory: board positions from white's perspective.
        // Black trajectory: mirror positions from black's perspective — observations should
        // be byte-identical to white's if the POV encoding is correct.
        //
        // We use the initial board for both (step 0) since the initial position is
        // symmetric and encode_board(board, White) == encode_board(board, Black) (proven
        // by test_encode_board_initial_position_symmetry in encoding.rs).
        let initial_board = {
            let p1 = Player::init_player(true);
            let p2 = Player::init_player(false);
            GameBoard::init_game_board(precomputed.clone(), p1, p2)
        };
        let initial_obs = {
            let encoded = encode_board(&initial_board, Color::White, &[]);
            BoardObservation { planes: encoded.planes.to_vec() }
        };
        // For the initial position, white and black observations are identical.
        // Use the same observation for all 6 steps (no actual moves applied) — the test
        // focuses on the target_policy construction path, not observation correctness.
        let obs = initial_obs;

        // Construct white trajectory: 6 steps (K+1=6 for unroll_k=5)
        // Steps 0,2,4 are white-to-move; steps 1,3 are black-to-move.
        let unroll_k = 5usize;
        let num_steps = unroll_k + 1;

        let white_steps: Vec<StepRecord> = (0..num_steps)
            .map(|k| {
                let action = action_seq[k.min(4)];
                let legal = make_legal(action);
                let dist = make_dist(&legal, action);
                StepRecord {
                    observation: obs.clone(),
                    action,
                    visit_distribution: dist,
                    root_value: 0.3,
                    reward: 0.0,
                    legal_moves: legal,
                    white_to_move: k % 2 == 0, // white moves on even plies
                }
            })
            .collect();

        // Construct black trajectory: mirror of white's trajectory.
        // Starts with black to move; same action IDs (POV-symmetric), same visit dists.
        let black_steps: Vec<StepRecord> = (0..num_steps)
            .map(|k| {
                let action = action_seq[k.min(4)];
                let legal = make_legal(action);
                let dist = make_dist(&legal, action);
                StepRecord {
                    observation: obs.clone(),
                    action,
                    visit_distribution: dist,
                    root_value: 0.3,
                    reward: 0.0,
                    legal_moves: legal,
                    white_to_move: k % 2 != 0, // black moves on even plies (mirror!)
                }
            })
            .collect();

        let white_sample = TrainingSample {
            steps: white_steps,
            game_outcome: 0.0, // draw so value targets cancel out
            is_draw: true,
        };
        let black_sample = TrainingSample {
            steps: black_steps,
            game_outcome: 0.0,
            is_draw: true,
        };

        // Run a single assembly (apply_flip=false guaranteed by env var).
        let arrays = assemble_batch_arrays(&[white_sample, black_sample], unroll_k);

        let pol_stride = NUM_ACTIONS;
        let kp1 = unroll_k + 1;

        let mut divergences: Vec<(usize, usize, f32, f32)> = Vec::new();

        for k in 0..kp1 {
            let base_w = (0 * kp1 + k) * pol_stride; // bi=0 is white
            let base_b = (1 * kp1 + k) * pol_stride; // bi=1 is black
            let pol_w = &arrays.target_policies[base_w..base_w + pol_stride];
            let pol_b = &arrays.target_policies[base_b..base_b + pol_stride];

            for a in 0..pol_stride {
                let diff = (pol_w[a] - pol_b[a]).abs();
                if diff > 1e-6 {
                    divergences.push((k, a, pol_w[a], pol_b[a]));
                }
            }
        }

        std::env::remove_var("HYZERO_DISABLE_COLOR_AUG");

        assert!(
            divergences.is_empty(),
            "mirror-trajectory target_policies diverge at {} (step, action) pairs.\
             \nFirst 10 divergences (k, a, pol_w, pol_b): {:?}",
            divergences.len(),
            &divergences[..divergences.len().min(10)],
        );
    }
}
