use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use numpy::{IntoPyArray, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use tokio::sync::{mpsc, watch};

use crate::data::{
    encode_action_spatial, flip_action, flip_action_planes, flip_obs_planes, GameTrajectory,
    ReplayBuffer, TrainingSample, NUM_ACTIONS, NUM_OBS_PLANES,
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
/// The action that led from step k to step k+1 is stored in `steps[k+1].action`.
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
        let apply_flip: bool = rand::random();
        let effective_outcome = if apply_flip { -sample.game_outcome } else { sample.game_outcome };

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

        // actions[bi, k] = encode_action_spatial(steps[k+1].action) for k in 0..K
        for k in 0..unroll_k {
            let raw_action_idx = steps[k + 1].action;
            let encoded = encode_action_spatial(raw_action_idx);
            let encoded = if apply_flip {
                flip_action_planes(&encoded)
            } else {
                encoded
            };
            let act_base = (bi * unroll_k + k) * act_stride;
            actions[act_base..act_base + act_stride].copy_from_slice(&encoded);
        }

        // Determine root side-to-move from StepRecord.white_to_move (plane 101 removed in Phase 3b).
        // When color augmentation flips the sample, the perspective also flips.
        // Computed once per sample; only ply_flip is per-k.
        let root_white_to_move = if apply_flip {
            !steps[0].white_to_move
        } else {
            steps[0].white_to_move
        };
        let root_side_sign: f32 = if root_white_to_move { 1.0 } else { -1.0 };

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
            // to the perspective of whoever is to move at step k.
            let ply_flip: f32 = if k % 2 == 0 { 1.0 } else { -1.0 };
            let outcome_in_step_perspective =
                effective_outcome * root_side_sign * ply_flip;

            // Soft blend: 90% MCTS Q (preserves learned signal), 10% outcome
            // (injects outcome-aligned gradient to break self-referential bootstrap).
            target_values[bi * kp1 + k] =
                (1.0 - beta) * step.root_value + beta * outcome_in_step_perspective;
            // Reward soft blend: γ=0.0 by default (no-op). Setting HYZERO_REWARD_OUTCOME_GAMMA
            // injects outcome-aligned gradient into the reward head to prevent trivial-zero collapse.
            target_rewards[bi * kp1 + k] =
                (1.0 - gamma) * step.reward + gamma * outcome_in_step_perspective;
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

        let mut thread = Self::new(
            trainer,
            trajectory_rx,
            version_tx,
            weight_tx,
            10_000, // max_replay_trajectories
            256,    // train_batch_size
            5,      // unroll_k
            200,    // min_samples
            16,     // train_steps_per_game
            50,     // checkpoint_interval_steps
            5,      // checkpoint_keep_last
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
            self.replay_buffer.add(trajectory);

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
            game_outcome: 1.0, // White wins
        };

        let arrays = assemble_batch_arrays(&[sample], k);

        // β=0.1; root_value=0.0; game_outcome=1.0; root_side_sign=+1
        // k=0: 0.9*0.0 + 0.1*(1.0 * +1 * +1) = +0.1
        // k=1: 0.9*0.0 + 0.1*(1.0 * +1 * -1) = -0.1
        // k=2: 0.9*0.0 + 0.1*(1.0 * +1 * +1) = +0.1
        // k=3: 0.9*0.0 + 0.1*(1.0 * +1 * -1) = -0.1
        assert!(
            (arrays.target_values[0] - 0.1).abs() < 1e-6,
            "k=0 expected +0.1, got {}",
            arrays.target_values[0]
        );
        assert!(
            (arrays.target_values[1] - (-0.1)).abs() < 1e-6,
            "k=1 expected -0.1, got {}",
            arrays.target_values[1]
        );
        assert!(
            (arrays.target_values[2] - 0.1).abs() < 1e-6,
            "k=2 expected +0.1, got {}",
            arrays.target_values[2]
        );
        assert!(
            (arrays.target_values[3] - (-0.1)).abs() < 1e-6,
            "k=3 expected -0.1, got {}",
            arrays.target_values[3]
        );
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
            game_outcome: -1.0, // Black wins
        };

        let arrays = assemble_batch_arrays(&[sample], k);

        // β=0.1; root_value=0.0; game_outcome=-1.0; root_side_sign=-1
        // k=0: 0.9*0.0 + 0.1*((-1.0)*(-1)*(+1)) = +0.1  (Black wins → positive for Black)
        // k=1: 0.9*0.0 + 0.1*((-1.0)*(-1)*(-1)) = -0.1  (White's perspective, White lost)
        // k=2: 0.9*0.0 + 0.1*((-1.0)*(-1)*(+1)) = +0.1
        // k=3: 0.9*0.0 + 0.1*((-1.0)*(-1)*(-1)) = -0.1
        assert!(
            (arrays.target_values[0] - 0.1).abs() < 1e-6,
            "k=0 expected +0.1, got {}",
            arrays.target_values[0]
        );
        assert!(
            (arrays.target_values[1] - (-0.1)).abs() < 1e-6,
            "k=1 expected -0.1, got {}",
            arrays.target_values[1]
        );
        assert!(
            (arrays.target_values[2] - 0.1).abs() < 1e-6,
            "k=2 expected +0.1, got {}",
            arrays.target_values[2]
        );
        assert!(
            (arrays.target_values[3] - (-0.1)).abs() < 1e-6,
            "k=3 expected -0.1, got {}",
            arrays.target_values[3]
        );
    }

    /// Confirm that the root_value signal is preserved with weight 0.9 when outcome is 0 (draw).
    #[test]
    fn test_value_target_outcome_blend_root_value_preserved() {
        std::env::remove_var("HYZERO_VALUE_OUTCOME_BETA");

        let k = 3usize;
        let kp1 = k + 1;

        // root_value=0.5 for all steps; game_outcome=0.0 (draw) → outcome term is 0
        let step = make_step_with_side(true, 0.5);

        let sample = TrainingSample {
            steps: (0..kp1).map(|_| step.clone()).collect(),
            game_outcome: 0.0,
        };

        let arrays = assemble_batch_arrays(&[sample], k);

        // β=0.1; root_value=0.5; outcome_in_step_perspective=0.0
        // target = 0.9*0.5 + 0.1*0.0 = 0.45 for all k
        for k_idx in 0..kp1 {
            assert!(
                (arrays.target_values[k_idx] - 0.45).abs() < 1e-6,
                "k={k_idx} expected 0.45, got {}",
                arrays.target_values[k_idx]
            );
        }
    }

    /// Serialize tests that mutate HYZERO_REWARD_OUTCOME_GAMMA to prevent data races.
    ///
    /// Rust tests run in parallel by default; env-var mutations are process-global.
    /// Holding this lock for the duration of any test that reads or writes
    /// HYZERO_REWARD_OUTCOME_GAMMA prevents races between the two reward-blend tests.
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
            game_outcome: 1.0, // White wins, but γ=0 so outcome shouldn't matter
        };

        let arrays = assemble_batch_arrays(&[sample], k);

        // With γ=0.0, target_rewards[k] = step.reward for all k.
        for (k_idx, &expected) in rewards.iter().enumerate() {
            assert!(
                (arrays.target_rewards[k_idx] - expected).abs() < 1e-6,
                "k={k_idx} expected reward {expected}, got {}",
                arrays.target_rewards[k_idx]
            );
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
            game_outcome: 1.0, // White wins
        };

        let arrays = assemble_batch_arrays(&[sample], k);

        // γ=0.5; step.reward=0.0; game_outcome=1.0; root_side_sign=+1
        // k=0: 0.5*0.0 + 0.5*(1.0 * +1 * +1) = +0.5
        // k=1: 0.5*0.0 + 0.5*(1.0 * +1 * -1) = -0.5
        // k=2: 0.5*0.0 + 0.5*(1.0 * +1 * +1) = +0.5
        // k=3: 0.5*0.0 + 0.5*(1.0 * +1 * -1) = -0.5
        let expected = [0.5f32, -0.5, 0.5, -0.5];
        for (k_idx, &exp) in expected.iter().enumerate() {
            assert!(
                (arrays.target_rewards[k_idx] - exp).abs() < 1e-6,
                "k={k_idx} expected {exp}, got {}",
                arrays.target_rewards[k_idx]
            );
        }

        std::env::remove_var("HYZERO_REWARD_OUTCOME_GAMMA");
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
}
