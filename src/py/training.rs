use numpy::{IntoPyArray, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use tokio::sync::{mpsc, watch};

use crate::data::{
    encode_action_spatial, GameTrajectory, ReplayBuffer, TrainingSample, NUM_ACTIONS,
};

/// Flat arrays assembled from a batch of `TrainingSample` for Python training.
///
/// Array shapes (all float32):
///   observations:    [B, 19, 8, 8]  stored flat as B * 1216
///   actions:         [B, K, 3, 8, 8] stored flat as B * K * 192
///   target_policies: [B, K+1, 4096] stored flat as B * (K+1) * 4096
///   target_values:   [B, K+1]       stored flat as B * (K+1)
///   target_rewards:  [B, K+1]       stored flat as B * (K+1)
pub struct BatchArrays {
    pub observations: Vec<f32>,
    pub actions: Vec<f32>,
    pub target_policies: Vec<f32>,
    pub target_values: Vec<f32>,
    pub target_rewards: Vec<f32>,
    pub batch_size: usize,
    pub unroll_k: usize,
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

    let obs_stride = 19 * 64; // 1216
    let act_stride = 3 * 64;  // 192
    let pol_stride = NUM_ACTIONS; // 4096

    let mut observations = vec![0.0f32; b * obs_stride];
    let mut actions = vec![0.0f32; b * unroll_k * act_stride];
    let mut target_policies = vec![0.0f32; b * kp1 * pol_stride];
    let mut target_values = vec![0.0f32; b * kp1];
    let mut target_rewards = vec![0.0f32; b * kp1];

    for (bi, sample) in samples.iter().enumerate() {
        let steps = &sample.steps;
        debug_assert!(
            steps.len() > unroll_k,
            "TrainingSample has {} steps but needs at least {} for unroll_k={}",
            steps.len(),
            unroll_k + 1,
            unroll_k
        );

        // observations[bi] = steps[0].observation.planes
        let obs_base = bi * obs_stride;
        observations[obs_base..obs_base + obs_stride]
            .copy_from_slice(&steps[0].observation.planes[..obs_stride]);

        // actions[bi, k] = encode_action_spatial(steps[k+1].action) for k in 0..K
        for k in 0..unroll_k {
            let action_idx = steps[k + 1].action;
            let encoded = encode_action_spatial(action_idx);
            let act_base = (bi * unroll_k + k) * act_stride;
            actions[act_base..act_base + act_stride].copy_from_slice(&encoded);
        }

        // target_policies, target_values, target_rewards for k in 0..=K
        for k in 0..kp1 {
            let step = &steps[k];

            // Zero-pad visit_distribution to NUM_ACTIONS (4096)
            let pol_base = (bi * kp1 + k) * pol_stride;
            let dist_len = step.visit_distribution.len().min(pol_stride);
            target_policies[pol_base..pol_base + dist_len]
                .copy_from_slice(&step.visit_distribution[..dist_len]);
            // remaining entries stay 0.0 from initialization

            target_values[bi * kp1 + k] = step.root_value;
            target_rewards[bi * kp1 + k] = step.reward;
        }
    }

    BatchArrays {
        observations,
        actions,
        target_policies,
        target_values,
        target_rewards,
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
}

/// Call `trainer.train_batch(batch_dict)` through the GIL and return all loss components.
///
/// Converts flat Rust `Vec<f32>` arrays into shaped numpy arrays, builds the
/// Python dict, calls `train_batch`, and extracts all four loss values.
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
    let obs_np = obs_arr.reshape([b, 19, 8, 8])?;

    let act_arr = arrays.actions.into_pyarray(py);
    let actions_np = act_arr.reshape([b, k, 3, 8, 8])?;

    let pol_arr = arrays.target_policies.into_pyarray(py);
    let policies_np = pol_arr.reshape([b, kp1, NUM_ACTIONS])?;

    let val_arr = arrays.target_values.into_pyarray(py);
    let values_np = val_arr.reshape([b, kp1])?;

    let rew_arr = arrays.target_rewards.into_pyarray(py);
    let rewards_np = rew_arr.reshape([b, kp1])?;

    // Build batch dict
    let batch_dict = PyDict::new(py);
    batch_dict.set_item("observations", obs_np)?;
    batch_dict.set_item("actions", actions_np)?;
    batch_dict.set_item("target_policies", policies_np)?;
    batch_dict.set_item("target_values", values_np)?;
    batch_dict.set_item("target_rewards", rewards_np)?;

    // Call train_batch
    let result_dict = trainer.call_method1(py, "train_batch", (batch_dict,))?;
    let bound = result_dict.bind(py);

    Ok(TrainResult {
        total_loss: bound.get_item("total_loss")?.extract()?,
        policy_loss: bound.get_item("policy_loss")?.extract()?,
        value_loss: bound.get_item("value_loss")?.extract()?,
        reward_loss: bound.get_item("reward_loss")?.extract()?,
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
        }
    }

    /// Construct a `PyTrainingThread` using `DEFAULT_CONFIG` and a fresh Python `Trainer`.
    ///
    /// Imports `hyzero.config.DEFAULT_CONFIG` and `hyzero.training.trainer.Trainer`,
    /// then calls `Trainer(config, device)`.
    pub fn from_default_config(
        device: &str,
        trajectory_rx: mpsc::Receiver<GameTrajectory>,
        version_tx: watch::Sender<u64>,
        weight_tx: watch::Sender<Option<Vec<u8>>>,
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

        Ok(Self::new(
            trainer,
            trajectory_rx,
            version_tx,
            weight_tx,
            10_000, // max_replay_trajectories
            256,    // train_batch_size
            5,      // unroll_k
            200,    // min_samples
            4,      // train_steps_per_game
        ))
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
                                "[py_training] step {}: total={:.4} policy={:.4} value={:.4} reward={:.4} (v{})",
                                self.total_train_steps,
                                result.total_loss,
                                result.policy_loss,
                                result.value_loss,
                                result.reward_loss,
                                self.model_version,
                            );
                        }
                        Err(e) => {
                            eprintln!("[py_training] train_batch error: {e}");
                            break;
                        }
                    }

                    // Checkpoint every 50 training steps
                    if self.total_train_steps.is_multiple_of(50) {
                        let _ = std::fs::create_dir_all("checkpoints");
                        let path = format!("checkpoints/model_v{}.pt", self.model_version);
                        let ckpt_result = Python::attach(|py| -> PyResult<()> {
                            let metrics = pyo3::types::PyDict::new(py);
                            self.trainer
                                .call_method1(py, "save_checkpoint", (&path, metrics))?;
                            Ok(())
                        });
                        match ckpt_result {
                            Ok(()) => println!("[py_training] Checkpoint saved: {path}"),
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
            b * 19 * 64,
            "observations length should be B * 19 * 64 = {}",
            b * 19 * 64
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
            "target_policies length should be B * (K+1) * 4096 = {}",
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
    fn test_batch_assembly_pads_short_policies() {
        let short_dist_len = 10usize;
        let k = 2usize;
        let kp1 = k + 1;

        // Create a sample where visit_distribution has only 10 entries (not 4096)
        let sample = TrainingSample {
            steps: (0..kp1)
                .map(|_| make_step_with_dist(vec![0.1f32; short_dist_len]))
                .collect(),
            game_outcome: 1.0,
        };

        let arrays = assemble_batch_arrays(&[sample], k);

        // Total policy entries: B=1 * (K+1) * 4096
        assert_eq!(arrays.target_policies.len(), kp1 * NUM_ACTIONS);

        // First `short_dist_len` entries of step 0's policy should be 0.1
        for i in 0..short_dist_len {
            assert!(
                (arrays.target_policies[i] - 0.1).abs() < 1e-6,
                "entry {i} should be 0.1, got {}",
                arrays.target_policies[i]
            );
        }

        // Entries beyond `short_dist_len` up to NUM_ACTIONS should be zero-padded
        for i in short_dist_len..NUM_ACTIONS {
            assert_eq!(
                arrays.target_policies[i], 0.0,
                "entry {i} should be 0.0 (zero-padded), got {}",
                arrays.target_policies[i]
            );
        }
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
