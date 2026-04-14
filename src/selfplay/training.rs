use std::path::PathBuf;

use tokio::sync::{mpsc, watch};

use crate::data::{GameTrajectory, ReplayBuffer};

/// Configuration for the training thread.
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Minimum total steps in replay buffer before training starts.
    pub min_samples_before_training: usize,
    /// Batch size for training samples.
    pub train_batch_size: usize,
    /// Number of forward steps to unroll (K in MuZero).
    pub unroll_k: usize,
    /// Max trajectories in the replay buffer.
    pub max_replay_trajectories: usize,
    /// Checkpoint to disk every N trajectories received.
    pub checkpoint_interval: usize,
    /// Path for replay buffer disk checkpoints.
    pub checkpoint_path: PathBuf,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            min_samples_before_training: 500,
            train_batch_size: 256,
            unroll_k: 5,
            max_replay_trajectories: 10_000,
            checkpoint_interval: 100,
            checkpoint_path: PathBuf::from("checkpoints/replay_buffer.bin"),
        }
    }
}

/// Stub training thread. Receives trajectories, manages replay buffer,
/// periodically samples batches (logged but not sent to Python yet),
/// and publishes model version increments.
pub struct TrainingThread {
    trajectory_rx: mpsc::Receiver<GameTrajectory>,
    version_tx: watch::Sender<u64>,
    replay_buffer: ReplayBuffer,
    config: TrainingConfig,
    current_version: u64,
    trajectories_since_checkpoint: usize,
}

impl TrainingThread {
    pub fn new(
        trajectory_rx: mpsc::Receiver<GameTrajectory>,
        version_tx: watch::Sender<u64>,
        config: TrainingConfig,
    ) -> Self {
        let replay_buffer = ReplayBuffer::new(config.max_replay_trajectories);
        Self {
            trajectory_rx,
            version_tx,
            replay_buffer,
            config,
            current_version: 1,
            trajectories_since_checkpoint: 0,
        }
    }

    /// Run the training loop. Receives trajectories and manages the replay buffer.
    /// Currently a stub: samples batches and logs stats, but doesn't call Python.
    pub async fn run(&mut self) {
        while let Some(trajectory) = self.trajectory_rx.recv().await {
            let num_steps = trajectory.steps.len();
            self.replay_buffer.add(trajectory);
            self.trajectories_since_checkpoint += 1;

            println!(
                "[training] Game received: {} steps, buffer: {} games / {} total steps, model v{}",
                num_steps,
                self.replay_buffer.len(),
                self.replay_buffer.total_steps(),
                self.current_version,
            );

            // Check if we have enough data to train
            if self.replay_buffer.total_steps() >= self.config.min_samples_before_training {
                let batch = self.replay_buffer.sample_batch(
                    self.config.train_batch_size,
                    self.config.unroll_k,
                );

                if !batch.is_empty() {
                    // Stub: log batch stats instead of calling Python
                    println!(
                        "[training] Sampled batch: {} samples, K={} unroll",
                        batch.len(),
                        self.config.unroll_k,
                    );

                    // Increment model version (stub: pretend training happened)
                    self.current_version += 1;
                    let _ = self.version_tx.send(self.current_version);
                }
            }

            // Periodic disk checkpoint
            if self.trajectories_since_checkpoint >= self.config.checkpoint_interval {
                if let Some(parent) = self.config.checkpoint_path.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                match self.replay_buffer.checkpoint_to_disk(&self.config.checkpoint_path) {
                    Ok(()) => println!(
                        "[training] Checkpoint saved: {} games",
                        self.replay_buffer.len(),
                    ),
                    Err(e) => eprintln!("[training] Checkpoint failed: {}", e),
                }
                self.trajectories_since_checkpoint = 0;
            }
        }

        println!("[training] Trajectory channel closed, shutting down");
    }

    pub fn replay_buffer(&self) -> &ReplayBuffer {
        &self.replay_buffer
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{BoardObservation, StepRecord, GameTrajectory};

    fn make_trajectory(num_steps: usize) -> GameTrajectory {
        GameTrajectory {
            steps: (0..num_steps).map(|_| StepRecord {
                observation: BoardObservation::default(),
                action: 0,
                visit_distribution: vec![1.0],
                root_value: 0.0,
                reward: 0.0,
                legal_moves: vec![0],
            }).collect(),
            game_outcome: 1.0,
            model_version: 1,
        }
    }

    #[tokio::test]
    async fn test_training_receives_trajectories() {
        let (traj_tx, traj_rx) = mpsc::channel(16);
        let (version_tx, version_rx) = watch::channel(1u64);

        let config = TrainingConfig {
            min_samples_before_training: 10,
            train_batch_size: 4,
            unroll_k: 2,
            max_replay_trajectories: 100,
            checkpoint_interval: 1000, // Don't checkpoint in test
            checkpoint_path: PathBuf::from("/tmp/hyzero_test_training.bin"),
        };

        let mut training = TrainingThread::new(traj_rx, version_tx, config);

        // Send trajectories then close channel
        for _ in 0..5 {
            traj_tx.send(make_trajectory(10)).await.unwrap();
        }
        drop(traj_tx);

        // Run training to completion
        training.run().await;

        assert_eq!(training.replay_buffer().len(), 5);
        assert_eq!(training.replay_buffer().total_steps(), 50);

        // Model version should have incremented (50 steps >= min 10)
        let version = *version_rx.borrow();
        assert!(version > 1, "Model version should have incremented, got {}", version);
    }
}
