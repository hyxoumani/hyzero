use std::collections::VecDeque;
use std::fs;
use std::io;
use std::path::Path;

use super::types::{GameTrajectory, StepRecord};
use rand::Rng;

/// A sample drawn from the replay buffer for training.
/// Contains K+1 consecutive steps from a single game.
#[derive(Debug, Clone)]
pub struct TrainingSample {
    pub steps: Vec<StepRecord>,
    pub game_outcome: f32,
}

/// Ring buffer of game trajectories with random sampling for training.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ReplayBuffer {
    trajectories: VecDeque<GameTrajectory>,
    max_trajectories: usize,
    total_steps: usize,
}

impl ReplayBuffer {
    pub fn new(max_trajectories: usize) -> Self {
        Self {
            trajectories: VecDeque::new(),
            max_trajectories,
            total_steps: 0,
        }
    }

    /// Add a trajectory. Evicts the oldest if at capacity.
    pub fn add(&mut self, trajectory: GameTrajectory) {
        self.total_steps += trajectory.steps.len();
        if self.trajectories.len() >= self.max_trajectories {
            if let Some(evicted) = self.trajectories.pop_front() {
                self.total_steps -= evicted.steps.len();
            }
        }
        self.trajectories.push_back(trajectory);
    }

    /// Sample a batch of training samples. Each sample contains K+1 consecutive steps.
    /// Trajectories are weighted by length and recency for on-policy-biased step sampling.
    ///
    /// Weight formula: `(steps.len() - unroll_k) * exp(-decay * age)` where
    /// `age = current_version.saturating_sub(traj.model_version)`.
    /// When `decay == 0.0`, behavior is identical to pure length-weighted sampling.
    ///
    /// Returns empty vec if buffer is empty or no trajectory is long enough.
    pub fn sample_batch(
        &self,
        batch_size: usize,
        unroll_k: usize,
        current_version: u64,
        decay: f64,
    ) -> Vec<TrainingSample> {
        if self.trajectories.is_empty() || self.total_steps == 0 {
            return Vec::new();
        }

        let min_len = unroll_k + 1;
        // Build weighted list: (trajectory index, f64 weight = length * recency)
        let weights: Vec<(usize, f64)> = self
            .trajectories
            .iter()
            .enumerate()
            .filter_map(|(i, t)| {
                if t.steps.len() >= min_len {
                    let age = current_version.saturating_sub(t.model_version) as f64;
                    let recency = (-decay * age).exp();
                    let w = (t.steps.len() - unroll_k) as f64 * recency;
                    Some((i, w))
                } else {
                    None
                }
            })
            .collect();

        if weights.is_empty() {
            return Vec::new();
        }

        let total_weight: f64 = weights.iter().map(|(_, w)| w).sum();
        if total_weight <= 0.0 {
            // Fallback: should be impossible unless all weights underflowed to zero
            return Vec::new();
        }

        let mut rng = rand::rng();
        let mut samples = Vec::with_capacity(batch_size);

        for _ in 0..batch_size {
            // Weighted random trajectory selection via prefix-sum walk
            let pick = rng.random_range(0.0..total_weight);
            let mut accumulated = 0.0f64;
            let mut traj_idx = weights[0].0;
            for &(idx, weight) in &weights {
                accumulated += weight;
                if pick < accumulated {
                    traj_idx = idx;
                    break;
                }
            }

            let traj = &self.trajectories[traj_idx];
            let max_start = traj.steps.len() - unroll_k;
            let start = rng.random_range(0..max_start);
            let steps = traj.steps[start..start + unroll_k + 1].to_vec();

            samples.push(TrainingSample {
                steps,
                game_outcome: traj.game_outcome,
            });
        }

        samples
    }

    pub fn len(&self) -> usize {
        self.trajectories.len()
    }

    pub fn is_empty(&self) -> bool {
        self.trajectories.is_empty()
    }

    pub fn total_steps(&self) -> usize {
        self.total_steps
    }

    /// Serialize to disk using bincode.
    pub fn checkpoint_to_disk(&self, path: &Path) -> Result<(), io::Error> {
        let bytes = bincode::serialize(self).map_err(io::Error::other)?;
        fs::write(path, bytes)
    }

    /// Deserialize from disk.
    pub fn load_from_disk(path: &Path) -> Result<Self, io::Error> {
        let bytes = fs::read(path)?;
        bincode::deserialize(&bytes).map_err(io::Error::other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::types::{BoardObservation, GameTrajectory, StepRecord};

    fn make_step() -> StepRecord {
        StepRecord {
            observation: BoardObservation::default(),
            action: 0,
            visit_distribution: vec![1.0],
            root_value: 0.0,
            reward: 0.0,
            legal_moves: vec![0],
        }
    }

    fn make_trajectory(num_steps: usize) -> GameTrajectory {
        GameTrajectory {
            steps: (0..num_steps).map(|_| make_step()).collect(),
            game_outcome: 1.0,
            model_version: 1,
        }
    }

    #[test]
    fn test_add_and_eviction() {
        let mut buf = ReplayBuffer::new(3);
        buf.add(make_trajectory(5));
        buf.add(make_trajectory(10));
        buf.add(make_trajectory(8));
        assert_eq!(buf.len(), 3);
        assert_eq!(buf.total_steps(), 23);

        // Adding a 4th evicts the first (5 steps)
        buf.add(make_trajectory(3));
        assert_eq!(buf.len(), 3);
        assert_eq!(buf.total_steps(), 21); // 10 + 8 + 3
    }

    #[test]
    fn test_empty_buffer_sample() {
        let buf = ReplayBuffer::new(10);
        let samples = buf.sample_batch(5, 3, 1, 0.0);
        assert!(samples.is_empty());
    }

    #[test]
    fn test_sample_batch_size() {
        let mut buf = ReplayBuffer::new(10);
        buf.add(make_trajectory(20));
        buf.add(make_trajectory(15));

        let samples = buf.sample_batch(8, 3, 1, 0.0);
        assert_eq!(samples.len(), 8);
    }

    #[test]
    fn test_sample_has_correct_steps() {
        let mut buf = ReplayBuffer::new(10);
        buf.add(make_trajectory(20));

        let k = 5;
        let samples = buf.sample_batch(10, k, 1, 0.0);
        for sample in &samples {
            assert_eq!(sample.steps.len(), k + 1);
        }
    }

    #[test]
    fn test_trajectories_too_short_for_unroll() {
        let mut buf = ReplayBuffer::new(10);
        buf.add(make_trajectory(3)); // too short for k=5

        let samples = buf.sample_batch(5, 5, 1, 0.0);
        assert!(samples.is_empty());
    }

    #[test]
    fn test_total_steps_tracking() {
        let mut buf = ReplayBuffer::new(100);
        buf.add(make_trajectory(10));
        assert_eq!(buf.total_steps(), 10);
        buf.add(make_trajectory(20));
        assert_eq!(buf.total_steps(), 30);
    }

    /// Helper: make a trajectory with an explicit model_version.
    fn make_trajectory_versioned(num_steps: usize, model_version: u64) -> GameTrajectory {
        GameTrajectory {
            steps: (0..num_steps).map(|_| make_step()).collect(),
            game_outcome: 1.0,
            model_version,
        }
    }

    /// With decay=0.5 and a large version gap, the newer trajectory should be
    /// selected with much higher frequency than the older one.
    #[test]
    fn test_recency_biases_toward_newer() {
        // Use distinguishable game_outcome values to identify which trajectory was picked.
        let mut buf = ReplayBuffer::new(10);
        let mut old_traj = make_trajectory_versioned(20, 1);
        old_traj.game_outcome = 0.0; // old
        let mut new_traj = make_trajectory_versioned(20, 10);
        new_traj.game_outcome = 1.0; // new
        buf.add(old_traj);
        buf.add(new_traj);

        let current_version = 10u64;
        let decay = 0.5f64;
        let n = 1000usize;
        let mut newer_count = 0usize;

        for _ in 0..n {
            let samples = buf.sample_batch(1, 1, current_version, decay);
            if samples[0].game_outcome > 0.5 {
                newer_count += 1;
            }
        }
        // With age=9, decay=0.5: weight_old = exp(-4.5) ≈ 0.011, weight_new = exp(0) = 1.0
        // Expected newer fraction ≈ 1.0 / 1.011 ≈ 0.989; threshold is >90%
        assert!(
            newer_count > 900,
            "newer trajectory should be picked >90% of the time with decay=0.5, got {}/{}",
            newer_count,
            n
        );
    }

    /// With decay=0.0 all recency factors are 1.0, so selection should be
    /// uniform across equal-length trajectories (within statistical tolerance).
    #[test]
    fn test_decay_zero_is_uniform() {
        let mut buf = ReplayBuffer::new(10);
        let mut old_traj = make_trajectory_versioned(20, 1);
        old_traj.game_outcome = 0.0;
        let mut new_traj = make_trajectory_versioned(20, 10);
        new_traj.game_outcome = 1.0;
        buf.add(old_traj);
        buf.add(new_traj);

        let n = 2000usize;
        let mut newer_count = 0usize;
        for _ in 0..n {
            let samples = buf.sample_batch(1, 1, 10, 0.0);
            if samples[0].game_outcome > 0.5 {
                newer_count += 1;
            }
        }
        // Both trajectories have equal weight; each has 19 valid start positions.
        // Expected fraction ≈ 0.5; allow ±10% tolerance.
        let fraction = newer_count as f64 / n as f64;
        assert!(
            (fraction - 0.5).abs() < 0.1,
            "with decay=0.0, selection should be ~50/50, got {:.3}",
            fraction
        );
    }

    #[test]
    fn test_checkpoint_roundtrip() {
        let mut buf = ReplayBuffer::new(10);
        buf.add(make_trajectory(5));
        buf.add(make_trajectory(8));

        let dir = std::env::temp_dir().join("hyzero_test_replay");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("test_checkpoint.bin");

        buf.checkpoint_to_disk(&path).unwrap();
        let loaded = ReplayBuffer::load_from_disk(&path).unwrap();

        assert_eq!(loaded.len(), buf.len());
        assert_eq!(loaded.total_steps(), buf.total_steps());
        assert_eq!(loaded.max_trajectories, buf.max_trajectories);

        let _ = fs::remove_dir_all(&dir);
    }
}
