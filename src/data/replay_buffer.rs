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
    pub is_draw: bool,
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
    ///
    /// Applies prioritized sampling by outcome type: decisive-outcome trajectories
    /// (checkmates, `is_draw=false`) are oversampled to a configurable fraction of the
    /// batch. This forces the value head to see high-variance targets (±1 for checkmates,
    /// ~0 for draws) within every batch, preventing the "dead value head" collapse where
    /// V→constant because the training distribution is dominated by near-zero draw targets.
    ///
    /// The decisive fraction is read from `HYZERO_DECISIVE_SAMPLE_FRAC` (default 0.25,
    /// clamped to [0.0, 1.0]). When no decisive trajectories exist (early training),
    /// falls back to uniform sampling across all trajectories.
    ///
    /// Within each pool (decisive / all), trajectories are weighted by the number of
    /// valid start positions (`steps.len() - unroll_k`) for uniform per-step sampling.
    ///
    /// Returns empty vec if buffer is empty or no trajectory is long enough.
    pub fn sample_batch(&self, batch_size: usize, unroll_k: usize) -> Vec<TrainingSample> {
        if self.trajectories.is_empty() || self.total_steps == 0 {
            return Vec::new();
        }

        let min_len = unroll_k + 1;

        // Read decisive fraction from env (default 0.25).
        let decisive_frac = std::env::var("HYZERO_DECISIVE_SAMPLE_FRAC")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .map(|v| v.clamp(0.0, 1.0))
            .unwrap_or(0.25);

        // Partition eligible trajectories into decisive (checkmate) and all pools.
        let decisive_indices: Vec<usize> = self
            .trajectories
            .iter()
            .enumerate()
            .filter(|(_, t)| !t.is_draw && t.steps.len() >= min_len)
            .map(|(i, _)| i)
            .collect();
        let all_indices: Vec<usize> = self
            .trajectories
            .iter()
            .enumerate()
            .filter(|(_, t)| t.steps.len() >= min_len)
            .map(|(i, _)| i)
            .collect();

        if all_indices.is_empty() {
            return Vec::new();
        }

        let mut rng = rand::rng();
        let mut samples = Vec::with_capacity(batch_size);

        // Compute how many samples should come from the decisive pool.
        // Falls back to 0 (uniform from all) when no decisive trajectories exist.
        let target_decisive = if !decisive_indices.is_empty() {
            (batch_size as f32 * decisive_frac) as usize
        } else {
            0
        };

        for i in 0..batch_size {
            // First `target_decisive` samples come from decisive pool, rest from all.
            let pool: &[usize] = if i < target_decisive {
                &decisive_indices
            } else {
                &all_indices
            };

            // Weighted random selection by trajectory length (uniform step sampling).
            let total_weight: usize = pool
                .iter()
                .map(|&idx| self.trajectories[idx].steps.len() - unroll_k)
                .sum();
            if total_weight == 0 {
                continue;
            }
            let mut pick = rng.random_range(0..total_weight);
            let mut traj_idx = pool[0];
            for &idx in pool {
                let weight = self.trajectories[idx].steps.len() - unroll_k;
                if pick < weight {
                    traj_idx = idx;
                    break;
                }
                pick -= weight;
            }

            let traj = &self.trajectories[traj_idx];
            let max_start = traj.steps.len() - unroll_k;
            let start = rng.random_range(0..max_start);
            let steps = traj.steps[start..start + unroll_k + 1].to_vec();

            samples.push(TrainingSample {
                steps,
                game_outcome: traj.game_outcome,
                is_draw: traj.is_draw,
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
            white_to_move: true,
        }
    }

    fn make_trajectory(num_steps: usize) -> GameTrajectory {
        GameTrajectory {
            steps: (0..num_steps).map(|_| make_step()).collect(),
            game_outcome: 1.0,
            model_version: 1,
            is_draw: false,
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
        let samples = buf.sample_batch(5, 3);
        assert!(samples.is_empty());
    }

    #[test]
    fn test_sample_batch_size() {
        let mut buf = ReplayBuffer::new(10);
        buf.add(make_trajectory(20));
        buf.add(make_trajectory(15));

        let samples = buf.sample_batch(8, 3);
        assert_eq!(samples.len(), 8);
    }

    #[test]
    fn test_sample_has_correct_steps() {
        let mut buf = ReplayBuffer::new(10);
        buf.add(make_trajectory(20));

        let k = 5;
        let samples = buf.sample_batch(10, k);
        for sample in &samples {
            assert_eq!(sample.steps.len(), k + 1);
        }
    }

    #[test]
    fn test_trajectories_too_short_for_unroll() {
        let mut buf = ReplayBuffer::new(10);
        buf.add(make_trajectory(3)); // too short for k=5

        let samples = buf.sample_batch(5, 5);
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

    /// Serialize tests that mutate HYZERO_DECISIVE_SAMPLE_FRAC to prevent data races.
    /// Rust tests run in parallel by default; env-var mutations are process-global.
    fn decisive_frac_env_lock() -> &'static std::sync::Mutex<()> {
        static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
        LOCK.get_or_init(|| std::sync::Mutex::new(()))
    }

    #[test]
    fn test_priority_sampling_prefers_decisive_when_set() {
        let _guard = decisive_frac_env_lock().lock().unwrap();

        // 1 decisive trajectory, 9 drawn trajectories
        let mut buf = ReplayBuffer::new(100);
        let mut decisive_traj = make_trajectory(20);
        decisive_traj.is_draw = false;
        decisive_traj.game_outcome = 1.0;
        buf.add(decisive_traj);
        for _ in 0..9 {
            let mut draw_traj = make_trajectory(20);
            draw_traj.is_draw = true;
            draw_traj.game_outcome = 0.0;
            buf.add(draw_traj);
        }

        // SAFETY: protected by decisive_frac_env_lock(); no concurrent env-var access.
        unsafe {
            std::env::set_var("HYZERO_DECISIVE_SAMPLE_FRAC", "0.5");
        }

        let samples = buf.sample_batch(100, 5);
        let from_decisive = samples.iter().filter(|s| !s.is_draw).count();

        std::env::remove_var("HYZERO_DECISIVE_SAMPLE_FRAC");

        // With decisive_frac=0.5, expect ~50% from decisive (at least 30 — conservative).
        assert!(
            from_decisive >= 30,
            "Expected >=30 samples from decisive trajectory with frac=0.5, got {from_decisive}"
        );
    }

    #[test]
    fn test_priority_sampling_falls_back_when_no_decisive() {
        let _guard = decisive_frac_env_lock().lock().unwrap();

        // All trajectories are draws
        let mut buf = ReplayBuffer::new(100);
        for _ in 0..5 {
            let mut draw_traj = make_trajectory(20);
            draw_traj.is_draw = true;
            buf.add(draw_traj);
        }

        // SAFETY: protected by decisive_frac_env_lock(); no concurrent env-var access.
        unsafe {
            std::env::set_var("HYZERO_DECISIVE_SAMPLE_FRAC", "0.5");
        }
        let samples = buf.sample_batch(50, 5);
        std::env::remove_var("HYZERO_DECISIVE_SAMPLE_FRAC");

        // Should still return 50 samples despite no decisive games (all from draws).
        assert_eq!(samples.len(), 50);
        let from_draws = samples.iter().filter(|s| s.is_draw).count();
        assert_eq!(
            from_draws, 50,
            "All samples should be from drawn trajectories"
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
