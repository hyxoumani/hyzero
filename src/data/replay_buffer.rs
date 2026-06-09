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
    /// Per-step n-step TD value targets, one entry per step in the K+1 slice.
    /// `Some(g)` carries the n-step TD return for that step (already in step POV and
    /// outcome-aware via its discounted tail); `None` ⇒ no TD target for that step
    /// (legacy β-blend value-target path). This struct is never bincode-serialized,
    /// so on-disk `ReplayBuffer.bin` / `.pt` artifacts are unaffected.
    pub td_targets: Vec<Option<f32>>,
}

/// Whether n-step TD value targets are computed at sample time.
///
/// Read from `HYZERO_TD` (default ON). Any value that is not "0" / "false" / "no"
/// (case-insensitive, trimmed) enables it; an empty value or absence also enables it.
/// When disabled, `sample_batch` pushes `None` for every step and the trainer falls
/// back to the legacy β-blend value target byte-for-byte.
fn td_enabled() -> bool {
    match std::env::var("HYZERO_TD") {
        Ok(v) => {
            let s = v.trim().to_ascii_lowercase();
            !(s == "0" || s == "false" || s == "no")
        }
        Err(_) => true,
    }
}

/// The n-step TD horizon `n` from `HYZERO_TD_NSTEP` (default 5).
///
/// Clamped to at least 1. On parse failure or absence, defaults to 5.
fn td_nstep() -> usize {
    std::env::var("HYZERO_TD_NSTEP")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .map(|v| v.max(1))
        .unwrap_or(5)
}

/// The TD discount factor `γ` from `HYZERO_TD_GAMMA` (default 0.997).
///
/// Clamped to [0.0, 1.0]. On parse failure or absence, defaults to 0.997.
fn td_gamma() -> f32 {
    std::env::var("HYZERO_TD_GAMMA")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .map(|v| v.clamp(0.0, 1.0))
        .unwrap_or(0.997)
}

/// Compute the n-step TD return `G_t` for the step at absolute trajectory index `t`,
/// expressed in step-t's point of view.
///
/// `G_t = Σ_{i=0}^{m-1} (-1)^i · γ^i · r_{t+i} + (-1)^m · γ^m · bootstrap`
///
/// where `m = min(n, last - t)` and `last = traj.steps.len() - 1`. The `(-1)^i`
/// factors convert each future reward into step-t's POV (sides alternate every ply),
/// matching the canonical backup recurrence `G_{k-1} = r_k − G_k` (γ=1) generalized
/// to arbitrary γ. The bootstrap term is:
///   - `root_value[t+m]` (the same-ply MCTS root Q) when the window stops before the
///     trajectory end (`t+m < last`), converted to step-t POV by `(-1)^m`; or
///   - the terminal game outcome when the window runs to the end (`t+m == last`),
///     converted from White-absolute to step-t POV via `(-1)^(last-t) · outcome_sign`,
///     where `outcome_sign = game_outcome · (+1 if steps[t].white_to_move else -1)`.
fn compute_td_target(traj: &GameTrajectory, t: usize, n: usize, gamma: f32) -> f32 {
    let last = traj.steps.len() - 1;
    let m = n.min(last - t);

    let mut g = 0.0f32;
    let mut sign = 1.0f32; // (-1)^i
    let mut discount = 1.0f32; // γ^i
    for i in 0..m {
        g += sign * discount * traj.steps[t + i].reward;
        sign = -sign;
        discount *= gamma;
    }
    // After the loop: sign == (-1)^m, discount == γ^m.

    let bootstrap = if t + m == last {
        // Window runs to the trajectory end: bootstrap on the terminal signal.
        // Convert White-absolute game_outcome into step-t POV.
        let outcome_sign = if traj.steps[t].white_to_move {
            1.0
        } else {
            -1.0
        };
        let last_minus_t_sign = if (last - t).is_multiple_of(2) {
            1.0
        } else {
            -1.0
        };
        last_minus_t_sign * traj.game_outcome * outcome_sign
    } else {
        // Bootstrap on the same-ply MCTS root Q at step t+m (already in its own POV).
        traj.steps[t + m].root_value
    };

    g + sign * discount * bootstrap
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

        // Read n-step TD config once for the whole batch; env-var overhead amortized.
        let td_on = td_enabled();
        let td_n = td_nstep();
        let td_g = td_gamma();

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

            // Compute per-step n-step TD value targets from the FULL trajectory
            // (the K+1 slice is not enough — the n-step tail can extend past it).
            // When TD is disabled, push `None` for every step so the trainer keeps
            // the legacy β-blend value target byte-for-byte.
            let td_targets: Vec<Option<f32>> = (0..unroll_k + 1)
                .map(|k| {
                    if td_on {
                        Some(compute_td_target(traj, start + k, td_n, td_g))
                    } else {
                        None
                    }
                })
                .collect();

            samples.push(TrainingSample {
                steps,
                game_outcome: traj.game_outcome,
                is_draw: traj.is_draw,
                td_targets,
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

    /// Serialize tests that mutate the TD env vars (HYZERO_TD, HYZERO_TD_NSTEP,
    /// HYZERO_TD_GAMMA). Held for the full duration to prevent races with other
    /// env-mutating tests in this module.
    fn td_env_lock() -> &'static std::sync::Mutex<()> {
        static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
        LOCK.get_or_init(|| std::sync::Mutex::new(()))
    }

    /// Build a StepRecord with explicit side-to-move, root_value and reward.
    fn make_step_full(white_to_move: bool, root_value: f32, reward: f32) -> StepRecord {
        StepRecord {
            observation: BoardObservation::default(),
            action: 0,
            visit_distribution: vec![1.0],
            root_value,
            reward,
            legal_moves: vec![0],
            white_to_move,
        }
    }

    /// n-step TD return equals the signed, discounted reward sum plus the
    /// `(-1)^m·γ^m`-converted root_value bootstrap, including the `(-1)^i` sign chain.
    ///
    /// Hand-built trajectory of 4 steps so the window for an early step bootstraps on a
    /// non-terminal root_value (not the outcome). Sides alternate W,B,W,B from step 0.
    /// rewards = [0.0, 0.2, -0.4, 0.0]; root_values = [r0, r1, r2, r3] distinct.
    /// With n=2, γ known, for t=0: m=min(2, last-0)=2 and t+m=2 != last=3, so:
    ///   G_0 = (+1)·γ^0·r_{0} + (-1)·γ^1·r_{1} + (-1)^2·γ^2·root_value[2]
    ///       = reward[0] - γ·reward[1] + γ^2·root_value[2]
    #[test]
    fn td_target_equals_signed_discounted_reward_plus_bootstrap() {
        let _guard = td_env_lock().lock().unwrap();
        // SAFETY: protected by td_env_lock(); no concurrent env-var access.
        unsafe {
            std::env::set_var("HYZERO_TD", "1");
            std::env::set_var("HYZERO_TD_NSTEP", "2");
            std::env::set_var("HYZERO_TD_GAMMA", "0.9");
        }

        // steps: side W,B,W,B; rewards/root_values chosen distinct & known.
        let rewards = [0.0f32, 0.2, -0.4, 0.0];
        let root_values = [0.1f32, 0.3, 0.7, -0.5];
        let traj = GameTrajectory {
            steps: (0..4)
                .map(|i| make_step_full(i % 2 == 0, root_values[i], rewards[i]))
                .collect(),
            game_outcome: 1.0,
            model_version: 1,
            is_draw: false,
        };

        let g = 0.9f32;
        // Expected G_0 with n=2, t=0, m=2, t+m=2 != last=3 (bootstrap = root_value[2]):
        let expected_g0 = rewards[0] - g * rewards[1] + g * g * root_values[2];

        let g0 = compute_td_target(&traj, 0, 2, g);

        std::env::remove_var("HYZERO_TD");
        std::env::remove_var("HYZERO_TD_NSTEP");
        std::env::remove_var("HYZERO_TD_GAMMA");

        assert!(
            (g0 - expected_g0).abs() < 1e-6,
            "G_0 expected {expected_g0}, got {g0}"
        );
    }

    /// When the n-step window runs to the trajectory end (`t+m == last`), the bootstrap
    /// term uses the terminal game outcome (POV-converted), NOT root_value[last].
    ///
    /// This is the boundary foot-gun. Trajectory of 3 steps, sides W,B,W. White wins
    /// (game_outcome = +1.0, white-absolute). With n large (>= last-t) the window for
    /// t=0 runs to last=2 (m=2), so:
    ///   G_0 = reward[0] - γ·reward[1] + γ^2·(outcome in step-0 POV)
    /// step 0 is White-to-move and White wins, so the outcome in step-0 POV is +1.0.
    /// The net terminal factor is γ^m·outcome_sign(0) (the two (-1)^(last-t) cancel),
    /// so root_value[2] (= a large distractor value) must NOT appear in the result.
    #[test]
    fn td_target_uses_outcome_bootstrap_at_trajectory_end() {
        let _guard = td_env_lock().lock().unwrap();
        // SAFETY: protected by td_env_lock(); no concurrent env-var access.
        unsafe {
            std::env::set_var("HYZERO_TD", "1");
        }

        let g = 0.9f32;
        // root_value[2] is a deliberate distractor: if the terminal branch is missing
        // and the code bootstraps on root_value[2] instead, the result differs.
        let traj = GameTrajectory {
            steps: vec![
                make_step_full(true, 0.1, 0.0),  // step 0: White to move
                make_step_full(false, 0.3, 0.0), // step 1: Black to move
                make_step_full(true, 0.99, 0.0), // step 2: White to move (terminal)
            ],
            game_outcome: 1.0, // White wins (white-absolute)
            model_version: 1,
            is_draw: false,
        };

        // n=5 (>= last-0=2) so the window runs to the end. m=2, t+m==last.
        // Outcome in step-0 POV = +1.0 (White to move, White wins).
        // G_0 = 0.0 - γ·0.0 + γ^2·(+1.0) = γ^2.
        let expected_g0 = g * g;
        let g0 = compute_td_target(&traj, 0, 5, g);

        // Sanity: the distractor-bootstrap value would have been γ^2·root_value[2].
        let distractor = g * g * 0.99;

        std::env::remove_var("HYZERO_TD");

        assert!(
            (g0 - expected_g0).abs() < 1e-6,
            "G_0 should bootstrap on outcome (expected {expected_g0}), got {g0}"
        );
        assert!(
            (g0 - distractor).abs() > 1e-3,
            "G_0 must NOT bootstrap on root_value[last] (distractor {distractor})"
        );
    }

    /// With TD disabled (HYZERO_TD=0), every sampled step carries `None` (legacy path).
    #[test]
    fn td_disabled_yields_all_none_td_targets() {
        let _guard = td_env_lock().lock().unwrap();
        // SAFETY: protected by td_env_lock(); no concurrent env-var access.
        unsafe {
            std::env::set_var("HYZERO_TD", "0");
        }

        let mut buf = ReplayBuffer::new(10);
        buf.add(make_trajectory(20));

        let k = 5usize;
        let samples = buf.sample_batch(8, k);

        std::env::remove_var("HYZERO_TD");

        assert!(!samples.is_empty(), "expected non-empty batch");
        for s in &samples {
            assert_eq!(
                s.td_targets.len(),
                k + 1,
                "one entry per step in the K+1 slice"
            );
            assert!(
                s.td_targets.iter().all(|t| t.is_none()),
                "TD disabled must yield all-None td_targets"
            );
        }
    }

    /// With TD enabled (default), sampled steps carry `Some(_)` TD targets.
    #[test]
    fn td_enabled_yields_some_td_targets() {
        let _guard = td_env_lock().lock().unwrap();
        // SAFETY: protected by td_env_lock(); no concurrent env-var access.
        unsafe {
            std::env::set_var("HYZERO_TD", "1");
        }

        let mut buf = ReplayBuffer::new(10);
        buf.add(make_trajectory(20));

        let k = 5usize;
        let samples = buf.sample_batch(8, k);

        std::env::remove_var("HYZERO_TD");

        assert!(!samples.is_empty(), "expected non-empty batch");
        for s in &samples {
            assert_eq!(
                s.td_targets.len(),
                k + 1,
                "one entry per step in the K+1 slice"
            );
            assert!(
                s.td_targets.iter().all(|t| t.is_some()),
                "TD enabled must yield all-Some td_targets"
            );
        }
    }
}
