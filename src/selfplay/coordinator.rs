use std::sync::Arc;

use tokio::sync::{mpsc, watch, Semaphore};

use crate::PrecomputedItems;
use crate::data::GameTrajectory;
use crate::mcts::evaluator::Evaluator;
use crate::selfplay::game_task::{GameConfig, play_game};

/// Configuration for the self-play coordinator.
#[derive(Debug, Clone)]
pub struct SelfPlayConfig {
    pub max_concurrent_games: usize,
    pub game_config: GameConfig,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        Self {
            max_concurrent_games: 4,
            game_config: GameConfig::default(),
        }
    }
}

/// Orchestrates continuous self-play: spawns game tasks up to concurrency limit,
/// sends completed trajectories to the training thread.
pub struct SelfPlayCoordinator {
    precomputed: Arc<PrecomputedItems>,
    evaluator: Arc<dyn Evaluator>,
    trajectory_tx: mpsc::Sender<GameTrajectory>,
    model_version: watch::Receiver<u64>,
    config: SelfPlayConfig,
}

impl SelfPlayCoordinator {
    pub fn new(
        precomputed: Arc<PrecomputedItems>,
        evaluator: Arc<dyn Evaluator>,
        trajectory_tx: mpsc::Sender<GameTrajectory>,
        model_version: watch::Receiver<u64>,
        config: SelfPlayConfig,
    ) -> Self {
        Self {
            precomputed,
            evaluator,
            trajectory_tx,
            model_version,
            config,
        }
    }

    /// Run the coordinator loop. Spawns games continuously, limited by semaphore.
    /// Returns when the trajectory channel is closed (receiver dropped).
    pub async fn run(&self) {
        let semaphore = Arc::new(Semaphore::new(self.config.max_concurrent_games));

        loop {
            let permit = match semaphore.clone().acquire_owned().await {
                Ok(p) => p,
                Err(_) => return, // Semaphore closed
            };

            let precomputed = self.precomputed.clone();
            let evaluator = self.evaluator.clone();
            let trajectory_tx = self.trajectory_tx.clone();
            let model_version = *self.model_version.borrow();
            let game_config = self.config.game_config.clone();

            tokio::spawn(async move {
                let trajectory = play_game(
                    precomputed,
                    evaluator,
                    model_version,
                    game_config,
                ).await;

                // Send trajectory; if receiver is dropped, just discard
                let _ = trajectory_tx.send(trajectory).await;
                drop(permit);
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{BoardObservation, HiddenState, Policy, ActionIndex, NUM_ACTIONS};
    use crate::mcts::evaluator::Evaluator;
    use async_trait::async_trait;

    struct RandomEvaluator;

    #[async_trait]
    impl Evaluator for RandomEvaluator {
        async fn root_setup(&self, _obs: &BoardObservation) -> (HiddenState, Policy, f32) {
            let policy = vec![1.0 / NUM_ACTIONS as f32; NUM_ACTIONS];
            (HiddenState::new(64), policy, 0.0)
        }

        async fn expand_leaf(&self, _hs: &HiddenState, _action: ActionIndex) -> (HiddenState, f32, Policy, f32) {
            let policy = vec![1.0 / NUM_ACTIONS as f32; NUM_ACTIONS];
            (HiddenState::new(64), 0.0, policy, 0.0)
        }
    }

    #[tokio::test]
    async fn test_coordinator_produces_trajectories() {
        let precomputed = Arc::new(PrecomputedItems::begin_precomputing());
        let evaluator: Arc<dyn Evaluator> = Arc::new(RandomEvaluator);
        let (traj_tx, mut traj_rx) = mpsc::channel(16);
        let (_version_tx, version_rx) = watch::channel(1u64);

        let config = SelfPlayConfig {
            max_concurrent_games: 2,
            game_config: GameConfig {
                num_simulations: 2,
                exploration_constant: 1.5,
                temperature_moves: 2,
            },
        };

        let coordinator = SelfPlayCoordinator::new(
            precomputed, evaluator, traj_tx, version_rx, config,
        );

        // Run coordinator in background
        let coord_handle = tokio::spawn(async move { coordinator.run().await });

        // Collect a few trajectories
        let mut count = 0;
        while count < 2 {
            match tokio::time::timeout(
                std::time::Duration::from_secs(60),
                traj_rx.recv(),
            ).await {
                Ok(Some(traj)) => {
                    assert!(!traj.steps.is_empty());
                    count += 1;
                }
                _ => panic!("Timed out waiting for trajectory"),
            }
        }

        // Drop receiver to stop coordinator
        drop(traj_rx);
        coord_handle.abort();
    }
}
