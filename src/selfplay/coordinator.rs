use std::sync::Arc;

use tokio::sync::{mpsc, watch};

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

/// Orchestrates continuous self-play: spawns N persistent game loops, each
/// playing games independently and sending completed trajectories to the
/// training thread.
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

    /// Run N persistent game loops. Each loop plays games independently, reading
    /// the current model version at the start of each game and sending the resulting
    /// trajectory to the training thread. Returns when all loops terminate (i.e.,
    /// the trajectory channel is closed and receivers are dropped).
    pub async fn run(&self) {
        let mut handles = Vec::new();

        for _ in 0..self.config.max_concurrent_games {
            let precomputed = self.precomputed.clone();
            let evaluator = self.evaluator.clone();
            let trajectory_tx = self.trajectory_tx.clone();
            let model_version_rx = self.model_version.clone();
            let game_config = self.config.game_config.clone();

            handles.push(tokio::spawn(async move {
                loop {
                    let version = *model_version_rx.borrow();
                    let traj = play_game(
                        precomputed.clone(),
                        evaluator.clone(),
                        version,
                        game_config.clone(),
                    ).await;
                    if trajectory_tx.send(traj).await.is_err() {
                        break; // Channel closed, stop
                    }
                }
            }));
        }

        // Wait for all tasks (they run until channel closes)
        for handle in handles {
            let _ = handle.await;
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
        async fn root_setup(&self, _obs: &BoardObservation, _legal_mask: &[bool]) -> (HiddenState, Policy, f32) {
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
                replay_dir: None,
                adjudicate_at_cap: false,
                adjudication_material_margin: 5,
            },
        };

        let coordinator = SelfPlayCoordinator::new(
            precomputed, evaluator, traj_tx, version_rx, config,
        );

        // Run coordinator in background; N=2 persistent tasks each play games
        let coord_handle = tokio::spawn(async move { coordinator.run().await });

        // Collect at least 2 trajectories — one from each game loop
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

        // Dropping the receiver closes the channel; each game loop exits on next
        // send error, then the coordinator's run() returns after all handles join.
        drop(traj_rx);
        // Give tasks a moment to detect the closed channel and exit cleanly
        let _ = tokio::time::timeout(
            std::time::Duration::from_secs(120),
            coord_handle,
        ).await;
    }
}
