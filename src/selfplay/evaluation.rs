use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::watch;

use crate::PrecomputedItems;
use crate::data::{BoardObservation, HiddenState, Policy, ActionIndex, NUM_ACTIONS};
use crate::mcts::evaluator::Evaluator;
use crate::selfplay::game_task::{GameConfig, play_game};

/// Evaluator that returns uniform policy and zero value — a pure random baseline.
pub struct RandomEvaluator;

#[async_trait]
impl Evaluator for RandomEvaluator {
    async fn root_setup(&self, _obs: &BoardObservation, _legal_mask: &[bool]) -> (HiddenState, Policy, f32) {
        let policy = vec![1.0 / NUM_ACTIONS as f32; NUM_ACTIONS];
        (HiddenState::new(64), policy, 0.0)
    }

    async fn expand_leaf(
        &self,
        _hs: &HiddenState,
        _action: ActionIndex,
    ) -> (HiddenState, f32, Policy, f32) {
        let policy = vec![1.0 / NUM_ACTIONS as f32; NUM_ACTIONS];
        (HiddenState::new(64), 0.0, policy, 0.0)
    }
}

/// Configuration for the periodic evaluation task.
#[derive(Debug, Clone)]
pub struct EvaluationConfig {
    /// Training steps (model versions) between evaluation runs.
    pub eval_interval_steps: u64,
    /// Number of games to play per evaluation run.
    pub eval_games: usize,
    /// MCTS simulations per move during evaluation.
    pub num_simulations: u32,
    /// Moves before switching to greedy (temperature → 0).
    pub temperature_moves: u32,
}

impl Default for EvaluationConfig {
    fn default() -> Self {
        Self {
            eval_interval_steps: 200,
            eval_games: 10,
            num_simulations: 50,
            temperature_moves: 15,
        }
    }
}

/// Periodically evaluates the model by playing games against itself and logging
/// learning-signal metrics (win rate, game length, decisive game ratio).
///
/// Triggered by model version updates via a `watch::Receiver<u64>`. When the
/// model version increases by at least `eval_interval_steps` since the last
/// evaluation, a batch of `eval_games` games is played and metrics are logged.
pub struct EvaluationTask {
    precomputed: Arc<PrecomputedItems>,
    model_evaluator: Arc<dyn Evaluator>,
    model_version_rx: watch::Receiver<u64>,
    config: EvaluationConfig,
}

impl EvaluationTask {
    pub fn new(
        precomputed: Arc<PrecomputedItems>,
        model_evaluator: Arc<dyn Evaluator>,
        model_version_rx: watch::Receiver<u64>,
        config: EvaluationConfig,
    ) -> Self {
        Self {
            precomputed,
            model_evaluator,
            model_version_rx,
            config,
        }
    }

    /// Run the evaluation loop. Waits for the model version to advance by
    /// `eval_interval_steps`, then plays `eval_games` games, logs metrics, and
    /// repeats. Returns when the model version sender is dropped.
    pub async fn run(&mut self) {
        let mut last_eval_version: u64 = 0;

        loop {
            // Compute the version threshold for the next eval run.
            let next_eval_at = last_eval_version + self.config.eval_interval_steps;

            // Wait until the model version crosses the threshold.
            loop {
                let current = *self.model_version_rx.borrow();
                if current >= next_eval_at {
                    break;
                }
                // Wait for next version change; if sender dropped, return.
                if self.model_version_rx.changed().await.is_err() {
                    return;
                }
            }

            let current_version = *self.model_version_rx.borrow();
            last_eval_version = current_version;

            // Play eval_games games using the model evaluator (model vs itself).
            let game_config = GameConfig {
                num_simulations: self.config.num_simulations,
                exploration_constant: 2.0,
                temperature_moves: self.config.temperature_moves,
            };

            let mut white_wins: usize = 0;
            let mut black_wins: usize = 0;
            let mut draws: usize = 0;
            let mut total_length: usize = 0;

            for _ in 0..self.config.eval_games {
                let traj = play_game(
                    self.precomputed.clone(),
                    self.model_evaluator.clone(),
                    current_version,
                    game_config.clone(),
                )
                .await;

                total_length += traj.steps.len();

                match traj.game_outcome {
                    o if o > 0.5 => white_wins += 1,
                    o if o < -0.5 => black_wins += 1,
                    _ => draws += 1,
                }
            }

            let total = self.config.eval_games;
            let decisive = white_wins + black_wins;
            let decisive_ratio = decisive as f64 / total as f64;
            let white_win_rate = white_wins as f64 / total as f64;
            let avg_length = total_length as f64 / total as f64;

            println!(
                "[eval] v{version} white_wins={ww} black_wins={bw} draws={draws} \
                 white_win_rate={wwr:.2} decisive_ratio={dr:.2} avg_length={avg:.1} self_play",
                version = current_version,
                ww = white_wins,
                bw = black_wins,
                draws = draws,
                wwr = white_win_rate,
                dr = decisive_ratio,
                avg = avg_length,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::watch;

    #[tokio::test]
    async fn test_evaluation_task_completes() {
        let precomputed = Arc::new(PrecomputedItems::begin_precomputing());
        let evaluator: Arc<dyn Evaluator> = Arc::new(RandomEvaluator);

        let (version_tx, version_rx) = watch::channel(0u64);

        let config = EvaluationConfig {
            eval_interval_steps: 1,
            eval_games: 2,
            num_simulations: 2,
            temperature_moves: 5,
        };

        let mut task = EvaluationTask::new(
            precomputed,
            evaluator,
            version_rx,
            config,
        );

        // Send version=1 so the eval task triggers immediately.
        version_tx.send(1).expect("send failed");

        // Run one eval cycle: the task will play 2 games then wait for the next version.
        // We drop the sender after a short delay to make run() return.
        let task_handle = tokio::spawn(async move {
            task.run().await;
        });

        // Give the task time to complete one eval cycle (2 games with 2 sims each).
        tokio::time::sleep(std::time::Duration::from_secs(30)).await;

        // Drop sender — this causes model_version_rx.changed() to return Err, ending the loop.
        drop(version_tx);

        let result = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            task_handle,
        )
        .await;

        assert!(result.is_ok(), "EvaluationTask did not complete in time");
        assert!(result.unwrap().is_ok(), "EvaluationTask panicked");
    }
}
