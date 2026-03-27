use std::sync::Arc;

use tokio::sync::{mpsc, watch};

use hyzero::PrecomputedItems;
use hyzero::selfplay::{
    InferenceBatcher, BatcherConfig, RandomBackend, ChannelEvaluator,
    SelfPlayConfig, SelfPlayCoordinator, TrainingConfig, TrainingThread,
};
use hyzero::selfplay::game_task::GameConfig;

#[tokio::main]
async fn main() {
    println!("[selfplay] Initializing...");

    // 1. Precompute move tables
    let precomputed = Arc::new(PrecomputedItems::begin_precomputing());
    println!("[selfplay] Precomputed move tables ready");

    // 2. Create channels
    let (inference_tx, inference_rx) = mpsc::channel(256);
    let (trajectory_tx, trajectory_rx) = mpsc::channel(64);
    let (version_tx, version_rx) = watch::channel(1u64);

    // 3. Spawn inference batcher with random backend
    let backend = Box::new(RandomBackend::new(64));
    let batcher_config = BatcherConfig {
        max_batch_size: 32,
        batch_timeout_ms: 1,
    };
    let mut batcher = InferenceBatcher::new(inference_rx, backend, batcher_config);
    tokio::spawn(async move {
        batcher.run().await;
        println!("[selfplay] Inference batcher stopped");
    });

    // 4. Spawn training thread
    let training_config = TrainingConfig {
        min_samples_before_training: 100,
        train_batch_size: 32,
        unroll_k: 5,
        max_replay_trajectories: 1_000,
        checkpoint_interval: 50,
        ..TrainingConfig::default()
    };
    let mut training = TrainingThread::new(trajectory_rx, version_tx, training_config);
    tokio::spawn(async move {
        training.run().await;
    });

    // 5. Create evaluator and coordinator
    let evaluator: Arc<dyn hyzero::mcts::evaluator::Evaluator> =
        Arc::new(ChannelEvaluator::new(inference_tx));

    let selfplay_config = SelfPlayConfig {
        max_concurrent_games: 4,
        game_config: GameConfig {
            num_simulations: 50,
            exploration_constant: 1.5,
            temperature_moves: 15,
        },
    };

    let coordinator = SelfPlayCoordinator::new(
        precomputed,
        evaluator,
        trajectory_tx,
        version_rx,
        selfplay_config,
    );

    println!("[selfplay] Starting self-play loop (4 concurrent games, 50 sims/move)");
    coordinator.run().await;
}
