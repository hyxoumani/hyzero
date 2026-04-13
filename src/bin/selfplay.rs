use std::env;
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use tokio::sync::{mpsc, watch};

use hyzero::PrecomputedItems;
use hyzero::py::{PyO3Backend, PyTrainingThread};
use hyzero::selfplay::{
    InferenceBatcher, BatcherConfig, ChannelEvaluator,
    SelfPlayConfig, SelfPlayCoordinator,
    EvaluationConfig, EvaluationTask, RandomEvaluator,
};
use hyzero::selfplay::game_task::GameConfig;
use hyzero::mcts::evaluator::Evaluator;

/// Runtime configuration for the self-play binary.
/// All fields can be overridden via environment variables; falls back to Default.
struct RunConfig {
    // Self-play
    max_concurrent_games: usize,
    num_simulations: u32,
    temperature_moves: u32,
    // Batching
    max_batch_size: usize,
    batch_timeout_ms: u64,
    // Evaluation
    eval_interval_steps: u64,
    eval_games: usize,
    eval_num_simulations: u32,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            max_concurrent_games: 4,
            num_simulations: 50,
            temperature_moves: 15,
            max_batch_size: 32,
            batch_timeout_ms: 10,
            eval_interval_steps: 200,
            eval_games: 10,
            eval_num_simulations: 50,
        }
    }
}

#[tokio::main]
async fn main() {
    println!("[selfplay] Initializing...");

    let defaults = RunConfig::default();
    let config = RunConfig {
        max_concurrent_games: env::var("HYZERO_GAMES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(defaults.max_concurrent_games),
        num_simulations: env::var("HYZERO_SIMS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(defaults.num_simulations),
        temperature_moves: env::var("HYZERO_TEMP_MOVES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(defaults.temperature_moves),
        max_batch_size: env::var("HYZERO_BATCH_SIZE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(defaults.max_batch_size),
        batch_timeout_ms: env::var("HYZERO_BATCH_TIMEOUT_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(defaults.batch_timeout_ms),
        eval_interval_steps: env::var("HYZERO_EVAL_INTERVAL")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(defaults.eval_interval_steps),
        eval_games: env::var("HYZERO_EVAL_GAMES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(defaults.eval_games),
        eval_num_simulations: env::var("HYZERO_EVAL_SIMS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(defaults.eval_num_simulations),
    };

    // 1. Precompute move tables
    let precomputed = Arc::new(PrecomputedItems::begin_precomputing());
    println!("[selfplay] Precomputed move tables ready");

    // 2. Create channels
    let (inference_tx, inference_rx) = mpsc::channel(256);
    let (trajectory_tx, trajectory_rx) = mpsc::channel(64);
    let (version_tx, version_rx) = watch::channel(1u64);
    let (weight_tx, weight_rx) = watch::channel::<Option<Vec<u8>>>(None);

    // 3. Create the Python InferenceServer first so we can share it.
    //    We need one reference for the backend and a clone for the weight loader.
    println!("[selfplay] Creating Python InferenceServer...");
    let server: Py<PyAny> = Python::attach(|py| {
        let config_obj = PyModule::import(py, "hyzero.config")
            .expect("hyzero Python package not found — ensure it is installed")
            .getattr("DEFAULT_CONFIG")
            .expect("DEFAULT_CONFIG missing from hyzero.config")
            .into_pyobject(py)
            .expect("into_pyobject failed")
            .unbind();
        let cls = PyModule::import(py, "hyzero.inference.server")
            .expect("hyzero.inference.server not found")
            .getattr("InferenceServer")
            .expect("InferenceServer class not found");
        let srv: Py<PyAny> = cls
            .call1((config_obj, "cpu"))
            .expect("InferenceServer() constructor failed")
            .unbind();
        srv
    });

    // Clone the Py<PyAny> ref-counted handle for the weight loader task.
    let server_for_weights: Py<PyAny> = Python::attach(|py| server.clone_ref(py));

    // 4. Spawn inference batcher with the PyO3Backend.
    let backend = Box::new(PyO3Backend::new(server, 64));
    let batcher_config = BatcherConfig {
        max_batch_size: config.max_batch_size,
        batch_timeout_ms: config.batch_timeout_ms,
    };
    let mut batcher = InferenceBatcher::new(inference_rx, backend, batcher_config);
    tokio::spawn(async move {
        batcher.run().await;
        println!("[selfplay] Inference batcher stopped");
    });

    // 5. Spawn training thread backed by the Python Trainer.
    println!("[selfplay] Creating Python Trainer...");
    let mut training = PyTrainingThread::from_default_config("cpu", trajectory_rx, version_tx, weight_tx, None)
        .expect("Failed to create PyTrainingThread — is hyzero Python package installed?");
    tokio::spawn(async move {
        training.run().await;
    });

    // 6. Spawn weight loader: watch for new weights and push them into the InferenceServer.
    let mut weight_rx_task = weight_rx;
    tokio::spawn(async move {
        while weight_rx_task.changed().await.is_ok() {
            let maybe_weights = weight_rx_task.borrow_and_update().clone();
            if let Some(bytes) = maybe_weights {
                Python::attach(|py| {
                    let py_bytes = PyBytes::new(py, &bytes);
                    if let Err(e) = server_for_weights.call_method1(py, "load_weights", (py_bytes,)) {
                        eprintln!("[selfplay] load_weights error: {e}");
                    }
                });
            }
        }
        println!("[selfplay] Weight loader stopped");
    });

    // 7. Create evaluator and coordinator.
    let evaluator: Arc<dyn Evaluator> =
        Arc::new(ChannelEvaluator::new(inference_tx));

    let selfplay_config = SelfPlayConfig {
        max_concurrent_games: config.max_concurrent_games,
        game_config: GameConfig {
            num_simulations: config.num_simulations,
            exploration_constant: 1.5,
            temperature_moves: config.temperature_moves,
        },
    };

    let coordinator = SelfPlayCoordinator::new(
        precomputed.clone(),
        evaluator,
        trajectory_tx,
        version_rx.clone(),
        selfplay_config,
    );

    // 8. Spawn evaluation task — runs games against itself to track learning signal.
    let eval_evaluator: Arc<dyn Evaluator> = Arc::new(RandomEvaluator);
    let eval_config = EvaluationConfig {
        eval_interval_steps: config.eval_interval_steps,
        eval_games: config.eval_games,
        num_simulations: config.eval_num_simulations,
        temperature_moves: config.temperature_moves,
    };
    let mut eval_task = EvaluationTask::new(
        precomputed,
        eval_evaluator,
        version_rx,
        eval_config,
    );
    tokio::spawn(async move {
        eval_task.run().await;
        println!("[selfplay] Evaluation task stopped");
    });

    println!(
        "[selfplay] Starting self-play loop ({} concurrent games, {} sims/move)",
        config.max_concurrent_games, config.num_simulations
    );
    coordinator.run().await;
}
