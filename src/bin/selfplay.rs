use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use tokio::sync::{mpsc, watch};

use hyzero::PrecomputedItems;
use hyzero::py::{PyO3Backend, PyTrainingThread};
use hyzero::selfplay::{
    InferenceBatcher, BatcherConfig, ChannelEvaluator,
    SelfPlayConfig, SelfPlayCoordinator,
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
    let (weight_tx, weight_rx) = watch::channel::<Option<Vec<u8>>>(None);

    // 3. Create the Python InferenceServer first so we can share it.
    //    We need one reference for the backend and a clone for the weight loader.
    println!("[selfplay] Creating Python InferenceServer...");
    let server: Py<PyAny> = Python::attach(|py| {
        let config = PyModule::import(py, "hyzero.config")
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
            .call1((config, "cpu"))
            .expect("InferenceServer() constructor failed")
            .unbind();
        srv
    });

    // Clone the Py<PyAny> ref-counted handle for the weight loader task.
    let server_for_weights: Py<PyAny> = Python::attach(|py| server.clone_ref(py));

    // 4. Spawn inference batcher with the PyO3Backend.
    let backend = Box::new(PyO3Backend::new(server, 64));
    let batcher_config = BatcherConfig {
        max_batch_size: 32,
        batch_timeout_ms: 10,
    };
    let mut batcher = InferenceBatcher::new(inference_rx, backend, batcher_config);
    tokio::spawn(async move {
        batcher.run().await;
        println!("[selfplay] Inference batcher stopped");
    });

    // 5. Spawn training thread backed by the Python Trainer.
    println!("[selfplay] Creating Python Trainer...");
    let mut training = PyTrainingThread::from_default_config("cpu", trajectory_rx, version_tx, weight_tx)
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
    let evaluator: Arc<dyn hyzero::mcts::evaluator::Evaluator> =
        Arc::new(ChannelEvaluator::new(inference_tx));

    let selfplay_config = SelfPlayConfig {
        max_concurrent_games: 8,
        game_config: GameConfig {
            num_simulations: 25,
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

    println!("[selfplay] Starting self-play loop (8 concurrent games, 25 sims/move)");
    coordinator.run().await;
}
