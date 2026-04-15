use std::env;
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use tokio::sync::{mpsc, watch};

use hyzero::PrecomputedItems;
use hyzero::py::{PyO3Backend, PyTrainingThread};
use hyzero::selfplay::{
    InferenceBatcher, BatcherConfig, ChannelEvaluator, SwappableBackend,
    RandomBackend,
    SelfPlayConfig, SelfPlayCoordinator,
    EvaluationConfig, EvaluationTask,
    ChampionStore,
};
use hyzero::selfplay::game_task::GameConfig;
use hyzero::selfplay::evaluation::RandomEvaluator;
use hyzero::mcts::evaluator::Evaluator;

/// Scan `checkpoints/` for `best_vNNN.pt` files and return the highest NNN found.
///
/// Returns `None` if the directory does not exist or contains no matching files.
fn find_latest_archive_version() -> Option<u64> {
    let dir = std::fs::read_dir("checkpoints").ok()?;
    let mut max_version: Option<u64> = None;
    for entry in dir.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        // Match pattern: best_vNNN.pt
        if let Some(inner) = name_str.strip_prefix("best_v") {
            if let Some(num_str) = inner.strip_suffix(".pt") {
                if let Ok(v) = num_str.parse::<u64>() {
                    max_version = Some(max_version.map_or(v, |m: u64| m.max(v)));
                }
            }
        }
    }
    max_version
}

/// Load the bytes from `checkpoints/best.pt`.
fn read_best_pt() -> std::io::Result<Vec<u8>> {
    std::fs::read("checkpoints/best.pt")
}

/// Runtime configuration for the self-play binary.
/// All fields can be overridden via environment variables; falls back to Default.
struct RunConfig {
    // Self-play
    /// Total game slots (1 reserved for eval, rest for self-play). Default 5.
    total_games: usize,
    num_simulations: u32,
    temperature_moves: u32,
    // Batching
    max_batch_size: usize,
    batch_timeout_ms: u64,
    // Evaluation ladder
    games_per_side: usize,
    promotion_threshold: f64,
    promotion_cooldown_games: usize,
    eval_num_simulations: u32,
    champion_score_weight: f64,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            total_games: 5,
            num_simulations: 40,
            temperature_moves: 15,
            max_batch_size: 32,
            batch_timeout_ms: 10,
            games_per_side: 4,
            promotion_threshold: 0.55,
            promotion_cooldown_games: 0,
            eval_num_simulations: 50,
            champion_score_weight: 2.0,
        }
    }
}

#[tokio::main]
async fn main() {
    println!("[selfplay] Initializing...");

    let defaults = RunConfig::default();
    let config = RunConfig {
        total_games: env::var("HYZERO_GAMES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(defaults.total_games),
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
        games_per_side: env::var("HYZERO_GAMES_PER_SIDE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(defaults.games_per_side),
        promotion_threshold: env::var("HYZERO_PROMOTION_THRESHOLD")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(defaults.promotion_threshold),
        promotion_cooldown_games: env::var("HYZERO_PROMOTION_COOLDOWN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(defaults.promotion_cooldown_games),
        eval_num_simulations: env::var("HYZERO_EVAL_SIMS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(defaults.eval_num_simulations),
        champion_score_weight: env::var("HYZERO_CHAMPION_SCORE_WEIGHT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(defaults.champion_score_weight),
    };

    // Derive self-play concurrency: N-1 slots for games, 1 for eval.
    let selfplay_games = config.total_games.saturating_sub(1).max(1);

    // 1. Precompute move tables
    let precomputed = Arc::new(PrecomputedItems::begin_precomputing());
    println!("[selfplay] Precomputed move tables ready");

    // 2. Create channels
    let (inference_tx, inference_rx) = mpsc::channel(256);
    let (trajectory_tx, trajectory_rx) = mpsc::channel(64);
    let (version_tx, version_rx) = watch::channel(1u64);
    let (weight_tx, weight_rx) = watch::channel::<Option<Vec<u8>>>(None);

    // 3. Create the Python InferenceServer first so we can share it.
    println!("[selfplay] Creating Python InferenceServer...");
    let (server, hidden_channels): (Py<PyAny>, usize) = Python::attach(|py| {
        let config_obj = PyModule::import(py, "hyzero.config")
            .expect("hyzero Python package not found — ensure it is installed")
            .getattr("DEFAULT_CONFIG")
            .expect("DEFAULT_CONFIG missing from hyzero.config")
            .into_pyobject(py)
            .expect("into_pyobject failed");
        let hidden_channels: usize = config_obj
            .cast::<PyDict>()
            .expect("DEFAULT_CONFIG is not a dict")
            .get_item("hidden_channels")
            .expect("hidden_channels lookup failed")
            .expect("hidden_channels not in DEFAULT_CONFIG")
            .extract()
            .expect("hidden_channels is not a usize");
        let config_unbound = config_obj.unbind();
        let cls = PyModule::import(py, "hyzero.inference.server")
            .expect("hyzero.inference.server not found")
            .getattr("InferenceServer")
            .expect("InferenceServer class not found");
        let srv: Py<PyAny> = cls
            .call1((config_unbound, "cpu"))
            .expect("InferenceServer() constructor failed")
            .unbind();
        (srv, hidden_channels)
    });

    // Clone the Py<PyAny> ref-counted handle for the weight loader task.
    let server_for_weights: Py<PyAny> = Python::attach(|py| server.clone_ref(py));

    // 4. Spawn inference batcher with the PyO3Backend (for challenger / self-play).
    let backend = Box::new(PyO3Backend::new(server, hidden_channels));
    let batcher_config = BatcherConfig {
        max_batch_size: config.max_batch_size,
        batch_timeout_ms: config.batch_timeout_ms,
    };
    let mut batcher = InferenceBatcher::new(inference_rx, backend, batcher_config.clone());
    tokio::spawn(async move {
        batcher.run().await;
        println!("[selfplay] Inference batcher stopped");
    });

    // 5. Create swappable champion backend handle for hot-swap on promotion.
    //    If best.pt exists on disk, we boot the champion batcher immediately
    //    with the frozen weights instead of starting from RandomBackend.
    let best_pt_path = std::path::Path::new("checkpoints/best.pt");
    let (champion_store_evaluator, champion_store_version, champion_backend_handle) =
        if best_pt_path.exists() {
            // Determine starting version from archived best_vNNN.pt files.
            let starting_version = match find_latest_archive_version() {
                Some(v) => v,
                None => {
                    eprintln!(
                        "[selfplay] WARNING: best.pt exists but no best_vNNN.pt found; \
                         defaulting starting_version to 1"
                    );
                    1
                }
            };

            // Load frozen weights from best.pt.
            match read_best_pt() {
                Ok(best_pt_bytes) => {
                    // Create a fresh Python InferenceServer for the champion.
                    let (champion_server, champion_hidden_channels): (Py<PyAny>, usize) =
                        Python::attach(|py| {
                            let config_obj = PyModule::import(py, "hyzero.config")
                                .expect("hyzero Python package not found")
                                .getattr("DEFAULT_CONFIG")
                                .expect("DEFAULT_CONFIG missing")
                                .into_pyobject(py)
                                .expect("into_pyobject failed");
                            let hc: usize = config_obj
                                .cast::<PyDict>()
                                .expect("DEFAULT_CONFIG is not a dict")
                                .get_item("hidden_channels")
                                .expect("hidden_channels lookup failed")
                                .expect("hidden_channels not in DEFAULT_CONFIG")
                                .extract()
                                .expect("hidden_channels is not a usize");
                            let config_unbound = config_obj.unbind();
                            let cls = PyModule::import(py, "hyzero.inference.server")
                                .expect("hyzero.inference.server not found")
                                .getattr("InferenceServer")
                                .expect("InferenceServer class not found");
                            let srv: Py<PyAny> = cls
                                .call1((config_unbound, "cpu"))
                                .expect("champion InferenceServer() constructor failed")
                                .unbind();
                            (srv, hc)
                        });

                    // Load frozen weights into the champion server.
                    Python::attach(|py| {
                        let py_bytes = PyBytes::new(py, &best_pt_bytes);
                        champion_server
                            .call_method1(py, "load_weights", (py_bytes,))
                            .expect("[selfplay] failed to load best.pt into champion server");
                    });

                    // Spawn champion inference batcher backed by the frozen PyO3Backend.
                    let (champion_tx, champion_rx) = mpsc::channel(256);
                    let champion_backend_box =
                        Box::new(PyO3Backend::new(champion_server, champion_hidden_channels));
                    let initial_swappable_inner: Box<dyn hyzero::selfplay::InferenceBackend> =
                        champion_backend_box;
                    let (champion_swappable, champion_handle) =
                        SwappableBackend::new(initial_swappable_inner);
                    let mut champion_batcher = InferenceBatcher::new(
                        champion_rx,
                        Box::new(champion_swappable),
                        batcher_config.clone(),
                    );
                    tokio::spawn(async move {
                        champion_batcher.run().await;
                        println!("[selfplay] Champion inference batcher stopped");
                    });

                    let champion_eval: Arc<dyn Evaluator> =
                        Arc::new(ChannelEvaluator::new(champion_tx));

                    println!(
                        "[selfplay] Loaded champion from checkpoints/best.pt (version={starting_version})"
                    );

                    (champion_eval, starting_version, champion_handle)
                }
                Err(e) => {
                    eprintln!("[selfplay] WARNING: best.pt exists but could not be read ({e}); falling back to RandomEvaluator");
                    let initial_champion_backend: Box<dyn hyzero::selfplay::InferenceBackend> =
                        Box::new(RandomBackend::new(hidden_channels));
                    let (_swappable, champion_handle) =
                        SwappableBackend::new(initial_champion_backend);
                    let eval: Arc<dyn Evaluator> = Arc::new(RandomEvaluator);
                    println!("[selfplay] No existing best.pt; starting with RandomEvaluator (version=0)");
                    (eval, 0, champion_handle)
                }
            }
        } else {
            let initial_champion_backend: Box<dyn hyzero::selfplay::InferenceBackend> =
                Box::new(RandomBackend::new(hidden_channels));
            let (_swappable, champion_handle) = SwappableBackend::new(initial_champion_backend);
            let eval: Arc<dyn Evaluator> = Arc::new(RandomEvaluator);
            println!("[selfplay] No existing best.pt; starting with RandomEvaluator (version=0)");
            (eval, 0, champion_handle)
        };

    // 6. Spawn training thread backed by the Python Trainer.
    println!("[selfplay] Creating Python Trainer...");
    let mut training = PyTrainingThread::from_default_config(
        "cpu",
        trajectory_rx,
        version_tx,
        weight_tx,
        None,
    )
    .expect("Failed to create PyTrainingThread — is hyzero Python package installed?");

    // Share the latest-checkpoint-path handle with the eval task.
    let latest_ckpt_path = training.latest_checkpoint_path.clone();

    tokio::spawn(async move {
        training.run().await;
    });

    // 7. Spawn weight loader: watch for new weights and push them into the InferenceServer.
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

    // 8. Create evaluator and coordinator.
    let evaluator: Arc<dyn Evaluator> = Arc::new(ChannelEvaluator::new(inference_tx.clone()));

    let selfplay_config = SelfPlayConfig {
        max_concurrent_games: selfplay_games,
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

    // 9. Create the champion store using the evaluator and version resolved in step 5.
    //    If best.pt was found, this uses the loaded frozen model; otherwise RandomEvaluator.
    let champion_store = Arc::new(ChampionStore::new_with_version(
        champion_store_evaluator,
        5,
        champion_store_version,
    ));

    // 10. Spawn evaluation ladder task.
    let challenger_eval: Arc<dyn Evaluator> = Arc::new(ChannelEvaluator::new(inference_tx));
    let eval_config = EvaluationConfig {
        games_per_side: config.games_per_side,
        promotion_threshold: config.promotion_threshold,
        promotion_cooldown_games: config.promotion_cooldown_games,
        num_simulations: config.eval_num_simulations,
        temperature_moves: config.temperature_moves,
        poll_interval_ms: 500,
        champion_score_weight: config.champion_score_weight,
    };

    println!(
        "[selfplay] Starting evaluation ladder ({} games/side, threshold={:.2}, weight={:.1})",
        config.games_per_side, config.promotion_threshold, config.champion_score_weight
    );

    let eval_task_obj = EvaluationTask::new(
        precomputed.clone(),
        challenger_eval,
        version_rx,
        latest_ckpt_path,
        champion_store,
        eval_config,
    )
    .with_champion_backend(champion_backend_handle);

    let mut eval_task = eval_task_obj;
    tokio::spawn(async move {
        eval_task.run().await;
        println!("[selfplay] Evaluation task stopped");
    });

    println!(
        "[selfplay] Starting self-play loop ({} concurrent games, {} sims/move)",
        selfplay_games, config.num_simulations
    );
    coordinator.run().await;
}
