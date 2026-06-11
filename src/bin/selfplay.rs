use std::env;
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use tokio::sync::{mpsc, watch};

use hyzero::mcts::evaluator::Evaluator;
use hyzero::py::{PyO3Backend, PyTrainingThread};
use hyzero::selfplay::evaluation::RandomEvaluator;
use hyzero::selfplay::game_task::{temperature_moves, GameConfig};
use hyzero::selfplay::{
    BatcherConfig, ChampionStore, ChannelEvaluator, EvaluationConfig, EvaluationTask,
    InferenceBatcher, RandomBackend, SelfPlayConfig, SelfPlayCoordinator, SwappableBackend,
};
use hyzero::PrecomputedItems;

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

/// Path of the checkpoint used for champion init + trainer resume.
/// Defaults to `checkpoints/best.pt`; override via `HYZERO_RESUME_FROM` to
/// start from a different checkpoint (e.g. `checkpoints/pretrain_dynamics.pt`
/// for a SimSiam-dynamics warm-start). Both champion and trainer read the
/// same file so the eval ladder's baseline matches the challenger's starting
/// weights.
fn resume_checkpoint_path() -> String {
    std::env::var("HYZERO_RESUME_FROM").unwrap_or_else(|_| "checkpoints/best.pt".to_string())
}

/// Load the bytes from the configured resume checkpoint.
fn read_resume_checkpoint() -> std::io::Result<Vec<u8>> {
    std::fs::read(resume_checkpoint_path())
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
    // Elo ladder (pool-based promotion gate)
    elo_k_factor: f32,
    pool_size: usize,
    promotion_elo_delta: f32,
    opponent_initial_elo: f32,
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
            elo_k_factor: 32.0,
            pool_size: 3,
            promotion_elo_delta: 20.0,
            opponent_initial_elo: 1500.0,
        }
    }
}

/// Build a `RunConfig` from the environment + defaults. Factored out so unit
/// tests can drive it under a serial lock without forking a binary.
fn run_config_from_env() -> RunConfig {
    let defaults = RunConfig::default();
    RunConfig {
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
        elo_k_factor: env::var("HYZERO_ELO_K_FACTOR")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(defaults.elo_k_factor),
        pool_size: env::var("HYZERO_POOL_SIZE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(defaults.pool_size),
        promotion_elo_delta: env::var("HYZERO_PROMOTION_ELO_DELTA")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(defaults.promotion_elo_delta),
        opponent_initial_elo: env::var("HYZERO_OPPONENT_INITIAL_ELO")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(defaults.opponent_initial_elo),
    }
}

#[tokio::main]
async fn main() {
    println!("[selfplay] Initializing...");

    let device = std::env::var("HYZERO_DEVICE").unwrap_or_else(|_| "cpu".to_string());
    println!("[selfplay] Device: {device}");

    let config = run_config_from_env();

    // Bootstrap-path notice: HYZERO_PROMOTION_THRESHOLD now only governs the
    // empty-pool path. Once any best_v{NNN}.pt exists, gating switches to Elo.
    if env::var("HYZERO_PROMOTION_THRESHOLD").is_ok() {
        eprintln!(
            "[selfplay] NOTE: HYZERO_PROMOTION_THRESHOLD applies only to the empty-pool bootstrap path; once any archive exists, gating switches to Elo (HYZERO_PROMOTION_ELO_DELTA)."
        );
    }

    // Cooldown semantics notice: `promotion_cooldown_games` counts games, not
    // cycles. With pool_size=K and games_per_side=g, one cycle = 2*K*g games.
    // The default (0) is a no-op so existing baselines are unaffected.
    if config.promotion_cooldown_games > 0 {
        let n = 2 * config.pool_size * config.games_per_side;
        eprintln!(
            "[selfplay] NOTE: promotion_cooldown_games={cd} counts games (not cycles). \
             With pool_size={ps} and games_per_side={gps}, one cycle = {n} games.",
            cd = config.promotion_cooldown_games,
            ps = config.pool_size,
            gps = config.games_per_side,
        );
    }

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
            .call1((config_unbound, device.as_str()))
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
    //    If the configured resume checkpoint exists on disk, we boot the
    //    champion batcher immediately with those frozen weights instead of
    //    starting from RandomBackend.
    let resume_path_str = resume_checkpoint_path();
    let resume_path = std::path::Path::new(&resume_path_str).to_path_buf();
    let (champion_store_evaluator, champion_store_version, champion_backend_handle, champion_error_flag) =
        if resume_path.exists() {
            // Determine starting version from archived best_vNNN.pt files.
            let starting_version = match find_latest_archive_version() {
                Some(v) => v,
                None => {
                    eprintln!(
                        "[selfplay] No best_vNNN.pt archive found; starting ladder at version=1 \
                         (resume from {})",
                        resume_path.display()
                    );
                    1
                }
            };

            // Load frozen weights from the configured resume checkpoint.
            match read_resume_checkpoint() {
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
                                .call1((config_unbound, device.as_str()))
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

                    // Keepalive sender: the only other holder of `champion_tx` is the
                    // champion `ChannelEvaluator` stored in `ChampionStore`. A promotion
                    // swaps that stored evaluator (dropping the old one), which would
                    // close `champion_rx` and stop the champion batcher — stranding any
                    // in-flight bootstrap-cycle eval request. Leaking one clone for the
                    // process lifetime keeps the batcher alive across promotions so the
                    // teardown can never strand mid-cycle work.
                    std::mem::forget(champion_tx.clone());

                    // Capture the champion evaluator's sticky error-flag handle so
                    // the bootstrap (empty-pool) eval path can skip a promotion
                    // decision when this champion's batcher dies mid-cycle and its
                    // games degrade to neutral evals.
                    let champion_channel_eval =
                        ChannelEvaluator::with_channels(champion_tx, champion_hidden_channels);
                    let champion_error_flag = Some(champion_channel_eval.error_flag());
                    let champion_eval: Arc<dyn Evaluator> = Arc::new(champion_channel_eval);

                    println!(
                        "[selfplay] Loaded champion from {} (version={starting_version})",
                        resume_path.display()
                    );

                    (champion_eval, starting_version, champion_handle, champion_error_flag)
                }
                Err(e) => {
                    eprintln!(
                        "[selfplay] WARNING: {} exists but could not be read ({e}); falling back to RandomEvaluator",
                        resume_path.display()
                    );
                    let initial_champion_backend: Box<dyn hyzero::selfplay::InferenceBackend> =
                        Box::new(RandomBackend::new(hidden_channels));
                    let (_swappable, champion_handle) =
                        SwappableBackend::new(initial_champion_backend);
                    let eval: Arc<dyn Evaluator> = Arc::new(RandomEvaluator);
                    println!(
                        "[selfplay] No existing resume checkpoint; starting with RandomEvaluator (version=0)"
                    );
                    // RandomEvaluator has no inference batcher to die → no flag.
                    (eval, 0, champion_handle, None)
                }
            }
        } else {
            let initial_champion_backend: Box<dyn hyzero::selfplay::InferenceBackend> =
                Box::new(RandomBackend::new(hidden_channels));
            let (_swappable, champion_handle) = SwappableBackend::new(initial_champion_backend);
            let eval: Arc<dyn Evaluator> = Arc::new(RandomEvaluator);
            println!(
                "[selfplay] No existing resume checkpoint at {}; starting with RandomEvaluator (version=0)",
                resume_path.display()
            );
            // RandomEvaluator has no inference batcher to die → no flag.
            (eval, 0, champion_handle, None)
        };

    // 6. Spawn training thread backed by the Python Trainer.
    println!("[selfplay] Creating Python Trainer...");
    let resume_ckpt: Option<&str> = if resume_path.exists() {
        println!("[selfplay] Resuming training from {}", resume_path.display());
        Some(resume_path_str.as_str())
    } else {
        None
    };
    let mut training =
        PyTrainingThread::from_default_config(&device, trajectory_rx, version_tx, weight_tx, resume_ckpt)
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
                    if let Err(e) = server_for_weights.call_method1(py, "load_weights", (py_bytes,))
                    {
                        eprintln!("[selfplay] load_weights error: {e}");
                    }
                });
            }
        }
        println!("[selfplay] Weight loader stopped");
    });

    // 7b. Spawn an additional InferenceServer + batcher dedicated to the
    //     opponent (pool) side of the Elo ladder. Weights are reloaded from
    //     `checkpoints/best_v{NNN}.pt` once per pool member per cycle by the
    //     `EvaluationTask`. The server starts uninitialized; if the pool is
    //     empty (bootstrap), the ladder falls back to the legacy win-rate gate
    //     and this opponent batcher sits idle.
    println!("[selfplay] Creating opponent InferenceServer (Elo ladder)...");
    let (opponent_server, opponent_hidden_channels): (Py<PyAny>, usize) = Python::attach(|py| {
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
            .call1((config_unbound, device.as_str()))
            .expect("opponent InferenceServer() constructor failed")
            .unbind();
        (srv, hc)
    });
    // Direct Py<PyAny> handle for the EvaluationTask to call `load_weights`.
    let opponent_server_handle: Arc<std::sync::Mutex<Py<PyAny>>> =
        Arc::new(std::sync::Mutex::new(Python::attach(|py| {
            opponent_server.clone_ref(py)
        })));
    let (opponent_tx, opponent_rx) = mpsc::channel(256);
    let opponent_backend = Box::new(PyO3Backend::new(opponent_server, opponent_hidden_channels));
    let mut opponent_batcher =
        InferenceBatcher::new(opponent_rx, opponent_backend, batcher_config.clone());
    tokio::spawn(async move {
        opponent_batcher.run().await;
        println!("[selfplay] Opponent inference batcher stopped");
    });
    let opponent_evaluator: Arc<dyn Evaluator> = Arc::new(ChannelEvaluator::with_channels(
        opponent_tx,
        opponent_hidden_channels,
    ));

    // 8. Create evaluator and coordinator.
    let evaluator: Arc<dyn Evaluator> =
        Arc::new(ChannelEvaluator::with_channels(inference_tx.clone(), hidden_channels));

    // Replay capture: opt-in via HYZERO_REPLAY_DIR. When set, every completed
    // self-play game writes a `.replay` file (bincode) into the given directory
    // for use with `cargo run --bin replay -- <file>`. Files are large
    // (per-ply MCTS dump) — do not enable for unattended long runs.
    let replay_dir: Option<Arc<std::path::PathBuf>> = env::var("HYZERO_REPLAY_DIR")
        .ok()
        .filter(|s| !s.is_empty())
        .map(|s| Arc::new(std::path::PathBuf::from(s)));
    if let Some(dir) = replay_dir.as_ref() {
        println!("[selfplay] Replay capture ON → {}", dir.display());
    }

    let selfplay_config = SelfPlayConfig {
        max_concurrent_games: selfplay_games,
        game_config: GameConfig {
            num_simulations: config.num_simulations,
            exploration_constant: 1.5,
            // Self-play exploration window from HYZERO_TEMPERATURE_MOVES (default
            // 30, clamped [1,200]). Eval keeps config.temperature_moves
            // (HYZERO_TEMP_MOVES) below so the ladder is unaffected.
            temperature_moves: temperature_moves(),
            replay_dir: replay_dir.clone(),
            // Self-play must never adjudicate (passivity-attractor guard).
            adjudicate_at_cap: false,
            adjudication_material_margin: 5,
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
    // Build the concrete challenger evaluator first so its sticky error-flag handle
    // can be registered with the eval task: a cycle whose challenger recovered an
    // inference call to neutral (dead batcher) must skip its promotion decision
    // rather than record a garbage promotion.
    let challenger_channel_eval = ChannelEvaluator::with_channels(inference_tx, hidden_channels);
    let challenger_error_flag = challenger_channel_eval.error_flag();
    let challenger_eval: Arc<dyn Evaluator> = Arc::new(challenger_channel_eval);
    let eval_config = EvaluationConfig {
        games_per_side: config.games_per_side,
        promotion_threshold: config.promotion_threshold,
        promotion_cooldown_games: config.promotion_cooldown_games,
        num_simulations: config.eval_num_simulations,
        temperature_moves: config.temperature_moves,
        poll_interval_ms: 500,
        champion_score_weight: config.champion_score_weight,
        elo_k_factor: config.elo_k_factor,
        pool_size: config.pool_size,
        promotion_elo_delta: config.promotion_elo_delta,
        opponent_initial_elo: config.opponent_initial_elo,
        ..EvaluationConfig::default()
    };

    println!(
        "[selfplay] Starting evaluation ladder ({} games/side, pool_size={}, elo_delta={:.1}, \
         weight={:.1}, bootstrap_threshold={:.2})",
        config.games_per_side,
        config.pool_size,
        config.promotion_elo_delta,
        config.champion_score_weight,
        config.promotion_threshold,
    );

    let eval_task_obj = EvaluationTask::new(
        precomputed.clone(),
        challenger_eval,
        version_rx,
        latest_ckpt_path,
        champion_store,
        eval_config,
    )
    .with_champion_backend(champion_backend_handle)
    .with_opponent(opponent_evaluator, opponent_server_handle)
    .with_error_flags(Some(challenger_error_flag), champion_error_flag);

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

#[cfg(test)]
mod tests {
    use super::*;

    /// Serial lock for the env-var tests below. Mirrors the
    /// `decisive_frac_env_lock` pattern in `src/data/replay_buffer.rs:264-266`.
    /// `std::env::set_var` is process-global and unsafe to race.
    fn elo_env_lock() -> &'static std::sync::Mutex<()> {
        static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
        LOCK.get_or_init(|| std::sync::Mutex::new(()))
    }

    fn clear_elo_env() {
        // SAFETY: protected by elo_env_lock() at all call sites.
        unsafe {
            std::env::remove_var("HYZERO_POOL_SIZE");
            std::env::remove_var("HYZERO_PROMOTION_ELO_DELTA");
            std::env::remove_var("HYZERO_ELO_K_FACTOR");
            std::env::remove_var("HYZERO_OPPONENT_INITIAL_ELO");
        }
    }

    #[test]
    fn from_env_returns_defaults_when_unset() {
        let _guard = elo_env_lock().lock().unwrap();
        clear_elo_env();
        let cfg = run_config_from_env();
        assert!((cfg.elo_k_factor - 32.0).abs() < f32::EPSILON);
        assert_eq!(cfg.pool_size, 3);
        assert!((cfg.promotion_elo_delta - 20.0).abs() < f32::EPSILON);
        assert!((cfg.opponent_initial_elo - 1500.0).abs() < f32::EPSILON);
    }

    #[test]
    fn from_env_parses_pool_size_override() {
        let _guard = elo_env_lock().lock().unwrap();
        clear_elo_env();
        // SAFETY: protected by elo_env_lock(); no concurrent env-var access.
        unsafe {
            std::env::set_var("HYZERO_POOL_SIZE", "5");
        }
        let cfg = run_config_from_env();
        clear_elo_env();
        assert_eq!(cfg.pool_size, 5);
    }

    #[test]
    fn from_env_parses_elo_delta_override() {
        let _guard = elo_env_lock().lock().unwrap();
        clear_elo_env();
        // SAFETY: protected by elo_env_lock(); no concurrent env-var access.
        unsafe {
            std::env::set_var("HYZERO_PROMOTION_ELO_DELTA", "30.0");
        }
        let cfg = run_config_from_env();
        clear_elo_env();
        assert!((cfg.promotion_elo_delta - 30.0).abs() < f32::EPSILON);
    }
}
