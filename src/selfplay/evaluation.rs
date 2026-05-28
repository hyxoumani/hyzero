use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use pyo3::prelude::*;
use tokio::sync::watch;

use crate::PrecomputedItems;
use crate::data::{BoardObservation, HiddenState, Policy, ActionIndex, NUM_ACTIONS};
use crate::mcts::evaluator::Evaluator;
use crate::selfplay::champion::ChampionStore;
use crate::selfplay::game_task::{DualGameOutcome, GameConfig, play_game_dual};

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

/// Configuration for the champion-challenger evaluation ladder.
#[derive(Debug, Clone)]
pub struct EvaluationConfig {
    /// Games played per side in each ladder match (total = 2 × games_per_side).
    pub games_per_side: usize,
    /// Win-rate threshold for promotion (0.0–1.0). Default 0.55.
    pub promotion_threshold: f64,
    /// Minimum games between promotion decisions (cooldown). Default 0.
    pub promotion_cooldown_games: usize,
    /// MCTS simulations per move during evaluation.
    pub num_simulations: u32,
    /// Moves before switching to greedy (temperature → 0).
    pub temperature_moves: u32,
    /// How often to poll for new training versions (ms).
    pub poll_interval_ms: u64,
    /// Multiplier applied to champion_version in the scoring formula.
    /// Read from HYZERO_CHAMPION_SCORE_WEIGHT at runtime (default 2.0).
    pub champion_score_weight: f64,
}

impl Default for EvaluationConfig {
    fn default() -> Self {
        Self {
            games_per_side: 4,
            promotion_threshold: 0.55,
            promotion_cooldown_games: 0,
            num_simulations: 50,
            temperature_moves: 15,
            poll_interval_ms: 500,
            champion_score_weight: 2.0,
        }
    }
}

/// Challenger evaluator: wraps `ChannelEvaluator` (or any `Arc<dyn Evaluator>`)
/// and represents the latest trained model.
pub struct EvaluationTask {
    precomputed: Arc<PrecomputedItems>,
    /// Challenger evaluator (latest trained model via inference batcher).
    challenger_evaluator: Arc<dyn Evaluator>,
    /// Watch channel for model version (written by training thread).
    model_version_rx: watch::Receiver<u64>,
    /// Shared latest checkpoint path (written by training thread).
    latest_checkpoint_path: Arc<Mutex<Option<PathBuf>>>,
    /// Champion store (shared with potential champion batcher).
    champion_store: Arc<ChampionStore>,
    /// Champion backend handle for hot-swap. When None, champion uses a closure-based
    /// approach (the champion_store is the source of truth).
    champion_backend: Option<Arc<Mutex<Box<dyn crate::selfplay::inference::InferenceBackend>>>>,
    /// Opponent evaluator used for pool ladder games. The opponent batcher
    /// is shared across all pool members — weights are swapped via
    /// `opponent_server_handle` before each opponent's games.
    opponent_evaluator: Option<Arc<dyn Evaluator>>,
    /// Direct handle to the Python `InferenceServer` backing `opponent_evaluator`,
    /// used to call `load_weights(bytes)` between pool members. When `None`, the
    /// Elo-ladder code path is skipped and the task falls back to the legacy
    /// single-opponent (champion) eval.
    opponent_server_handle: Option<Arc<Mutex<Py<PyAny>>>>,
    config: EvaluationConfig,
    cycle: u64,
    total_games_since_last_promotion: usize,
}

impl EvaluationTask {
    /// Create a new ladder evaluation task.
    pub fn new(
        precomputed: Arc<PrecomputedItems>,
        challenger_evaluator: Arc<dyn Evaluator>,
        model_version_rx: watch::Receiver<u64>,
        latest_checkpoint_path: Arc<Mutex<Option<PathBuf>>>,
        champion_store: Arc<ChampionStore>,
        config: EvaluationConfig,
    ) -> Self {
        Self {
            precomputed,
            challenger_evaluator,
            model_version_rx,
            latest_checkpoint_path,
            champion_store,
            champion_backend: None,
            opponent_evaluator: None,
            opponent_server_handle: None,
            config,
            cycle: 0,
            total_games_since_last_promotion: 0,
        }
    }

    /// Attach the swappable champion backend handle so promotion can hot-swap weights.
    pub fn with_champion_backend(
        mut self,
        backend: Arc<Mutex<Box<dyn crate::selfplay::inference::InferenceBackend>>>,
    ) -> Self {
        self.champion_backend = Some(backend);
        self
    }

    /// Attach the opponent evaluator + its `InferenceServer` handle for the
    /// pool-based Elo ladder. When set, each cycle iterates over archived
    /// `best_v{NNN}.pt` files, calls `load_weights(bytes)` on the held server,
    /// and plays `2 * games_per_side` games per pool member against this
    /// evaluator. When unset, the task falls back to single-opponent eval.
    pub fn with_opponent(
        mut self,
        evaluator: Arc<dyn Evaluator>,
        server_handle: Arc<Mutex<Py<PyAny>>>,
    ) -> Self {
        self.opponent_evaluator = Some(evaluator);
        self.opponent_server_handle = Some(server_handle);
        self
    }

    /// Write a single game to `logs/eval_games.pgn` in standard PGN format.
    fn write_pgn_game(
        cycle: u64,
        game_num: usize,
        white_label: &str,
        black_label: &str,
        outcome: &DualGameOutcome,
    ) {
        let result_str = if outcome.game_outcome > 0.5 {
            "1-0"
        } else if outcome.game_outcome < -0.5 {
            "0-1"
        } else {
            "1/2-1/2"
        };
        crate::selfplay::pgn::write_pgn_game(
            "logs/eval_games.pgn",
            &format!("Eval Cycle {cycle} Game {game_num}"),
            white_label,
            black_label,
            result_str,
            &outcome.moves,
        );
    }

    /// Run the evaluation ladder loop.
    ///
    /// On each cycle:
    /// 1. Wait for a new training version.
    /// 2. Play `2 × games_per_side` games (balanced White/Black assignment).
    /// 3. Compute win_rate for challenger.
    /// 4. If win_rate ≥ promotion_threshold → promote challenger to champion.
    /// 5. Log structured output for run_baseline.sh grep anchors.
    pub async fn run(&mut self) {
        let mut last_evaluated_version: u64 = 0;

        loop {
            // Wait for a new model version.
            loop {
                let current = *self.model_version_rx.borrow();
                if current > last_evaluated_version {
                    last_evaluated_version = current;
                    break;
                }
                if self.model_version_rx.changed().await.is_err() {
                    return; // Sender dropped → training done
                }
            }

            let challenger_version = last_evaluated_version;
            self.cycle += 1;

            // Get current champion evaluator for this eval cycle.
            let champion_eval = self.champion_store.champion().await;
            let champion_version = self.champion_store.version();

            let game_config = GameConfig {
                num_simulations: self.config.num_simulations,
                exploration_constant: 1.5,
                temperature_moves: self.config.temperature_moves,
                replay_dir: None,
            };

            let gps = self.config.games_per_side;
            let mut ladder_wins: usize = 0;
            let mut ladder_draws: usize = 0;
            let mut ladder_losses: usize = 0;

            // games_per_side games with challenger as White, champion as Black.
            for game_idx in 0..gps {
                let outcome = play_game_dual(
                    self.precomputed.clone(),
                    self.challenger_evaluator.clone(),
                    champion_eval.clone(),
                    game_config.clone(),
                )
                .await;

                Self::write_pgn_game(
                    self.cycle,
                    game_idx + 1,
                    &format!("challenger v{challenger_version}"),
                    &format!("champion v{champion_version}"),
                    &outcome,
                );

                // game_outcome is White-perspective. Challenger = White.
                match outcome.game_outcome {
                    o if o > 0.5 => ladder_wins += 1,
                    o if o < -0.5 => ladder_losses += 1,
                    _ => ladder_draws += 1,
                }
                self.total_games_since_last_promotion += 1;
            }

            // games_per_side games with champion as White, challenger as Black.
            for game_idx in 0..gps {
                let outcome = play_game_dual(
                    self.precomputed.clone(),
                    champion_eval.clone(),
                    self.challenger_evaluator.clone(),
                    game_config.clone(),
                )
                .await;

                Self::write_pgn_game(
                    self.cycle,
                    gps + game_idx + 1,
                    &format!("champion v{champion_version}"),
                    &format!("challenger v{challenger_version}"),
                    &outcome,
                );

                // game_outcome is White-perspective. Champion = White, challenger = Black.
                // Flip to get challenger-perspective.
                let challenger_perspective = -outcome.game_outcome;
                match challenger_perspective {
                    o if o > 0.5 => ladder_wins += 1,
                    o if o < -0.5 => ladder_losses += 1,
                    _ => ladder_draws += 1,
                }
                self.total_games_since_last_promotion += 1;
            }

            let total_games = 2 * gps;
            let win_rate = (ladder_wins as f64 + ladder_draws as f64 * 0.5) / total_games as f64;

            println!(
                "[eval] v{challenger_version} cycle={cycle} ladder_wins={w} ladder_draws={d} \
                 ladder_losses={l} win_rate={r:.3} champion_version={cv} ladder_match",
                cycle = self.cycle,
                w = ladder_wins,
                d = ladder_draws,
                l = ladder_losses,
                r = win_rate,
                cv = champion_version,
            );

            // Check cooldown
            let cooldown_ok = self.total_games_since_last_promotion
                >= self.config.promotion_cooldown_games
                || self.config.promotion_cooldown_games == 0;

            if win_rate >= self.config.promotion_threshold && cooldown_ok {
                // Read latest completed checkpoint path (may be None if no checkpoint yet).
                let ckpt_path = self
                    .latest_checkpoint_path
                    .lock()
                    .ok()
                    .and_then(|g| g.clone());

                // Promote: new champion = current challenger evaluator.
                let new_champ = self.challenger_evaluator.clone();
                self.champion_store
                    .promote(new_champ, challenger_version, ckpt_path.as_ref())
                    .await;

                self.total_games_since_last_promotion = 0;

                println!(
                    "[eval] promoted champion_version={cv} challenger_version={cv_train} win_rate={r:.3}",
                    cv = challenger_version,
                    cv_train = challenger_version,
                    r = win_rate,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::watch;

    #[tokio::test]
    async fn test_evaluation_config_defaults() {
        let config = EvaluationConfig::default();
        assert_eq!(config.games_per_side, 4);
        assert!((config.promotion_threshold - 0.55).abs() < f64::EPSILON);
        assert_eq!(config.num_simulations, 50);
    }

    #[tokio::test]
    async fn test_evaluation_task_completes_one_cycle() {
        let precomputed = Arc::new(crate::PrecomputedItems::begin_precomputing());
        let challenger: Arc<dyn Evaluator> = Arc::new(RandomEvaluator);
        let champion_eval: Arc<dyn Evaluator> = Arc::new(RandomEvaluator);

        let (version_tx, version_rx) = watch::channel(0u64);

        let ckpt_path: Arc<Mutex<Option<PathBuf>>> = Arc::new(Mutex::new(None));
        let champion_store = Arc::new(ChampionStore::new(champion_eval, 5));

        let config = EvaluationConfig {
            games_per_side: 1,
            promotion_threshold: 2.0, // Force no promotion in this test
            promotion_cooldown_games: 0,
            num_simulations: 2,
            temperature_moves: 2,
            poll_interval_ms: 10,
            champion_score_weight: 2.0,
        };

        let mut task = EvaluationTask::new(
            precomputed,
            challenger,
            version_rx,
            ckpt_path,
            champion_store,
            config,
        );

        // Send version=1 to trigger one cycle.
        version_tx.send(1).expect("send failed");

        let task_handle = tokio::spawn(async move {
            task.run().await;
        });

        // Give time for one eval cycle (2 games at 2 sims each).
        tokio::time::sleep(std::time::Duration::from_secs(30)).await;

        // Drop sender to end the loop.
        drop(version_tx);

        let result = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            task_handle,
        )
        .await;

        assert!(result.is_ok(), "EvaluationTask should complete");
        assert!(result.unwrap().is_ok(), "EvaluationTask should not panic");
    }

    #[tokio::test]
    async fn test_evaluation_task_promotes_when_threshold_zero() {
        let precomputed = Arc::new(crate::PrecomputedItems::begin_precomputing());
        let challenger: Arc<dyn Evaluator> = Arc::new(RandomEvaluator);
        let champion_eval: Arc<dyn Evaluator> = Arc::new(RandomEvaluator);

        let (version_tx, version_rx) = watch::channel(0u64);

        let ckpt_path: Arc<Mutex<Option<PathBuf>>> = Arc::new(Mutex::new(None));
        let champion_store = Arc::new(ChampionStore::new(champion_eval, 5));
        let store_ref = champion_store.clone();

        let config = EvaluationConfig {
            games_per_side: 1,
            promotion_threshold: 0.0, // Always promote
            promotion_cooldown_games: 0,
            num_simulations: 2,
            temperature_moves: 2,
            poll_interval_ms: 10,
            champion_score_weight: 2.0,
        };

        let mut task = EvaluationTask::new(
            precomputed,
            challenger,
            version_rx,
            ckpt_path,
            champion_store,
            config,
        );

        version_tx.send(5).expect("send failed");

        let task_handle = tokio::spawn(async move {
            task.run().await;
        });

        tokio::time::sleep(std::time::Duration::from_secs(30)).await;
        drop(version_tx);

        let result = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            task_handle,
        )
        .await;

        assert!(result.is_ok());
        // With threshold=0.0, champion_version should have been updated to 5.
        assert_eq!(store_ref.version(), 5, "champion_version should be 5 after forced promotion");
    }

    /// Verify the opponent `Py<PyAny>` reload path swaps actual weights into a held
    /// `InferenceServer`. Mirrors `python/tests/test_inference.py:102-138` byte-format:
    /// drive a `Trainer` for a handful of steps, dump weights, call `load_weights` via
    /// the held handle, and assert `root_setup_batch` output differs pre- vs. post-load.
    #[test]
    #[ignore = "requires hyzero Python package"]
    fn opponent_load_weights_changes_root_setup_output() {
        use pyo3::types::PyBytes;

        let result: pyo3::PyResult<()> = Python::attach(|py| {
            // Build two InferenceServers with the same config (defaults to "cpu").
            let cfg_mod = PyModule::import(py, "hyzero.config")?;
            let cfg = cfg_mod.getattr("DEFAULT_CONFIG")?;
            let srv_cls = PyModule::import(py, "hyzero.inference.server")?
                .getattr("InferenceServer")?;
            let server: Py<PyAny> = srv_cls.call1((cfg.clone(), "cpu"))?.unbind();

            // Hold a directly cloned handle, as EvaluationTask does.
            let opp_handle: Arc<Mutex<Py<PyAny>>> =
                Arc::new(Mutex::new(server.clone_ref(py)));

            // Build a numpy obs batch of shape [2, INPUT_PLANES, 8, 8]; pull
            // INPUT_PLANES from the config to keep this independent of constants.
            let np = PyModule::import(py, "numpy")?;
            let input_planes: usize = cfg
                .cast::<pyo3::types::PyDict>()?
                .get_item("input_planes")?
                .ok_or_else(|| {
                    pyo3::exceptions::PyKeyError::new_err("input_planes not in config")
                })?
                .extract()?;
            let randn = np.getattr("random")?.getattr("randn")?;
            let obs_f64 = randn.call1((2, input_planes, 8, 8))?;
            let obs = obs_f64.call_method1("astype", ("float32",))?;

            // Capture pre-load output (policies tensor index 1).
            let pre = server
                .call_method1(py, "root_setup_batch", (obs.clone(),))?;
            let policies_before = pre.bind(py).get_item(1)?.unbind();

            // Drive a Trainer for a few steps to diverge from init weights.
            let trainer_cls = PyModule::import(py, "hyzero.training.trainer")?
                .getattr("Trainer")?;
            let trainer = trainer_cls.call1(("cpu",))?;
            let num_actions: usize = cfg
                .cast::<pyo3::types::PyDict>()?
                .get_item("num_actions")?
                .ok_or_else(|| {
                    pyo3::exceptions::PyKeyError::new_err("num_actions not in config")
                })?
                .extract()?;
            let batch = pyo3::types::PyDict::new(py);
            let zeros = np.getattr("zeros")?;
            let full = np.getattr("full")?;
            batch.set_item(
                "observations",
                randn
                    .call1((4, 4, input_planes, 8, 8))?
                    .call_method1("astype", ("float32",))?,
            )?;
            batch.set_item(
                "actions",
                randn
                    .call1((4, 3, 3, 8, 8))?
                    .call_method1("astype", ("float32",))?,
            )?;
            batch.set_item(
                "target_policies",
                full.call1(((4, 4, num_actions), 1.0_f64 / num_actions as f64))?
                    .call_method1("astype", ("float32",))?,
            )?;
            batch.set_item(
                "target_values",
                zeros
                    .call1(((4, 4),))?
                    .call_method1("astype", ("float32",))?,
            )?;
            batch.set_item(
                "target_rewards",
                zeros
                    .call1(((4, 4),))?
                    .call_method1("astype", ("float32",))?,
            )?;
            for _ in 0..5 {
                trainer.call_method1("train_batch", (batch.clone(),))?;
            }
            let weight_bytes: Vec<u8> = trainer
                .call_method0("get_weights")?
                .extract()?;

            // Apply weights via the held handle (the exact path used by EvaluationTask).
            {
                let guard = opp_handle.lock().unwrap();
                guard
                    .call_method1(py, "load_weights", (PyBytes::new(py, &weight_bytes),))?;
            }

            // Capture post-load output and verify it differs.
            let post = server
                .call_method1(py, "root_setup_batch", (obs,))?;
            let policies_after = post.bind(py).get_item(1)?.unbind();

            let allclose = np
                .getattr("allclose")?
                .call1((policies_before, policies_after, 1e-6_f64))?
                .extract::<bool>()?;
            assert!(
                !allclose,
                "policies unchanged after load_weights — weights may not have been loaded"
            );
            Ok(())
        });
        result.expect("opponent load_weights test failed");
    }

    /// Validate win_rate sign convention for Black-side games.
    ///
    /// When champion=White wins (game_outcome=+1.0), that's a loss for challenger (Black).
    /// challenger_perspective = -game_outcome = -1.0 → ladder_losses += 1.
    #[test]
    fn test_win_rate_black_side_sign() {
        // Simulate: champion=White wins (game_outcome = +1.0)
        let game_outcome: f32 = 1.0; // White (champion) wins
        let challenger_perspective = -game_outcome;
        // challenger lost: challenger_perspective < -0.5
        assert!(challenger_perspective < -0.5, "challenger lost when champion won as White");

        // Simulate: challenger=Black wins (game_outcome = -1.0)
        let game_outcome: f32 = -1.0; // Black (challenger) wins
        let challenger_perspective = -game_outcome;
        assert!(challenger_perspective > 0.5, "challenger won when Black won");

        // Draw
        let game_outcome: f32 = 0.0;
        let challenger_perspective = -game_outcome;
        assert_eq!(challenger_perspective, 0.0, "draw is neutral");
    }
}
