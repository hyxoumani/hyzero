use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use pyo3::prelude::*;
use tokio::sync::watch;

use std::sync::OnceLock;

use crate::data::{ActionIndex, BoardObservation, HiddenState, Policy, NUM_ACTIONS};
use crate::game::board::GameResult;
use crate::game::fen::board_from_fen;
use crate::game::{GameBoard, Player};
use crate::mcts::evaluator::Evaluator;
use crate::selfplay::champion::ChampionStore;
use crate::selfplay::game_task::{
    pick_starting_position, play_game_dual, play_game_dual_from, DualGameOutcome, GameConfig,
};
use crate::{Color, PrecomputedItems};

// --- Eval-side adjudication (HYZERO_EVAL_ADJUDICATE* gates) ---
//
// Read per-call from the environment (mirroring `material_shaping_enabled` and
// the HYZERO_RESIGN* helpers in game_task.rs) so env-controlled tests can vary
// them within one process; serialize such tests via the module `Mutex`. Eval
// outcomes never enter training targets, so adjudication here is safe and the
// antisymmetry/passivity-attractor risk that bars it from self-play does not apply.

/// Env-gate: true (DEFAULT) unless HYZERO_EVAL_ADJUDICATE is "0"/"false"/"no"/empty.
/// When enabled, eval games (`play_game_dual`) award ±1 at the move cap to the
/// side ahead by at least `eval_adjudication_margin()` material instead of
/// scoring every non-checkmate terminal as a draw.
fn eval_adjudicate_enabled() -> bool {
    match std::env::var("HYZERO_EVAL_ADJUDICATE") {
        Ok(v) => {
            let s = v.trim().to_ascii_lowercase();
            !(s.is_empty() || s == "0" || s == "false" || s == "no")
        }
        Err(_) => true,
    }
}

/// Material lead (white-absolute, standard piece values) required to adjudicate a
/// non-checkmate eval terminal as decisive. `HYZERO_EVAL_ADJ_MARGIN`, default 5
/// (clamped to >= 1).
fn eval_adjudication_margin() -> i32 {
    std::env::var("HYZERO_EVAL_ADJ_MARGIN")
        .ok()
        .and_then(|v| v.parse::<i32>().ok())
        .filter(|&m| m >= 1)
        .unwrap_or(5)
}

// --- Mirrored (antithetic) eval start pairs (HYZERO_EVAL_MIRRORED_STARTS) ---
//
// When enabled, each of the `games_per_side` ladder slots becomes a PAIR: one
// curriculum start is sampled once and played twice with the challenger's color
// swapped, so both games of a pair begin from the identical position. This is an
// antithetic-variates scheme that correlates the start within a pair to reduce
// win_rate/candidate_elo variance without changing the total game count, the
// aggregation, the Elo math, or either promotion gate. DEFAULT OFF: with the gate
// off, the two legacy `play_game_dual` loops are byte-identical to before.

/// Pure parse helper for `HYZERO_EVAL_MIRRORED_STARTS`. Enabled only when the
/// (trimmed, lowercased) value is `"1"` or `"true"`; anything else (including
/// unset/`None`) is OFF. Extracted from the cached gate so env-parse tests can
/// exercise it without tripping the process-wide `OnceLock`.
fn parse_mirrored_starts(value: Option<&str>) -> bool {
    match value {
        Some(v) => {
            let s = v.trim().to_ascii_lowercase();
            s == "1" || s == "true"
        }
        None => false,
    }
}

/// Cached env-gate: true when `HYZERO_EVAL_MIRRORED_STARTS` is `"1"`/`"true"`.
/// Read once per process via `OnceLock`, mirroring the other self-play knobs.
fn eval_mirrored_starts_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        parse_mirrored_starts(std::env::var("HYZERO_EVAL_MIRRORED_STARTS").ok().as_deref())
    })
}

/// Sample ONE curriculum start for a mirrored eval pair, reusing the self-play
/// `pick_starting_position` + `board_from_fen` path. Mirrors
/// `init_self_play_board`: an already-terminal or unparseable sampled FEN falls
/// back to the standard initial position so eval never aborts on a bad start.
/// Sampled exactly once per pair (never per game) so both games of the pair share
/// the identical board.
fn sample_eval_start(precomputed: &Arc<PrecomputedItems>) -> (GameBoard, Color, Option<String>) {
    if let Some(fen) = pick_starting_position() {
        match board_from_fen(fen, precomputed.clone()) {
            Ok((board, side_to_move, _fullmove)) => {
                if board.result() == GameResult::Ongoing {
                    return (board, side_to_move, Some(fen.to_string()));
                }
                eprintln!(
                    "[eval] WARN: sampled start FEN is already terminal; \
                     falling back to standard start"
                );
            }
            Err(e) => {
                eprintln!(
                    "[eval] WARN: failed to parse start FEN {fen:?}: {e}; \
                     falling back to standard start"
                );
            }
        }
    }
    let player1 = Player::init_player(true);
    let player2 = Player::init_player(false);
    let board = GameBoard::init_game_board(precomputed.clone(), player1, player2);
    (board, Color::White, None)
}

/// A mirrored evaluation pair: one shared start played twice with the
/// challenger's color swapped. `board_a`/`board_b` are clones of the SAME sampled
/// position, so both games begin identically; only the White/Black role differs.
/// Game A = challenger White vs. opponent Black; Game B = opponent White vs.
/// challenger Black.
struct MirroredPair {
    white_a: Arc<dyn Evaluator>,
    black_a: Arc<dyn Evaluator>,
    board_a: GameBoard,
    white_b: Arc<dyn Evaluator>,
    black_b: Arc<dyn Evaluator>,
    board_b: GameBoard,
    side_to_move: Color,
    starting_fen: Option<String>,
}

/// Build a mirrored pair from a single sampled start: the challenger takes White
/// in game A and Black in game B; the opponent (bootstrap champion or pool
/// member) takes the complementary color. The board is sampled once and cloned so
/// the two games are true antithetic variates on the start.
fn build_mirrored_pair(
    precomputed: &Arc<PrecomputedItems>,
    challenger: &Arc<dyn Evaluator>,
    opponent: &Arc<dyn Evaluator>,
) -> MirroredPair {
    let (board_a, side_to_move, starting_fen) = sample_eval_start(precomputed);
    let board_b = board_a.clone();
    MirroredPair {
        white_a: challenger.clone(),
        black_a: opponent.clone(),
        board_a,
        white_b: opponent.clone(),
        black_b: challenger.clone(),
        board_b,
        side_to_move,
        starting_fen,
    }
}

/// Evaluator that returns uniform policy and zero value — a pure random baseline.
pub struct RandomEvaluator;

#[async_trait]
impl Evaluator for RandomEvaluator {
    async fn root_setup(
        &self,
        _obs: &BoardObservation,
        _legal_mask: &[bool],
    ) -> (HiddenState, Policy, f32, Option<f32>) {
        let policy = vec![1.0 / NUM_ACTIONS as f32; NUM_ACTIONS];
        (HiddenState::new(64), policy, 0.0, None)
    }

    async fn expand_leaf(
        &self,
        _hs: &HiddenState,
        _action: ActionIndex,
    ) -> (HiddenState, f32, Policy, f32, Option<f32>) {
        let policy = vec![1.0 / NUM_ACTIONS as f32; NUM_ACTIONS];
        (HiddenState::new(64), 0.0, policy, 0.0, None)
    }
}

/// Configuration for the champion-challenger evaluation ladder.
#[derive(Debug, Clone)]
pub struct EvaluationConfig {
    /// Games played per side in each ladder match (total = 2 × games_per_side).
    pub games_per_side: usize,
    /// Win-rate threshold for promotion (0.0–1.0). Default 0.55. Active only on the
    /// empty-pool bootstrap path (no archived champions yet); once at least one
    /// `best_v{NNN}.pt` exists, gating switches to Elo (`promotion_elo_delta`).
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
    /// K-factor used in per-game Elo updates against the pool. Default 32.0.
    pub elo_k_factor: f32,
    /// Maximum number of archived champions used as ladder opponents per cycle.
    /// Default 3.
    pub pool_size: usize,
    /// Promotion gate: candidate is promoted when its post-cycle Elo exceeds
    /// `opponent_initial_elo + promotion_elo_delta`. Default 20.0.
    pub promotion_elo_delta: f32,
    /// Fixed rating assigned to every pool opponent at the start of each cycle.
    /// Default 1500.0.
    pub opponent_initial_elo: f32,
    /// Directory scanned for `best_v{NNN}.pt` archives when building the pool.
    /// Default `checkpoints`.
    pub checkpoints_dir: PathBuf,
    /// File every eval ladder game is appended to in PGN format.
    /// Default `logs/eval_games.pgn`. Overridden in tests to a temp path so they
    /// never write to the shared repo log used by the live training run.
    pub eval_pgn_path: PathBuf,
}

impl Default for EvaluationConfig {
    fn default() -> Self {
        Self {
            games_per_side: 8,
            promotion_threshold: 0.55,
            promotion_cooldown_games: 0,
            num_simulations: 50,
            temperature_moves: 15,
            poll_interval_ms: 500,
            champion_score_weight: 2.0,
            elo_k_factor: crate::selfplay::elo::K_FACTOR,
            pool_size: 3,
            promotion_elo_delta: 20.0,
            opponent_initial_elo: crate::selfplay::elo::INITIAL_RATING,
            checkpoints_dir: PathBuf::from("checkpoints"),
            eval_pgn_path: PathBuf::from("logs/eval_games.pgn"),
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
    /// Sticky error-flag handle of the challenger `ChannelEvaluator`, when it is
    /// one. Set whenever the challenger recovered an inference call to a neutral
    /// result during a cycle's games. Checked (and cleared) after the games so a
    /// cycle whose challenger evals were degraded does NOT reach a promotion
    /// decision. `None` when the challenger is not a `ChannelEvaluator` (e.g. tests
    /// using `RandomEvaluator`), in which case it contributes no degradation.
    challenger_error_flag: Option<Arc<std::sync::atomic::AtomicBool>>,
    /// Sticky error-flag handle of the bootstrap champion `ChannelEvaluator`, when
    /// it is one. Only the empty-pool (bootstrap) path plays games through the
    /// live champion; a dead champion batcher there yields neutral champion evals
    /// that would let the challenger win trivially, so its degradation must also
    /// block the promotion decision. `None` for a `RandomEvaluator` champion.
    champion_error_flag: Option<Arc<std::sync::atomic::AtomicBool>>,
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
            challenger_error_flag: None,
            champion_error_flag: None,
        }
    }

    /// Register the challenger and (optional) bootstrap-champion sticky error-flag
    /// handles. After each cycle's games the loop reads-and-clears these; if either
    /// fired, the cycle is degraded by inference errors and the promotion decision
    /// is skipped (the loop re-arms for the next version). Pass `None` for an
    /// evaluator that is not a `ChannelEvaluator` and so cannot report recoveries.
    pub fn with_error_flags(
        mut self,
        challenger: Option<Arc<std::sync::atomic::AtomicBool>>,
        champion: Option<Arc<std::sync::atomic::AtomicBool>>,
    ) -> Self {
        self.challenger_error_flag = challenger;
        self.champion_error_flag = champion;
        self
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

    /// Write a single game to `pgn_path` (default `logs/eval_games.pgn`) in
    /// standard PGN format.
    fn write_pgn_game(
        pgn_path: &std::path::Path,
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
            &pgn_path.to_string_lossy(),
            &format!("Eval Cycle {cycle} Game {game_num}"),
            white_label,
            black_label,
            result_str,
            &outcome.termination,
            outcome.starting_fen.as_deref(),
            &outcome.moves,
        );
    }

    /// Pure helper: fold per-game scores into a final candidate Elo against a
    /// fixed-rating opponent. Each `score` ∈ {1.0, 0.5, 0.0} = win/draw/loss
    /// from the candidate's perspective. Exposed for unit testing — production
    /// `run()` inlines the update (per-game `candidate_elo` is needed for
    /// log output between updates), so this helper is test-only.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn compute_candidate_elo_from_results(
        initial: f32,
        opp_initial: f32,
        k: f32,
        scores: &[f32],
    ) -> f32 {
        let mut r = initial;
        for s in scores {
            r = crate::selfplay::elo::update_rating(r, opp_initial, *s, k);
        }
        r
    }

    /// Run the evaluation ladder loop.
    ///
    /// On each cycle:
    /// 1. Wait for a new training version.
    /// 2. Enumerate up to `pool_size` archived champions from `checkpoints_dir`.
    /// 3. If pool is nonempty: per opponent, reload weights via the held
    ///    `opponent_server_handle.load_weights(bytes)` then play
    ///    `2 * games_per_side` games against `opponent_evaluator`. Update the
    ///    candidate's Elo per game (opponents pinned at `opponent_initial_elo`).
    ///    Promotion gate: `candidate_elo > opponent_initial_elo + promotion_elo_delta`.
    ///    Otherwise (bootstrap): play 2×gps games against the live
    ///    `champion_store.champion()` and use the legacy `win_rate >=
    ///    promotion_threshold` gate. This is the ONLY path that produces the
    ///    FIRST promotion (transitions to the Elo gate once `best_v001.pt` lands).
    /// 4. Log structured output for run_baseline.sh grep anchors (existing
    ///    fields preserved verbatim; new fields appended before `ladder_match`).
    pub async fn run(&mut self) {
        let mut last_evaluated_version: u64 = 0;

        // Re-arm invariant: the ONLY exit from this loop is the training-done
        // signal (sender dropped, below). Every cycle — including one that
        // promotes a new champion or one whose games degrade because a backing
        // inference batcher stopped — must fall through to the next iteration and
        // wait for the next version. The eval games call through `ChannelEvaluator`,
        // which recovers from a dropped batcher with a neutral result instead of
        // panicking (see `inference::ChannelEvaluator`), so a stopped batcher can no
        // longer silently kill this task mid-cycle and strand the ladder.
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

            let champion_version = self.champion_store.version();
            let pool = crate::selfplay::pool::latest_archive_versions(
                &self.config.checkpoints_dir,
                champion_version,
                self.config.pool_size,
            );

            let game_config = GameConfig {
                num_simulations: self.config.num_simulations,
                exploration_constant: 1.5,
                temperature_moves: self.config.temperature_moves,
                replay_dir: None,
                // Eval-side adjudication is ON by default (HYZERO_EVAL_ADJUDICATE):
                // eval outcomes never enter training targets, so adjudicating a
                // material lead at the move cap discriminates models that would
                // otherwise all draw, without the passivity-attractor risk that
                // bars adjudication from self-play.
                adjudicate_at_cap: eval_adjudicate_enabled(),
                adjudication_material_margin: eval_adjudication_margin(),
            };

            let gps = self.config.games_per_side;
            let mut ladder_wins: usize = 0;
            let mut ladder_draws: usize = 0;
            let mut ladder_losses: usize = 0;
            let mut candidate_elo = self.config.opponent_initial_elo;
            let opp_initial = self.config.opponent_initial_elo;
            let k = self.config.elo_k_factor;
            let mut scored_games: Vec<f32> = Vec::new();
            let mut opponents_label = String::from("none");

            if pool.is_empty() {
                // Bootstrap path: legacy single-opponent (live champion) ladder
                // with `win_rate` gating. Only path that can fire the FIRST
                // promotion. Transition to the Elo gate happens once
                // `best_v{NNN}.pt` exists. When `champion_version > 0` but
                // pool is empty (unexpected: archives were deleted), emit a
                // WARN — still safe to run.
                if champion_version > 0 {
                    eprintln!(
                        "[eval] WARN: pool empty despite champion_version={champion_version} > 0; using win-rate fallback"
                    );
                }
                let champion_eval = self.champion_store.champion().await;

                if eval_mirrored_starts_enabled() {
                    // Mirrored pairs: sample one start per pair and play it twice
                    // with the challenger's color swapped. Total games unchanged
                    // (2*gps); only the start is correlated within a pair.
                    for pair in 0..gps {
                        let mp = build_mirrored_pair(
                            &self.precomputed,
                            &self.challenger_evaluator,
                            &champion_eval,
                        );

                        // Game A: challenger White, champion Black.
                        let outcome_a = play_game_dual_from(
                            mp.white_a,
                            mp.black_a,
                            game_config.clone(),
                            mp.board_a,
                            mp.side_to_move,
                            mp.starting_fen.clone(),
                        )
                        .await;

                        Self::write_pgn_game(
                            &self.config.eval_pgn_path,
                            self.cycle,
                            pair * 2 + 1,
                            &format!("challenger v{challenger_version}"),
                            &format!("champion v{champion_version}"),
                            &outcome_a,
                        );

                        match outcome_a.game_outcome {
                            o if o > 0.5 => ladder_wins += 1,
                            o if o < -0.5 => ladder_losses += 1,
                            _ => ladder_draws += 1,
                        }
                        self.total_games_since_last_promotion += 1;

                        // Game B: champion White, challenger Black (same start).
                        let outcome_b = play_game_dual_from(
                            mp.white_b,
                            mp.black_b,
                            game_config.clone(),
                            mp.board_b,
                            mp.side_to_move,
                            mp.starting_fen,
                        )
                        .await;

                        Self::write_pgn_game(
                            &self.config.eval_pgn_path,
                            self.cycle,
                            pair * 2 + 2,
                            &format!("champion v{champion_version}"),
                            &format!("challenger v{challenger_version}"),
                            &outcome_b,
                        );

                        let challenger_perspective = -outcome_b.game_outcome;
                        match challenger_perspective {
                            o if o > 0.5 => ladder_wins += 1,
                            o if o < -0.5 => ladder_losses += 1,
                            _ => ladder_draws += 1,
                        }
                        self.total_games_since_last_promotion += 1;
                    }
                } else {
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
                            &self.config.eval_pgn_path,
                            self.cycle,
                            game_idx + 1,
                            &format!("challenger v{challenger_version}"),
                            &format!("champion v{champion_version}"),
                            &outcome,
                        );

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
                            &self.config.eval_pgn_path,
                            self.cycle,
                            gps + game_idx + 1,
                            &format!("champion v{champion_version}"),
                            &format!("challenger v{challenger_version}"),
                            &outcome,
                        );

                        let challenger_perspective = -outcome.game_outcome;
                        match challenger_perspective {
                            o if o > 0.5 => ladder_wins += 1,
                            o if o < -0.5 => ladder_losses += 1,
                            _ => ladder_draws += 1,
                        }
                        self.total_games_since_last_promotion += 1;
                    }
                }
            } else {
                // Elo-gate path: per-opponent ladder against archived champions.
                // The opponent evaluator + server handle MUST be set; otherwise
                // we cannot reload weights, so we fall back to the bootstrap
                // log and skip the ladder.
                let (opp_eval, opp_handle) = match (
                    self.opponent_evaluator.clone(),
                    self.opponent_server_handle.clone(),
                ) {
                    (Some(e), Some(h)) => (e, h),
                    _ => {
                        eprintln!(
                            "[eval] WARN: pool nonempty (size={}) but opponent evaluator/server handle unset; skipping ladder",
                            pool.len()
                        );
                        // Build opponents= label for the log line, no games played.
                        let labels: Vec<String> =
                            pool.iter().map(|(v, _)| format!("v{v}")).collect();
                        opponents_label = labels.join(",");
                        let total_games = 0usize;
                        let win_rate = 0.0_f64;
                        let pool_score = 0.0_f64;
                        println!(
                            "[eval] v{challenger_version} cycle={cycle} ladder_wins={w} ladder_draws={d} \
                             ladder_losses={l} win_rate={r:.3} champion_version={cv} \
                             candidate_elo={elo:.1} pool_size={ps} opponents={opps} \
                             pool_score={ps_score:.3} ladder_match",
                            cycle = self.cycle,
                            w = ladder_wins,
                            d = ladder_draws,
                            l = ladder_losses,
                            r = win_rate,
                            cv = champion_version,
                            elo = candidate_elo,
                            ps = pool.len(),
                            opps = opponents_label,
                            ps_score = pool_score,
                        );
                        let _ = total_games;
                        continue;
                    }
                };

                let labels: Vec<String> = pool.iter().map(|(v, _)| format!("v{v}")).collect();
                opponents_label = labels.join(",");

                'pool_loop: for (opponent_version, ckpt_path) in pool.iter() {
                    // Read checkpoint bytes; skip on read error.
                    let bytes = match std::fs::read(ckpt_path) {
                        Ok(b) => b,
                        Err(e) => {
                            eprintln!(
                                "[eval] WARN: failed to read pool member v{opponent_version}: {e}"
                            );
                            continue 'pool_loop;
                        }
                    };

                    // Swap weights via the held Py<PyAny>. On error, skip this opponent.
                    let load_res: pyo3::PyResult<()> = Python::attach(|py| {
                        let guard = opp_handle.lock().unwrap();
                        guard.call_method1(
                            py,
                            "load_weights",
                            (pyo3::types::PyBytes::new(py, &bytes),),
                        )?;
                        Ok(())
                    });
                    if let Err(e) = load_res {
                        eprintln!(
                            "[eval] WARN: load_weights failed for pool member v{opponent_version}: {e}"
                        );
                        continue 'pool_loop;
                    }

                    if eval_mirrored_starts_enabled() {
                        // Mirrored pairs against this pool member: one shared start
                        // per pair, played both colors. Game count, Elo updates, and
                        // scored_games accounting are unchanged (2*gps per opponent).
                        for pair in 0..gps {
                            let mp = build_mirrored_pair(
                                &self.precomputed,
                                &self.challenger_evaluator,
                                &opp_eval,
                            );

                            // Game A: challenger White vs. this opponent.
                            let outcome_a = play_game_dual_from(
                                mp.white_a,
                                mp.black_a,
                                game_config.clone(),
                                mp.board_a,
                                mp.side_to_move,
                                mp.starting_fen.clone(),
                            )
                            .await;

                            Self::write_pgn_game(
                                &self.config.eval_pgn_path,
                                self.cycle,
                                pair * 2 + 1,
                                &format!("challenger v{challenger_version}"),
                                &format!("pool v{opponent_version}"),
                                &outcome_a,
                            );

                            let challenger_score: f32 = if outcome_a.game_outcome > 0.5 {
                                ladder_wins += 1;
                                1.0
                            } else if outcome_a.game_outcome < -0.5 {
                                ladder_losses += 1;
                                0.0
                            } else {
                                ladder_draws += 1;
                                0.5
                            };
                            candidate_elo = crate::selfplay::elo::update_rating(
                                candidate_elo,
                                opp_initial,
                                challenger_score,
                                k,
                            );
                            scored_games.push(challenger_score);
                            self.total_games_since_last_promotion += 1;

                            // Game B: opponent White vs. challenger Black (same start).
                            let outcome_b = play_game_dual_from(
                                mp.white_b,
                                mp.black_b,
                                game_config.clone(),
                                mp.board_b,
                                mp.side_to_move,
                                mp.starting_fen,
                            )
                            .await;

                            Self::write_pgn_game(
                                &self.config.eval_pgn_path,
                                self.cycle,
                                pair * 2 + 2,
                                &format!("pool v{opponent_version}"),
                                &format!("challenger v{challenger_version}"),
                                &outcome_b,
                            );

                            let challenger_perspective = -outcome_b.game_outcome;
                            let challenger_score: f32 = if challenger_perspective > 0.5 {
                                ladder_wins += 1;
                                1.0
                            } else if challenger_perspective < -0.5 {
                                ladder_losses += 1;
                                0.0
                            } else {
                                ladder_draws += 1;
                                0.5
                            };
                            candidate_elo = crate::selfplay::elo::update_rating(
                                candidate_elo,
                                opp_initial,
                                challenger_score,
                                k,
                            );
                            scored_games.push(challenger_score);
                            self.total_games_since_last_promotion += 1;
                        }
                    } else {
                        // games_per_side games challenger=White vs. this opponent.
                        for game_idx in 0..gps {
                            let outcome = play_game_dual(
                                self.precomputed.clone(),
                                self.challenger_evaluator.clone(),
                                opp_eval.clone(),
                                game_config.clone(),
                            )
                            .await;

                            Self::write_pgn_game(
                                &self.config.eval_pgn_path,
                                self.cycle,
                                game_idx + 1,
                                &format!("challenger v{challenger_version}"),
                                &format!("pool v{opponent_version}"),
                                &outcome,
                            );

                            let challenger_score: f32 = if outcome.game_outcome > 0.5 {
                                ladder_wins += 1;
                                1.0
                            } else if outcome.game_outcome < -0.5 {
                                ladder_losses += 1;
                                0.0
                            } else {
                                ladder_draws += 1;
                                0.5
                            };
                            candidate_elo = crate::selfplay::elo::update_rating(
                                candidate_elo,
                                opp_initial,
                                challenger_score,
                                k,
                            );
                            scored_games.push(challenger_score);
                            self.total_games_since_last_promotion += 1;
                        }

                        // games_per_side games opponent=White vs. challenger=Black.
                        for game_idx in 0..gps {
                            let outcome = play_game_dual(
                                self.precomputed.clone(),
                                opp_eval.clone(),
                                self.challenger_evaluator.clone(),
                                game_config.clone(),
                            )
                            .await;

                            Self::write_pgn_game(
                                &self.config.eval_pgn_path,
                                self.cycle,
                                gps + game_idx + 1,
                                &format!("pool v{opponent_version}"),
                                &format!("challenger v{challenger_version}"),
                                &outcome,
                            );

                            let challenger_perspective = -outcome.game_outcome;
                            let challenger_score: f32 = if challenger_perspective > 0.5 {
                                ladder_wins += 1;
                                1.0
                            } else if challenger_perspective < -0.5 {
                                ladder_losses += 1;
                                0.0
                            } else {
                                ladder_draws += 1;
                                0.5
                            };
                            candidate_elo = crate::selfplay::elo::update_rating(
                                candidate_elo,
                                opp_initial,
                                challenger_score,
                                k,
                            );
                            scored_games.push(challenger_score);
                            self.total_games_since_last_promotion += 1;
                        }
                    }
                }
            }

            let total_games = if pool.is_empty() {
                2 * gps
            } else {
                scored_games.len()
            };
            // `win_rate` keeps its existing semantics (win_rate = pool_score for
            // the pool path); preserved under the legacy field name so
            // run_baseline.sh extractors keep working.
            let win_rate = if total_games > 0 {
                (ladder_wins as f64 + ladder_draws as f64 * 0.5) / total_games as f64
            } else {
                0.0
            };
            let pool_score = win_rate;

            println!(
                "[eval] v{challenger_version} cycle={cycle} ladder_wins={w} ladder_draws={d} \
                 ladder_losses={l} win_rate={r:.3} champion_version={cv} \
                 candidate_elo={elo:.1} pool_size={ps} opponents={opps} \
                 pool_score={ps_score:.3} ladder_match",
                cycle = self.cycle,
                w = ladder_wins,
                d = ladder_draws,
                l = ladder_losses,
                r = win_rate,
                cv = champion_version,
                elo = candidate_elo,
                ps = pool.len(),
                opps = opponents_label,
                ps_score = pool_score,
            );

            // Error-degradation gate (re-arm without promoting). If any eval
            // evaluator recovered an inference call to a neutral result during
            // this cycle's games — a batcher genuinely died mid-cycle (OOM /
            // backend panic; the champion keepalive makes this abnormal, not
            // impossible) — the games no longer reflect the real models: champion
            // evals collapse to neutral, the challenger wins trivially, and the
            // gate would record a garbage promotion. Such a cycle must NOT reach a
            // promotion decision. Read-and-clear BOTH flags (no short-circuit) so a
            // single degraded cycle does not poison later ones, then skip to the
            // next loop iteration (re-arm). Self-play games keep the existing
            // neutral-degradation behavior — their output feeds replay, not this
            // champion gate — so only the eval evaluators are inspected here.
            let challenger_degraded = self
                .challenger_error_flag
                .as_ref()
                .map(|f| f.swap(false, std::sync::atomic::Ordering::AcqRel))
                .unwrap_or(false);
            let champion_degraded = self
                .champion_error_flag
                .as_ref()
                .map(|f| f.swap(false, std::sync::atomic::Ordering::AcqRel))
                .unwrap_or(false);
            if challenger_degraded || champion_degraded {
                eprintln!(
                    "[eval] WARN: cycle degraded by inference errors, skipping promotion decision"
                );
                continue;
            }

            // Promotion gate: bootstrap (empty-pool) uses legacy win-rate; pool
            // path uses Elo. The bootstrap branch is single-shot — once any
            // archive lands, all subsequent cycles route through the Elo gate.
            let cooldown_ok = self.total_games_since_last_promotion
                >= self.config.promotion_cooldown_games
                || self.config.promotion_cooldown_games == 0;

            let promote = if pool.is_empty() {
                win_rate >= self.config.promotion_threshold
            } else {
                candidate_elo > self.config.opponent_initial_elo + self.config.promotion_elo_delta
            };

            if promote && cooldown_ok {
                let ckpt_path = self
                    .latest_checkpoint_path
                    .lock()
                    .ok()
                    .and_then(|g| g.clone());

                let new_champ = self.challenger_evaluator.clone();
                self.champion_store
                    .promote(new_champ, challenger_version, ckpt_path.as_ref())
                    .await;

                self.total_games_since_last_promotion = 0;

                println!(
                    "[eval] promoted champion_version={cv} challenger_version={cv_train} win_rate={r:.3} candidate_elo={elo:.1}",
                    cv = challenger_version,
                    cv_train = challenger_version,
                    r = win_rate,
                    elo = candidate_elo,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::types::TestEnvGuard;
    use tokio::sync::watch;

    /// Per-test isolation for `run()`-driving tests: an empty temp dir for the
    /// champion pool and a temp PGN path, so a test never reads the real
    /// `checkpoints/` archive (which the live training run populates) nor writes
    /// to the shared `logs/eval_games.pgn`. The returned `TempDir` must be held
    /// for the duration of the test — dropping it deletes the directory.
    fn isolated_paths() -> (tempfile::TempDir, PathBuf, PathBuf) {
        let dir = tempfile::tempdir().expect("tempdir");
        let checkpoints_dir = dir.path().join("checkpoints");
        let eval_pgn_path = dir.path().join("eval_games.pgn");
        (dir, checkpoints_dir, eval_pgn_path)
    }

    /// Eval-side adjudication is wired into the `GameConfig` that drives
    /// `play_game_dual`: with `HYZERO_EVAL_ADJUDICATE` set truthy, the config
    /// the ladder builds carries `adjudicate_at_cap == true`; with it OFF, the
    /// config falls back to a pure-draw cap. FAILS without the env wiring in
    /// `run()` (the S2 stub hard-coded `adjudicate_at_cap: false`).
    #[test]
    fn eval_game_config_enables_adjudication_when_env_set() {
        let _env = TestEnvGuard::new(&["HYZERO_EVAL_ADJUDICATE", "HYZERO_EVAL_ADJ_MARGIN"]);
        std::env::set_var("HYZERO_EVAL_ADJUDICATE", "1");
        std::env::set_var("HYZERO_EVAL_ADJ_MARGIN", "7");
        // Mirror the exact construction `run()` uses for the eval GameConfig.
        let game_config = GameConfig {
            num_simulations: 1,
            exploration_constant: 1.5,
            temperature_moves: 1,
            replay_dir: None,
            adjudicate_at_cap: eval_adjudicate_enabled(),
            adjudication_material_margin: eval_adjudication_margin(),
        };
        assert!(game_config.adjudicate_at_cap);
        assert_eq!(game_config.adjudication_material_margin, 7);

        std::env::set_var("HYZERO_EVAL_ADJUDICATE", "0");
        assert!(!eval_adjudicate_enabled());
    }

    /// Default (env unset) keeps eval adjudication ON and the margin at 5.
    #[test]
    fn eval_adjudication_defaults_on_with_margin_five() {
        let _env = TestEnvGuard::new(&["HYZERO_EVAL_ADJUDICATE", "HYZERO_EVAL_ADJ_MARGIN"]);
        std::env::remove_var("HYZERO_EVAL_ADJUDICATE");
        std::env::remove_var("HYZERO_EVAL_ADJ_MARGIN");
        assert!(eval_adjudicate_enabled());
        assert_eq!(eval_adjudication_margin(), 5);
    }

    /// Env-parse (pure helper): `HYZERO_EVAL_MIRRORED_STARTS` is OFF by default
    /// and turns ON only for "1"/"true" (space/case-insensitive). Any other value
    /// — including "0", "false", empty, and unset — stays OFF.
    #[test]
    fn mirrored_starts_env_parse_default_off_truthy_on() {
        assert!(!parse_mirrored_starts(None));
        assert!(!parse_mirrored_starts(Some("")));
        assert!(!parse_mirrored_starts(Some("0")));
        assert!(!parse_mirrored_starts(Some("false")));
        assert!(!parse_mirrored_starts(Some("no")));
        assert!(parse_mirrored_starts(Some("1")));
        assert!(parse_mirrored_starts(Some("true")));
        assert!(parse_mirrored_starts(Some("  TRUE  ")));
    }

    /// With the env unset the cached gate reports OFF, so `run()` takes the legacy
    /// per-game `play_game_dual` sampling branch rather than the mirrored-pair
    /// path. Serialized via `TestEnvGuard` so the ambient env is deterministically
    /// unset when the process-wide gate first reads it.
    #[test]
    fn mirrored_starts_disabled_preserves_per_game_sampling() {
        let _env = TestEnvGuard::new(&["HYZERO_EVAL_MIRRORED_STARTS"]);
        std::env::remove_var("HYZERO_EVAL_MIRRORED_STARTS");
        assert!(
            !eval_mirrored_starts_enabled(),
            "gate must default OFF so the legacy per-game sampling path runs"
        );
    }

    /// The mirrored-pair builder samples ONE start and hands both games the
    /// identical board (same Zobrist) and starting FEN, with the challenger's
    /// color swapped between game A (White) and game B (Black).
    #[test]
    fn mirrored_pair_games_share_identical_start() {
        let precomputed = Arc::new(crate::PrecomputedItems::begin_precomputing());
        let challenger: Arc<dyn Evaluator> = Arc::new(RandomEvaluator);
        let opponent: Arc<dyn Evaluator> = Arc::new(RandomEvaluator);

        let mp = build_mirrored_pair(&precomputed, &challenger, &opponent);

        // Both games of the pair begin from the identical position.
        assert_eq!(
            mp.board_a.zobrist_hash, mp.board_b.zobrist_hash,
            "pair games must share the identical start board"
        );
        assert_eq!(
            mp.starting_fen, None,
            "default (no starts file) yields the standard start for both games"
        );

        // Colors swapped: challenger is White in game A and Black in game B; the
        // opponent takes the complementary color in each.
        assert!(Arc::ptr_eq(&mp.white_a, &challenger));
        assert!(Arc::ptr_eq(&mp.black_a, &opponent));
        assert!(Arc::ptr_eq(&mp.white_b, &opponent));
        assert!(Arc::ptr_eq(&mp.black_b, &challenger));
    }

    #[tokio::test]
    async fn default_games_per_side_is_eight() {
        let config = EvaluationConfig::default();
        assert_eq!(config.games_per_side, 8);
        assert!((config.promotion_threshold - 0.55).abs() < f64::EPSILON);
        assert_eq!(config.num_simulations, 50);
    }

    /// Regression guard: extended defaults from the Elo refactor + verifies the
    /// preserved `champion_score_weight` field still defaults to 2.0.
    #[test]
    fn evaluation_config_defaults_have_elo_fields() {
        let config = EvaluationConfig::default();
        assert!((config.elo_k_factor - 32.0).abs() < f32::EPSILON);
        assert_eq!(config.pool_size, 3);
        assert!((config.promotion_elo_delta - 20.0).abs() < f32::EPSILON);
        assert!((config.opponent_initial_elo - 1500.0).abs() < f32::EPSILON);
        assert_eq!(config.checkpoints_dir, PathBuf::from("checkpoints"));
        assert_eq!(config.eval_pgn_path, PathBuf::from("logs/eval_games.pgn"));
        // Preserved field — MUST remain 2.0 (existing tests at lines 320, 374
        // construct EvaluationConfig literals that set this).
        assert!((config.champion_score_weight - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn compute_candidate_elo_empty_scores_returns_initial() {
        let r = EvaluationTask::compute_candidate_elo_from_results(1500.0, 1500.0, 32.0, &[]);
        assert!((r - 1500.0).abs() < f32::EPSILON);
    }

    #[test]
    fn compute_candidate_elo_all_wins_against_equal() {
        // 8 wins vs. 1500 with K=32 starting at 1500. Must clear the
        // promotion threshold (default 1520) so the gate is reachable in a sweep.
        let scores = [1.0_f32; 8];
        let r = EvaluationTask::compute_candidate_elo_from_results(1500.0, 1500.0, 32.0, &scores);
        assert!(r > 1520.0, "expected r > 1520, got {r}");
    }

    #[test]
    fn compute_candidate_elo_50_percent_against_equal_is_noop() {
        // Alternating [W, L, W, L] vs. fixed 1500 with K=32: ends near 1498.6
        // (delta ≈ -1.41 from start). Not exactly 1500 due to compounding
        // asymmetry — after a W the candidate is rated higher, so the next L
        // costs slightly MORE than the symmetric −16, and after a L the
        // candidate is rated lower, so the next W earns slightly LESS than
        // the symmetric +16. The plan reviewer flagged this exact tolerance:
        // "|final - 1500| < 1.0 accounts for compounding asymmetry — add a
        // comment". The actual asymmetry for 4 games with K=32 is ~1.41, so
        // we use a 2.0 tolerance (covers the genuine asymmetry while still
        // failing if the helper inverts a sign or skips an update).
        let scores = [1.0_f32, 0.0, 1.0, 0.0];
        let r = EvaluationTask::compute_candidate_elo_from_results(1500.0, 1500.0, 32.0, &scores);
        assert!(
            (r - 1500.0).abs() < 2.0,
            "expected |r - 1500| < 2 (compounding asymmetry), got r={r}"
        );
    }

    #[test]
    fn compute_candidate_elo_all_losses_against_equal() {
        let scores = [0.0_f32; 8];
        let r = EvaluationTask::compute_candidate_elo_from_results(1500.0, 1500.0, 32.0, &scores);
        assert!(r < 1480.0, "expected r < 1480, got {r}");
    }

    /// Bootstrap path: with empty pool and `champion_version == 0`, the legacy
    /// `win_rate >= promotion_threshold` gate fires. The test drives the
    /// evaluation task with `RandomEvaluator` opponents and asserts the
    /// champion store version is bumped (promotion fired) when threshold=0.0
    /// (always promote). Conversely, with threshold=2.0 (impossible), no
    /// promotion fires. This exercises the bootstrap branch end-to-end.
    #[tokio::test]
    async fn bootstrap_path_uses_win_rate_gate() {
        // Case 1: threshold=0.0 → always promote on the bootstrap branch.
        let precomputed = Arc::new(crate::PrecomputedItems::begin_precomputing());
        let challenger: Arc<dyn Evaluator> = Arc::new(RandomEvaluator);
        let champion_eval: Arc<dyn Evaluator> = Arc::new(RandomEvaluator);

        let (version_tx, version_rx) = watch::channel(0u64);
        let ckpt_path: Arc<Mutex<Option<PathBuf>>> = Arc::new(Mutex::new(None));
        let champion_store = Arc::new(ChampionStore::new(champion_eval, 5));
        let store_ref = champion_store.clone();

        // Use a non-existent checkpoints dir so the pool is always empty
        // (forces the bootstrap branch). Redirect PGN writes to a temp file so
        // this test never appends to the shared `logs/eval_games.pgn`.
        let (_tmp, _checkpoints_dir, eval_pgn_path) = isolated_paths();
        let config = EvaluationConfig {
            games_per_side: 1,
            promotion_threshold: 0.0, // Always promote on the bootstrap path.
            promotion_cooldown_games: 0,
            num_simulations: 2,
            temperature_moves: 2,
            poll_interval_ms: 10,
            champion_score_weight: 2.0,
            checkpoints_dir: PathBuf::from("/nonexistent/test/dir/abc"),
            eval_pgn_path,
            ..EvaluationConfig::default()
        };

        // champion_store.version() == 0 (no promote yet) — true bootstrap state.
        assert_eq!(store_ref.version(), 0);

        let mut task = EvaluationTask::new(
            precomputed,
            challenger,
            version_rx,
            ckpt_path,
            champion_store,
            config,
        );

        version_tx.send(7).expect("send failed");
        let task_handle = tokio::spawn(async move {
            task.run().await;
        });
        tokio::time::sleep(std::time::Duration::from_secs(30)).await;
        drop(version_tx);
        let _ = tokio::time::timeout(std::time::Duration::from_secs(30), task_handle).await;

        assert_eq!(
            store_ref.version(),
            7,
            "bootstrap path with threshold=0.0 must promote"
        );
    }

    /// Bootstrap path: with `promotion_threshold` above 1.0, no promotion fires.
    #[tokio::test]
    async fn bootstrap_path_blocks_when_threshold_unreachable() {
        let precomputed = Arc::new(crate::PrecomputedItems::begin_precomputing());
        let challenger: Arc<dyn Evaluator> = Arc::new(RandomEvaluator);
        let champion_eval: Arc<dyn Evaluator> = Arc::new(RandomEvaluator);

        let (version_tx, version_rx) = watch::channel(0u64);
        let ckpt_path: Arc<Mutex<Option<PathBuf>>> = Arc::new(Mutex::new(None));
        let champion_store = Arc::new(ChampionStore::new(champion_eval, 5));
        let store_ref = champion_store.clone();

        // Redirect PGN writes to a temp file so this test never appends to the
        // shared `logs/eval_games.pgn`.
        let (_tmp, _checkpoints_dir, eval_pgn_path) = isolated_paths();
        let config = EvaluationConfig {
            games_per_side: 1,
            promotion_threshold: 2.0, // Impossible — never promote.
            promotion_cooldown_games: 0,
            num_simulations: 2,
            temperature_moves: 2,
            poll_interval_ms: 10,
            champion_score_weight: 2.0,
            checkpoints_dir: PathBuf::from("/nonexistent/test/dir/xyz"),
            eval_pgn_path,
            ..EvaluationConfig::default()
        };

        let mut task = EvaluationTask::new(
            precomputed,
            challenger,
            version_rx,
            ckpt_path,
            champion_store,
            config,
        );

        version_tx.send(11).expect("send failed");
        let task_handle = tokio::spawn(async move {
            task.run().await;
        });
        tokio::time::sleep(std::time::Duration::from_secs(30)).await;
        drop(version_tx);
        let _ = tokio::time::timeout(std::time::Duration::from_secs(30), task_handle).await;

        assert_eq!(
            store_ref.version(),
            0,
            "bootstrap path with unreachable threshold must NOT promote"
        );
    }

    /// Integration test (helper-based form): runs the sequential Elo math
    /// helper against a canned outcome sequence representing 8 wins against
    /// three equal-rated opponents, and verifies it crosses the default
    /// promotion threshold (1500 + 20 = 1520). Helper-based form runs
    /// unconditionally; the full per-opponent ladder path requires PyO3
    /// opponent setup which is covered by the `#[ignore]`-gated test.
    #[test]
    fn eval_task_runs_per_opponent_ladder_helper_form() {
        // 3 opponents × 2 gps = 6 games; assume challenger sweeps each.
        let scores = vec![1.0_f32; 6];
        let final_elo =
            EvaluationTask::compute_candidate_elo_from_results(1500.0, 1500.0, 32.0, &scores);
        assert!(
            final_elo > 1520.0,
            "expected final_elo > 1520 after a clean sweep, got {final_elo}"
        );
    }

    /// Log-format regression: with empty pool, the `opponents=` token reads
    /// `none` and `pool_size=0` is present. This is a string-construction
    /// shape check via the helper used in `run()` (we replicate the same
    /// join logic the production path uses).
    #[test]
    fn eval_log_format_with_empty_pool() {
        let pool: Vec<(u64, PathBuf)> = Vec::new();
        let labels: Vec<String> = pool.iter().map(|(v, _)| format!("v{v}")).collect();
        let opponents_label = if labels.is_empty() {
            String::from("none")
        } else {
            labels.join(",")
        };
        let line = format!(
            "[eval] v1 cycle=1 ladder_wins=0 ladder_draws=0 ladder_losses=0 \
             win_rate=0.000 champion_version=0 candidate_elo=1500.0 pool_size={} \
             opponents={} pool_score=0.000 ladder_match",
            pool.len(),
            opponents_label,
        );
        assert!(line.contains("pool_size=0"));
        assert!(line.contains("opponents=none"));
        assert!(line.ends_with("ladder_match"));
    }

    #[tokio::test]
    async fn test_evaluation_task_completes_one_cycle() {
        let precomputed = Arc::new(crate::PrecomputedItems::begin_precomputing());
        let challenger: Arc<dyn Evaluator> = Arc::new(RandomEvaluator);
        let champion_eval: Arc<dyn Evaluator> = Arc::new(RandomEvaluator);

        let (version_tx, version_rx) = watch::channel(0u64);

        let ckpt_path: Arc<Mutex<Option<PathBuf>>> = Arc::new(Mutex::new(None));
        let champion_store = Arc::new(ChampionStore::new(champion_eval, 5));

        // Isolate pool reads + PGN writes from the shared repo state.
        let (_tmp, checkpoints_dir, eval_pgn_path) = isolated_paths();
        let config = EvaluationConfig {
            games_per_side: 1,
            promotion_threshold: 2.0, // Force no promotion in this test
            promotion_cooldown_games: 0,
            num_simulations: 2,
            temperature_moves: 2,
            poll_interval_ms: 10,
            champion_score_weight: 2.0,
            checkpoints_dir,
            eval_pgn_path,
            ..EvaluationConfig::default()
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

        let result = tokio::time::timeout(std::time::Duration::from_secs(30), task_handle).await;

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

        // Isolate the pool from the shared `checkpoints/` archive so this test
        // always takes the empty-pool bootstrap (win-rate) promotion path even
        // when the live training run has populated real `best_v{NNN}.pt` files.
        // Without this, a nonempty real pool routes through the Elo ladder, which
        // has no opponent handle in a unit test, so it skips and never promotes.
        let (_tmp, checkpoints_dir, eval_pgn_path) = isolated_paths();
        let config = EvaluationConfig {
            games_per_side: 1,
            promotion_threshold: 0.0, // Always promote
            promotion_cooldown_games: 0,
            num_simulations: 2,
            temperature_moves: 2,
            poll_interval_ms: 10,
            champion_score_weight: 2.0,
            checkpoints_dir,
            eval_pgn_path,
            ..EvaluationConfig::default()
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

        let result = tokio::time::timeout(std::time::Duration::from_secs(30), task_handle).await;

        assert!(result.is_ok());
        // With threshold=0.0, champion_version should have been updated to 5.
        assert_eq!(
            store_ref.version(),
            5,
            "champion_version should be 5 after forced promotion"
        );
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
            let srv_cls =
                PyModule::import(py, "hyzero.inference.server")?.getattr("InferenceServer")?;
            let server: Py<PyAny> = srv_cls.call1((cfg.clone(), "cpu"))?.unbind();

            // Hold a directly cloned handle, as EvaluationTask does.
            let opp_handle: Arc<Mutex<Py<PyAny>>> = Arc::new(Mutex::new(server.clone_ref(py)));

            // Build a numpy obs batch of shape [2, INPUT_PLANES, 8, 8]; pull
            // INPUT_PLANES from the config to keep this independent of constants.
            let np = PyModule::import(py, "numpy")?;
            let input_planes: usize = cfg
                .cast::<pyo3::types::PyDict>()?
                .get_item("input_planes")?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("input_planes not in config"))?
                .extract()?;
            let randn = np.getattr("random")?.getattr("randn")?;
            let obs_f64 = randn.call1((2, input_planes, 8, 8))?;
            let obs = obs_f64.call_method1("astype", ("float32",))?;

            // Capture pre-load output (policies tensor index 1).
            let pre = server.call_method1(py, "root_setup_batch", (obs.clone(),))?;
            let policies_before = pre.bind(py).get_item(1)?.unbind();

            // Drive a Trainer for a few steps to diverge from init weights.
            let trainer_cls =
                PyModule::import(py, "hyzero.training.trainer")?.getattr("Trainer")?;
            let trainer = trainer_cls.call1(("cpu",))?;
            let num_actions: usize = cfg
                .cast::<pyo3::types::PyDict>()?
                .get_item("num_actions")?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("num_actions not in config"))?
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
            let weight_bytes: Vec<u8> = trainer.call_method0("get_weights")?.extract()?;

            // Apply weights via the held handle (the exact path used by EvaluationTask).
            {
                let guard = opp_handle.lock().unwrap();
                guard.call_method1(py, "load_weights", (PyBytes::new(py, &weight_bytes),))?;
            }

            // Capture post-load output and verify it differs.
            let post = server.call_method1(py, "root_setup_batch", (obs,))?;
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
        assert!(
            challenger_perspective < -0.5,
            "challenger lost when champion won as White"
        );

        // Simulate: challenger=Black wins (game_outcome = -1.0)
        let game_outcome: f32 = -1.0; // Black (challenger) wins
        let challenger_perspective = -game_outcome;
        assert!(
            challenger_perspective > 0.5,
            "challenger won when Black won"
        );

        // Draw
        let game_outcome: f32 = 0.0;
        let challenger_perspective = -game_outcome;
        assert_eq!(challenger_perspective, 0.0, "draw is neutral");
    }

    /// Backend that drops every reply oneshot without answering and counts the
    /// batches it dropped, so the test can wait until the champion eval has
    /// actually been exercised. Every dropped reply makes the champion
    /// `ChannelEvaluator` recover to a neutral result and set its sticky error
    /// flag — the "batcher died mid-cycle" condition the gate must catch.
    struct CountingDroppingBackend {
        dropped: Arc<std::sync::atomic::AtomicUsize>,
    }
    impl crate::selfplay::inference::InferenceBackend for CountingDroppingBackend {
        fn evaluate_batch(
            &mut self,
            requests: Vec<crate::selfplay::inference::InferenceRequest>,
        ) {
            self.dropped
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            drop(requests); // drop each reply oneshot → callers recover to neutral
        }
    }

    /// REGRESSION (promotion-gate error-blindness + loop re-arm): the promotion
    /// gate must be error-AWARE. A cycle whose champion eval degraded to neutral
    /// results — a batcher genuinely died mid-cycle — must NOT reach a promotion
    /// decision (else the challenger trivially "wins" against a neutral champion
    /// and a garbage promotion is recorded), and the loop must still re-arm so a
    /// subsequent healthy version is evaluated normally.
    ///
    /// Contract encoded here:
    ///   - Cycle 1 runs the bootstrap (empty-pool) path with a champion whose
    ///     batcher drops every reply. The champion recovers to neutral and sets its
    ///     sticky error flag → the gate logs `cycle degraded by inference errors`
    ///     and skips promotion. ASSERT the store version stays 0 (NO promotion)
    ///     even though the cycle completed (the dropping backend confirms it ran).
    ///   - Cycle 2 swaps the champion backend to a healthy one and feeds version 2.
    ///     The cycle is clean → the challenger promotes. ASSERT the store version
    ///     becomes 2, proving the loop re-armed after the skipped cycle.
    ///
    /// Discriminator: on the PARENT branch (promote-blind gate, no error flag) the
    /// degraded cycle 1 promotes immediately, so `store.version()` would read 1
    /// after cycle 1 and the `version stays 0` assertion fails — this test fails
    /// under the old behavior and passes only with the error-aware gate.
    ///
    /// All waits are bounded by timeouts so a regression that wedges the eval task
    /// fails the TEST rather than hanging CI.
    #[tokio::test]
    async fn eval_loop_skips_promotion_on_degraded_cycle_then_rearms() {
        use crate::selfplay::inference::{
            BatcherConfig, ChannelEvaluator, InferenceBatcher, RandomBackend, SwappableBackend,
        };
        use std::sync::atomic::{AtomicUsize, Ordering};
        use tokio::sync::mpsc;

        let precomputed = Arc::new(crate::PrecomputedItems::begin_precomputing());
        let challenger: Arc<dyn Evaluator> = Arc::new(RandomEvaluator);

        // Champion = ChannelEvaluator over a LIVE batcher whose swappable backend
        // starts in a "dropping" state: replies are dropped so the champion
        // recovers to neutral and trips its error flag (cycle 1). For cycle 2 we
        // swap the backend to a healthy RandomBackend so the champion answers
        // normally. The batcher task stays alive throughout (mirroring the
        // production swappable-champion architecture); only the inner backend
        // changes between cycles.
        let dropped = Arc::new(AtomicUsize::new(0));
        let initial_backend: Box<dyn crate::selfplay::inference::InferenceBackend> =
            Box::new(CountingDroppingBackend { dropped: dropped.clone() });
        let (champion_swappable, champion_backend_handle) = SwappableBackend::new(initial_backend);

        let (champion_tx, champion_rx) = mpsc::channel(32);
        // Keepalive clone so the batcher's rx stays open even after a promotion
        // swaps the stored champion away (cycle 2). Mirrors selfplay.rs keepalive.
        let _keepalive_tx = champion_tx.clone();
        let mut champion_batcher = InferenceBatcher::new(
            champion_rx,
            Box::new(champion_swappable),
            BatcherConfig { max_batch_size: 4, batch_timeout_ms: 5 },
        );
        let batcher_handle = tokio::spawn(async move { champion_batcher.run().await });

        let champion_channel_eval = ChannelEvaluator::with_channels(champion_tx, 64);
        let champion_error_flag = champion_channel_eval.error_flag();
        let champion_eval: Arc<dyn Evaluator> = Arc::new(champion_channel_eval);

        let (version_tx, version_rx) = watch::channel(0u64);
        let ckpt_path: Arc<Mutex<Option<PathBuf>>> = Arc::new(Mutex::new(None));
        let champion_store = Arc::new(ChampionStore::new(champion_eval, 5));
        let store_ref = champion_store.clone();

        // Empty pool (nonexistent dir) → bootstrap path exercises the champion.
        let (_tmp, _checkpoints_dir, eval_pgn_path) = isolated_paths();
        let config = EvaluationConfig {
            games_per_side: 1,
            promotion_threshold: 0.0, // A CLEAN cycle always promotes; degraded must still skip.
            promotion_cooldown_games: 0,
            num_simulations: 2,
            temperature_moves: 2,
            poll_interval_ms: 10,
            champion_score_weight: 2.0,
            checkpoints_dir: PathBuf::from("/nonexistent/test/dir/rearm"),
            eval_pgn_path,
            ..EvaluationConfig::default()
        };

        let mut task = EvaluationTask::new(
            precomputed,
            challenger,
            version_rx,
            ckpt_path,
            champion_store,
            config,
        )
        // Register the champion error flag (challenger is RandomEvaluator → None):
        // a cycle whose champion degraded must skip its promotion decision.
        .with_error_flags(None, Some(champion_error_flag));

        let task_handle = tokio::spawn(async move {
            task.run().await;
        });

        // --- Cycle 1: degraded champion → MUST NOT promote. ---
        version_tx.send(1).expect("send 1 failed");
        // Wait until the champion batcher has been hit (proves cycle 1 actually ran
        // the bootstrap games and degraded), rather than racing a fixed sleep.
        let ran = tokio::time::timeout(std::time::Duration::from_secs(40), async {
            while dropped.load(Ordering::SeqCst) == 0 {
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            }
        })
        .await;
        assert!(
            ran.is_ok(),
            "cycle 1 never reached the champion: bootstrap path did not run"
        );
        // Wait until the champion call count PLATEAUS: once `dropped` stops rising
        // the bootstrap games are over and the loop has reached the promotion gate.
        // The promotion (or its skip) happens synchronously right after the final
        // champion call, so a plateau is a reliable "cycle 1 has decided" signal —
        // no fixed sleep racing the cycle. Under a working error-aware gate the
        // store stays at version 0 here; under the parent's promote-blind gate the
        // degraded cycle would already have promoted to version 1 by this point.
        let decided = tokio::time::timeout(std::time::Duration::from_secs(40), async {
            let mut last = dropped.load(Ordering::SeqCst);
            let mut stable = 0u32;
            loop {
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                let now = dropped.load(Ordering::SeqCst);
                if now == last {
                    stable += 1;
                    // ~1s of no new champion calls ⇒ cycle 1's games are done and
                    // the gate has run.
                    if stable >= 10 {
                        break;
                    }
                } else {
                    stable = 0;
                    last = now;
                }
            }
        })
        .await;
        assert!(
            decided.is_ok(),
            "cycle 1 never settled: champion calls did not plateau (eval task wedged?)"
        );
        // THE DISCRIMINATOR: the degraded cycle must NOT have promoted. On the
        // parent branch (error-blind gate) `win_rate >= 0.0` promotes the
        // challenger against the neutral champion, so this reads 1 and fails.
        assert_eq!(
            store_ref.version(),
            0,
            "degraded cycle 1 promoted the challenger (version={}) — gate is error-blind",
            store_ref.version()
        );

        // --- Cycle 2: healthy champion → MUST re-arm and promote. ---
        // Swap the champion backend to a healthy one: replies now succeed, so the
        // champion no longer degrades and the gate lets the promotion through.
        {
            let mut guard = champion_backend_handle
                .lock()
                .expect("champion backend lock poisoned");
            *guard = Box::new(RandomBackend::new(64));
        }
        version_tx.send(2).expect("send 2 failed");
        let rearmed = tokio::time::timeout(std::time::Duration::from_secs(40), async {
            loop {
                if store_ref.version() == 2 {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            }
        })
        .await;
        assert!(
            rearmed.is_ok(),
            "eval loop failed to re-arm: a clean version 2 never promoted after the skipped cycle"
        );

        drop(version_tx);
        let _ = tokio::time::timeout(std::time::Duration::from_secs(20), task_handle).await;
        batcher_handle.abort();
        let _ = batcher_handle.await;
    }
}
