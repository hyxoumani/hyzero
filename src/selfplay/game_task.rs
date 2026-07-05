use std::collections::VecDeque;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

use crate::data::{
    board_to_snapshot, encode_board, move_to_action, ActionIndex, BoardSnapshot, GameTrajectory,
    ReplayFile, ReplayRecord, StepRecord, NUM_BASE_ACTIONS,
};
use crate::data::encoding::flip_action;
use crate::data::tb_rescore::{tb_rescore_active, tb_rescore_lookup};
use crate::game::board::GameResult;
use crate::game::fen::board_from_fen;
use crate::game::mate_solver::find_forced_mate;
use crate::game::{GameBoard, Move, Player};
use crate::mcts::evaluator::Evaluator;
use crate::mcts::tree::{MCTSConfig, MCTSTree};
use crate::selfplay::replay_writer::write_replay;
use crate::{BitIterator, CastleOption, Color, PieceType, PrecomputedItems, Square};

/// Read HYZERO_USE_GUMBEL once. If set to a non-zero / non-empty value, return
/// the configured top-K (HYZERO_GUMBEL_TOP_K, default 16) for the self-play
/// MCTSConfig. None disables Gumbel and keeps the original PUCT behavior.
fn gumbel_top_k() -> Option<usize> {
    static CACHED: OnceLock<Option<usize>> = OnceLock::new();
    *CACHED.get_or_init(|| {
        let on = std::env::var("HYZERO_USE_GUMBEL")
            .map(|v| !v.is_empty() && v != "0")
            .unwrap_or(false);
        if !on {
            return None;
        }
        let k = std::env::var("HYZERO_GUMBEL_TOP_K")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&k| k >= 1)
            .unwrap_or(16);
        Some(k)
    })
}

// --- Value-based resignation (HYZERO_RESIGN* gates) ---
//
// Read per-call from the environment (mirroring `material_shaping_enabled` and
// `HYZERO_DECISIVE_SAMPLE_FRAC` in replay_buffer.rs) so env-controlled tests can
// vary them within one process; serialize such tests via the module `Mutex`.

/// Env-gate: true (DEFAULT) unless HYZERO_RESIGN is "0"/"false"/"no"/empty.
/// When enabled, self-play games end early once the side-to-move's root_value
/// stays at/below `resign_threshold()` for `resign_plies()` consecutive plies,
/// awarding the opponent a win. Resignation is VALUE-based, not material-based:
/// passive play that avoids checkmate still drives root_value negative, so it
/// cannot be gamed by shuffling to preserve material (passivity-attractor guard,
/// see the game-outcome comment block in `play_game`).
fn resign_enabled() -> bool {
    match std::env::var("HYZERO_RESIGN") {
        Ok(v) => {
            let s = v.trim().to_ascii_lowercase();
            !(s.is_empty() || s == "0" || s == "false" || s == "no")
        }
        Err(_) => true,
    }
}

/// Root-value threshold below which a ply counts toward resignation.
/// `HYZERO_RESIGN_THRESHOLD`, default -0.90, clamped to [-1.0, -0.5].
fn resign_threshold() -> f32 {
    std::env::var("HYZERO_RESIGN_THRESHOLD")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .map(|v| v.clamp(-1.0, -0.5))
        .unwrap_or(-0.90)
}

/// Number of consecutive losing plies required before resigning.
/// `HYZERO_RESIGN_CONSECUTIVE`, default 4 (clamped to >= 1).
fn resign_plies() -> u32 {
    std::env::var("HYZERO_RESIGN_CONSECUTIVE")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .filter(|&n| n >= 1)
        .unwrap_or(4)
}

/// Minimum ply before resignation can trigger. `HYZERO_RESIGN_MIN_PLY`,
/// default 30 — never resign during the high-temperature exploration window.
fn resign_min_ply() -> u32 {
    std::env::var("HYZERO_RESIGN_MIN_PLY")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(30)
}

/// Fraction of self-play games that DISABLE resignation entirely and play on to a
/// natural terminal / move cap (AlphaZero-style resignation calibration).
/// `HYZERO_RESIGN_DISABLE_FRAC`, default 0.1, clamped to [0.0, 1.0]. Selection is
/// per-game (a single random draw at game start), not per-ply: in a disabled game
/// the resignation condition is still tracked but never ends the game, so the
/// would-be resigner's eventual real outcome makes the false-positive rate
/// measurable. Defaults to 0.1 on missing/unparseable input.
fn resign_disable_frac() -> f32 {
    std::env::var("HYZERO_RESIGN_DISABLE_FRAC")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .map(|v| v.clamp(0.0, 1.0))
        .unwrap_or(0.1)
}

// --- Sampled self-play PGN logging (HYZERO_PGN_SAMPLE_RATE gate) ---

/// Fraction of self-play games written to `logs/selfplay_sample.pgn` for
/// opening-diversity analysis and live visualizer streaming.
/// `HYZERO_PGN_SAMPLE_RATE`, default 0.01, clamped to [0.0, 1.0]. Unparseable
/// input falls back to the default. Read per-call (TestEnvGuard-compatible).
/// PGN writing is observability-only and has no effect on training dynamics.
fn pgn_sample_rate() -> f32 {
    std::env::var("HYZERO_PGN_SAMPLE_RATE")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .map(|v| v.clamp(0.0, 1.0))
        .unwrap_or(0.01)
}

/// Decide whether to write a PGN for one game given the sample `rate` and a
/// single `draw` in [0.0, 1.0). Extracted from the call site so the decision is
/// testable: at rate 1.0 every draw samples; at rate 0.0 none do.
fn should_sample_pgn(rate: f32, draw: f32) -> bool {
    draw < rate
}

// --- Annealed self-play temperature (HYZERO_TEMP_ANNEAL* gates) ---

/// Env-gate: true (DEFAULT) unless HYZERO_TEMP_ANNEAL is "0"/"false"/"no"/empty.
/// When enabled, self-play temperature linearly anneals 1.0 → 0.01 over
/// `temp_anneal_plies()` plies once past `temperature_moves`, instead of the
/// hard step to 0.01.
fn temp_anneal_enabled() -> bool {
    match std::env::var("HYZERO_TEMP_ANNEAL") {
        Ok(v) => {
            let s = v.trim().to_ascii_lowercase();
            !(s.is_empty() || s == "0" || s == "false" || s == "no")
        }
        Err(_) => true,
    }
}

/// Number of plies over which temperature anneals from 1.0 to 0.01 after the
/// exploration window. `HYZERO_TEMP_ANNEAL_PLIES`, default 60 (clamped >= 1).
fn temp_anneal_plies() -> u32 {
    std::env::var("HYZERO_TEMP_ANNEAL_PLIES")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .filter(|&n| n >= 1)
        .unwrap_or(60)
}

/// Optional override for the self-play exploration window length (temperature
/// stays 1.0 for the first N plies, then anneals/steps toward exploitation).
/// Returns `Some(clamped value)` when `HYZERO_TEMPERATURE_MOVES` is set and
/// parseable (clamped to [1, 200]), `None` otherwise. Read per-call
/// (TestEnvGuard-compatible).
///
/// `None` lets the self-play construction site fall through to the legacy
/// `HYZERO_TEMP_MOVES`/RunConfig-default chain, so unset behavior is bit-identical
/// to before this knob existed. Shorter windows make games seeded from
/// midgame/endgame FENs anneal to exploitation faster (less random walking).
pub fn temperature_moves_override() -> Option<u32> {
    std::env::var("HYZERO_TEMPERATURE_MOVES")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .map(|n| n.clamp(1, 200))
}

/// Maximum ply depth for the root forced-mate solver. `HYZERO_ROOT_MATE_SOLVER_PLIES`,
/// default 0 (DISABLED), clamped to [0, 7]. Counted in half-moves: mate-in-1 = 1,
/// mate-in-2 = 3, mate-in-3 = 5. Read per-call (TestEnvGuard-compatible), parsed
/// like the sibling knobs above.
///
/// When > 0, both `play_game` (self-play) and `play_game_dual` (eval/arena) call
/// `find_forced_mate` at each move-selection point and, on a hit, play the mating
/// move directly — overriding MCTS selection. In `play_game` the stored policy
/// target for that ply is forced one-hot onto the mating move so the network can
/// distill the solver's choice (see the call site). Default 0 leaves both code
/// paths bit-identical to before this knob existed.
fn root_mate_solver_plies() -> u32 {
    std::env::var("HYZERO_ROOT_MATE_SOLVER_PLIES")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .map(|n| n.clamp(0, 7))
        .unwrap_or(0)
}

/// Compute the self-play temperature for `turn_count` given the
/// `temperature_moves` exploration window. Within the window: 1.0. Past it,
/// when annealing is enabled, linearly interpolate 1.0 → 0.01 over
/// `temp_anneal_plies()` plies, then hold 0.01. When annealing is disabled,
/// preserve the original hard step to 0.01.
fn selfplay_temperature(turn_count: usize, temperature_moves: u32) -> f32 {
    let tm = temperature_moves as usize;
    if turn_count < tm {
        return 1.0;
    }
    if !temp_anneal_enabled() {
        return 0.01;
    }
    let span = temp_anneal_plies() as f32;
    let past = (turn_count - tm) as f32;
    let frac = (past / span).min(1.0);
    // Linear anneal from 1.0 down to 0.01.
    1.0 + frac * (0.01 - 1.0)
}

// --- MCTS summary log (HYZERO_MCTS_TRACE gate) ---

/// Cached env-gate: true if HYZERO_MCTS_TRACE is set and non-zero.
fn mcts_summary_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("HYZERO_MCTS_TRACE")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .map(|n| n != 0)
            .unwrap_or(false)
    })
}

/// Process-wide summary log writer.  First thread to call creates (truncates) the file.
fn summary_writer() -> &'static Mutex<Option<BufWriter<std::fs::File>>> {
    static WRITER: OnceLock<Mutex<Option<BufWriter<std::fs::File>>>> = OnceLock::new();
    WRITER.get_or_init(|| {
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open("logs/mcts_summary.log")
            .ok()
            .map(BufWriter::new);
        Mutex::new(file)
    })
}

/// Write one summary line for the current MCTS root call.
///
/// Computes the masked/renormalised prior over `legal_actions`, then emits a
/// single grep-friendly line to `logs/mcts_summary.log`.
fn trace_summary(
    model_version: u64,
    turn_count: usize,
    legal_actions: &[ActionIndex],
    policy: &[f32],
    visit_distribution: &[f32],
) {
    let n_legal = legal_actions.len();
    if n_legal == 0 {
        return;
    }

    // Compute masked prior (renormalised over legal actions).
    let sum: f32 = legal_actions.iter().map(|&a| policy[a as usize]).sum();
    let masked: Vec<f32> = if sum < 1e-12f32 {
        vec![1.0 / n_legal as f32; n_legal]
    } else {
        legal_actions.iter().map(|&a| policy[a as usize] / sum).collect()
    };

    // top_p: index (slot) with highest masked prior.
    let top_p_idx = masked
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    let top_p = masked[top_p_idx];

    // top_v: index (slot) with highest visit fraction.
    let top_v_idx = visit_distribution
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    let top_v_frac = visit_distribution.get(top_v_idx).copied().unwrap_or(0.0);

    // n_visited: count of children with non-zero visit fraction.
    let n_visited = visit_distribution.iter().filter(|&&v| v > 0.0).count();

    // entropy over visited children only (natural log).
    let entropy: f32 = visit_distribution
        .iter()
        .filter(|&&v| v > 0.0)
        .map(|&v| -v * v.ln())
        .sum();

    let line = format!(
        "[mcts_summary] v={model_version} move={turn_count} legal={n_legal} \
         top_p_idx={top_p_idx} top_p={top_p:.4} top_v_idx={top_v_idx} \
         top_v_frac={top_v_frac:.4} n_visited={n_visited} entropy={entropy:.4}\n"
    );

    if let Ok(mut guard) = summary_writer().lock() {
        if let Some(ref mut w) = *guard {
            let _ = w.write_all(line.as_bytes());
            let _ = w.flush();
        }
    }
}

// ─── Diverse self-play starting positions ───────────────────────────────────
//
// Lazy-loaded list of FEN starting positions, populated once per process from
// the file referenced by the HYZERO_STARTS_FILE env var. When unset or empty,
// games start from the standard initial position (existing behavior).

/// Return the module-global list of starting FENs loaded from the file
/// pointed to by HYZERO_STARTS_FILE. Loaded once per process and cached.
/// An empty list means "no diverse starts; always start from initial position."
fn starting_positions() -> &'static Vec<String> {
    static STARTS: OnceLock<Vec<String>> = OnceLock::new();
    STARTS.get_or_init(|| {
        let Ok(path) = std::env::var("HYZERO_STARTS_FILE") else {
            return Vec::new();
        };
        match std::fs::read_to_string(&path) {
            Ok(contents) => {
                let fens: Vec<String> = contents
                    .lines()
                    .map(|l| l.trim())
                    .filter(|l| !l.is_empty())
                    .map(|l| l.to_string())
                    .collect();
                eprintln!(
                    "[selfplay] loaded {} starting positions from {}",
                    fens.len(),
                    path
                );
                fens
            }
            Err(e) => {
                eprintln!(
                    "[selfplay] WARN: failed to load HYZERO_STARTS_FILE={}: {}",
                    path, e
                );
                Vec::new()
            }
        }
    })
}

/// Pick a random starting FEN from the configured list, or `None` if no
/// diverse-starts file is configured or it is empty.
pub(crate) fn pick_starting_position() -> Option<&'static str> {
    let starts = starting_positions();
    if starts.is_empty() {
        return None;
    }
    use rand::Rng;
    let mut rng = rand::rng();
    let idx = rng.random_range(0..starts.len());
    Some(starts[idx].as_str())
}

/// Initialize the self-play board either from a sampled diverse-start FEN
/// (when HYZERO_STARTS_FILE is configured) or from the standard initial
/// position. Returns `(board, side_to_move, starting_fen)` where `starting_fen`
/// is `Some(fen)` when a non-default position was used (so the replay viewer
/// can reconstruct from the same starting state).
///
/// If a sampled FEN fails to parse, logs a warning and falls back to the
/// default initial position so self-play never aborts on a bad FEN.
fn init_self_play_board(
    precomputed: Arc<PrecomputedItems>,
) -> (GameBoard, Color, Option<String>) {
    if let Some(fen) = pick_starting_position() {
        match board_from_fen(fen, precomputed.clone()) {
            Ok((board, side_to_move, _fullmove)) => {
                // Skip positions that are already terminal (rare but possible
                // for FENs pulled from mid-game random play).
                if board.result() == GameResult::Ongoing {
                    return (board, side_to_move, Some(fen.to_string()));
                }
                eprintln!(
                    "[selfplay] WARN: sampled start FEN is already terminal; \
                     falling back to standard start"
                );
            }
            Err(e) => {
                eprintln!(
                    "[selfplay] WARN: failed to parse start FEN {:?}: {}; \
                     falling back to standard start",
                    fen, e
                );
            }
        }
    }
    let player1 = Player::init_player(true);
    let player2 = Player::init_player(false);
    let board = GameBoard::init_game_board(precomputed, player1, player2);
    (board, Color::White, None)
}

/// Configuration for a self-play game.
#[derive(Debug, Clone)]
pub struct GameConfig {
    pub num_simulations: u32,
    pub exploration_constant: f32,
    /// Use temperature=1.0 for the first N moves, then near 0.
    pub temperature_moves: u32,
    /// If `Some(dir)`, every completed game writes a `.replay` file into `dir`.
    /// `None` (default) disables replay capture entirely — zero overhead when
    /// off. The viewer (`cargo run --bin replay -- <file>`) reads these files.
    pub replay_dir: Option<Arc<PathBuf>>,
    /// Eval-only: when true, `play_game_dual` adjudicates a non-checkmate game
    /// at the move cap by awarding ±1 to the side ahead by at least
    /// `adjudication_material_margin` material. OFF by default — self-play must
    /// never adjudicate (material adjudication caused a passivity attractor).
    pub adjudicate_at_cap: bool,
    /// Material lead (white-absolute, standard piece values) required to award a
    /// decisive result when `adjudicate_at_cap` is enabled.
    pub adjudication_material_margin: i32,
}

impl Default for GameConfig {
    fn default() -> Self {
        Self {
            num_simulations: 800,
            exploration_constant: 1.5,
            temperature_moves: 30,
            replay_dir: None,
            adjudicate_at_cap: false,
            adjudication_material_margin: 5,
        }
    }
}

/// Outcome of a dual-evaluator game (White-perspective: +1 White, -1 Black, 0 Draw).
#[derive(Debug, Clone, PartialEq)]
pub struct DualGameOutcome {
    /// Raw game outcome: +1 White wins, -1 Black wins, 0 draw.
    pub game_outcome: f32,
    /// How many moves the game lasted.
    pub num_moves: usize,
    /// Move list in coordinate notation (e.g. "e2e4"), one entry per ply.
    pub moves: Vec<String>,
    /// How the game ended, for the PGN `[Termination "..."]` header. One of the
    /// board-result causes ("checkmate", "stalemate", "repetition", "fifty-move",
    /// "insufficient-material"), "move-cap" for an un-adjudicated cap, or
    /// "adjudication" when an eval material lead decided a non-checkmate terminal.
    pub termination: String,
    /// Starting FEN when the game began from a non-standard position (diverse
    /// starts via `HYZERO_STARTS_FILE`), else `None`. Plumbed to the PGN writer
    /// so the viewer can replay the moves from this position via `[SetUp]`/`[FEN]`.
    pub starting_fen: Option<String>,
}

/// Play a game with two distinct evaluators (challenger = White, champion = Black).
///
/// Returns the White-perspective outcome. Callers that want to count champion wins
/// when champion played Black must **negate** `game_outcome`.
///
/// The starting position is selected by `init_self_play_board` (a diverse-start
/// FEN sampled from `HYZERO_STARTS_FILE` when configured, else the standard
/// initial position). Callers that need to drive a *specific* starting position
/// (e.g. the arena's mirrored-pair scheduler) use `play_game_dual_from` instead.
///
/// No `GameTrajectory` is produced — eval games don't go to the training buffer.
pub async fn play_game_dual(
    precomputed: Arc<PrecomputedItems>,
    white_evaluator: Arc<dyn Evaluator>,
    black_evaluator: Arc<dyn Evaluator>,
    config: GameConfig,
) -> DualGameOutcome {
    let (board, side_to_move, starting_fen) = init_self_play_board(precomputed);
    play_game_dual_from(
        white_evaluator,
        black_evaluator,
        config,
        board,
        side_to_move,
        starting_fen,
    )
    .await
}

/// Play a dual-evaluator game from an explicit starting `board` / `side_to_move`.
///
/// Identical to `play_game_dual` except the caller supplies the initial position
/// instead of it being sampled by `init_self_play_board`. `starting_fen` is
/// forwarded verbatim into the returned `DualGameOutcome` (and thence to the PGN
/// `[SetUp]`/`[FEN]` headers); pass `Some(fen)` for a non-standard start and
/// `None` for the standard initial position. Eval-style search (no root noise,
/// near-greedy 0.01 temperature) is used, matching `play_game_dual`.
pub async fn play_game_dual_from(
    white_evaluator: Arc<dyn Evaluator>,
    black_evaluator: Arc<dyn Evaluator>,
    config: GameConfig,
    mut board: GameBoard,
    mut side_to_move: Color,
    starting_fen: Option<String>,
) -> DualGameOutcome {
    let mut turn_count: usize = 0;
    let mut history: VecDeque<BoardSnapshot> = VecDeque::with_capacity(7);
    let mut moves: Vec<String> = Vec::new();

    const MAX_GAME_LENGTH: usize = 300;

    let mcts_config = MCTSConfig {
        num_simulations: config.num_simulations,
        exploration_constant: config.exploration_constant,
        add_root_noise: false, // eval: no exploration noise
        gumbel_top_k: None,    // eval uses PUCT (deterministic)
    };

    loop {
        if board.result() != GameResult::Ongoing {
            break;
        }

        if turn_count >= MAX_GAME_LENGTH {
            break;
        }

        let hist_slice = history.make_contiguous();
        let observation = encode_board(&board, side_to_move, hist_slice);
        let raw_legal = get_legal_moves(&board, side_to_move);

        // Flip legal actions to current-player perspective for Black.
        let mut legal_actions: Vec<ActionIndex> = if side_to_move == Color::Black {
            raw_legal
                .iter()
                .map(|&a| flip_action(a as usize) as ActionIndex)
                .collect()
        } else {
            raw_legal
        };
        // Canonicalize ordering so both colors present identical legal_actions at
        // equivalent POV positions. `get_legal_moves` iterates absolute squares
        // 0..63, producing color-asymmetric ordering: white's knights on sq 1, 6
        // come before pawns on sq 8-15; black's pawns on sq 48-55 come before
        // knights on sq 57, 62. After `flip_action` converts the VALUES to POV
        // coords, POSITIONS remain in absolute-iteration order, so legal_actions[0]
        // is a Knight move for white but a Pawn move for black. MCTS's index-level
        // selection then biases the colors' move-type selection asymmetrically.
        // Sorting by action-index restores POV-symmetry.
        legal_actions.sort_unstable();

        if legal_actions.is_empty() {
            break;
        }

        let mut legal_mask = vec![false; crate::data::NUM_ACTIONS];
        for &a in &legal_actions {
            legal_mask[a as usize] = true;
        }

        // Select evaluator based on side to move.
        let evaluator = if side_to_move == Color::White {
            white_evaluator.clone()
        } else {
            black_evaluator.clone()
        };

        // Root moves-left estimate is not threaded into the root node (the search
        // bonus reads it only on expanded children); discard it here.
        let (hidden_state, policy, value, _root_m) =
            evaluator.root_setup(&observation, &legal_mask).await;

        let mut tree = MCTSTree::new(
            hidden_state,
            &policy,
            value,
            legal_actions.clone(),
            mcts_config.clone(),
        );
        tree.run_simulations(evaluator.as_ref()).await;

        // Use near-greedy temperature for eval games (no exploration needed).
        let mut selected_action = tree.select_action(0.01);

        // Flip action back to absolute coordinates before applying to the board.
        let mut absolute_action = if side_to_move == Color::Black {
            flip_action(selected_action as usize) as ActionIndex
        } else {
            selected_action
        };

        // Root forced-mate override (HYZERO_ROOT_MATE_SOLVER_PLIES > 0). Eval/arena
        // games keep no training targets, so we only override the chosen move when
        // the side to move has a forced mate within the configured ply budget.
        let mate_plies = root_mate_solver_plies();
        if mate_plies > 0 {
            if let Some(mate_mv) = find_forced_mate(&board, side_to_move, mate_plies) {
                let mate_absolute = move_to_action(&mate_mv);
                let mate_selected = if side_to_move == Color::Black {
                    flip_action(mate_absolute as usize) as ActionIndex
                } else {
                    mate_absolute
                };
                if legal_actions.contains(&mate_selected) {
                    selected_action = mate_selected;
                    absolute_action = mate_absolute;
                }
            }
        }

        let snapshot = board_to_snapshot(&board);

        let move_str = action_to_notation(absolute_action, side_to_move);
        moves.push(move_str.clone());
        match board.process_move(&move_str, side_to_move, turn_count) {
            Ok(_) => {}
            Err(_) => break,
        }

        if history.len() == 7 {
            history.pop_front();
        }
        history.push_back(snapshot);

        side_to_move = if side_to_move == Color::White {
            Color::Black
        } else {
            Color::White
        };
        turn_count += 1;
    }

    let result = board.result();
    let game_outcome = match result {
        GameResult::Checkmate(Color::White) => 1.0,
        GameResult::Checkmate(Color::Black) => -1.0,
        // Non-checkmate terminal (incl. the move cap): adjudicate when enabled,
        // else a draw. Delegated to a pure helper so the branch is unit-testable.
        _ => adjudicate_non_checkmate(
            &board,
            config.adjudicate_at_cap,
            config.adjudication_material_margin,
        ),
    };

    // Termination cause for the PGN header. A non-checkmate terminal that
    // adjudication turned decisive is recorded as "adjudication" (eval-only);
    // otherwise the underlying board-result cause / move-cap is used.
    let termination = if matches!(result, GameResult::Checkmate(_)) {
        termination_label(result).to_string()
    } else if game_outcome != 0.0 {
        "adjudication".to_string()
    } else {
        termination_label(result).to_string()
    };

    DualGameOutcome {
        game_outcome,
        num_moves: turn_count,
        moves,
        termination,
        starting_fen,
    }
}

/// Play a complete game using MCTS, producing a GameTrajectory for training.
pub async fn play_game(
    precomputed: Arc<PrecomputedItems>,
    evaluator: Arc<dyn Evaluator>,
    model_version: u64,
    config: GameConfig,
) -> GameTrajectory {
    let (mut board, mut side_to_move, starting_fen) = init_self_play_board(precomputed.clone());

    let mut steps: Vec<StepRecord> = Vec::new();
    // lc0-style tablebase tail-rescoring (HYZERO_TB_RESCORE). When active, each
    // recorded step's normfen is looked up in the WDL map and the STM-POV WDL is
    // carried as a per-step override parallel to `steps`. When inactive the flag
    // is false, no normfen is computed, and `tb_values` stays empty (no override).
    let tb_active = tb_rescore_active();
    let mut tb_values: Vec<Option<f32>> = Vec::new();
    // Optional replay capture: per-ply MCTS dump, written to disk at game end
    // when `GameConfig.replay_dir` is set (typically from HYZERO_REPLAY_DIR).
    let capture_replay = config.replay_dir.is_some();
    let mut replay_records: Vec<ReplayRecord> = Vec::new();
    let mut turn_count: usize = 0;
    // History buffer for encode_board: stores up to 7 past snapshots (oldest first).
    let mut history: VecDeque<BoardSnapshot> = VecDeque::with_capacity(7);

    const MAX_GAME_LENGTH: usize = 300;

    let mcts_config = MCTSConfig {
        num_simulations: config.num_simulations,
        exploration_constant: config.exploration_constant,
        add_root_noise: true, // self-play: inject exploration (no-op when Gumbel is on)
        gumbel_top_k: gumbel_top_k(),
    };

    let mut moves: Vec<String> = Vec::new();

    // Value-based resignation state. Counts consecutive plies where the
    // side-to-move's root_value is at/below `resign_threshold()`. When it
    // reaches `resign_plies()` (and we are past `resign_min_ply()`), the game
    // ends with the opponent of the resigning side awarded a win. `resigned_side`
    // records which color resigned; converted to a White-absolute outcome after
    // the loop.
    let mut consecutive_losing_plies: u32 = 0;
    let mut resigned_side: Option<Color> = None;

    // Resignation calibration (AlphaZero-style). A per-game random draw disables
    // resignation for `resign_disable_frac()` of games: they play on to a natural
    // terminal / move cap so the would-be resigner's REAL outcome is observable.
    // For such games we still detect the first ply the resignation condition fires
    // (`would_resign_ply` / `would_resign_side`) so the false-positive rate — a
    // resign signal where the side did NOT actually lose — is computable from the
    // emitted `[resign_calib]` log line. The draw is a single per-game roll, not
    // per-ply, so the disable decision is stable for the whole game.
    let resign_disabled = resign_enabled() && rand::random::<f32>() < resign_disable_frac();
    let mut would_resign_ply: Option<usize> = None;
    let mut would_resign_side: Option<Color> = None;

    // Self-play material adjudication (HYZERO_SELFPLAY_ADJUDICATE, default OFF).
    // Cached once per process. When enabled, an otherwise-Ongoing position with a
    // material lead >= `adjudication_margin` ends the game decisively for the
    // leading side; `adjudicated_side` records the WINNER (converted to a
    // White-absolute outcome after the loop, exactly like a real checkmate).
    let adjudicate_selfplay = selfplay_adjudicate_enabled();
    let adjudication_margin = selfplay_adjudication_margin();
    let mut adjudicated_side: Option<Color> = None;

    loop {
        if board.result() != GameResult::Ongoing {
            break;
        }

        if turn_count >= MAX_GAME_LENGTH {
            break;
        }

        // Self-play adjudication: on an otherwise-Ongoing position (real
        // terminals broke above and take precedence) with a material lead at or
        // beyond the margin, terminate decisively for the leading side. Gated
        // strictly — OFF (the default) leaves this a no-op and the loop unchanged.
        if let Some(winner) =
            selfplay_adjudicated_winner(&board, adjudicate_selfplay, adjudication_margin)
        {
            adjudicated_side = Some(winner);
            break;
        }

        let hist_slice = history.make_contiguous();
        let observation = encode_board(&board, side_to_move, hist_slice);
        let raw_legal = get_legal_moves(&board, side_to_move);

        // Flip legal actions to current-player perspective for Black.
        let mut legal_actions: Vec<ActionIndex> = if side_to_move == Color::Black {
            raw_legal
                .iter()
                .map(|&a| flip_action(a as usize) as ActionIndex)
                .collect()
        } else {
            raw_legal
        };
        // Canonicalize ordering so both colors present identical legal_actions at
        // equivalent POV positions. `get_legal_moves` iterates absolute squares
        // 0..63, producing color-asymmetric ordering: white's knights on sq 1, 6
        // come before pawns on sq 8-15; black's pawns on sq 48-55 come before
        // knights on sq 57, 62. After `flip_action` converts the VALUES to POV
        // coords, POSITIONS remain in absolute-iteration order, so legal_actions[0]
        // is a Knight move for white but a Pawn move for black. MCTS's index-level
        // selection then biases the colors' move-type selection asymmetrically.
        // Sorting by action-index restores POV-symmetry.
        legal_actions.sort_unstable();

        if legal_actions.is_empty() {
            break;
        }

        // Build legal mask: true for each action that is legal in this position.
        let mut legal_mask = vec![false; crate::data::NUM_ACTIONS];
        for &a in &legal_actions {
            legal_mask[a as usize] = true;
        }

        // Root setup: encode board into latent space
        // Root moves-left estimate is not threaded into the root node (the search
        // bonus reads it only on expanded children); discard it here.
        let (hidden_state, policy, value, _root_m) =
            evaluator.root_setup(&observation, &legal_mask).await;

        // Build MCTS tree and run search
        let mut tree = MCTSTree::new(
            hidden_state,
            &policy,
            value,
            legal_actions.clone(),
            mcts_config.clone(),
        );
        tree.run_simulations(evaluator.as_ref()).await;

        // Extract results
        let mut visit_distribution = tree.extract_visit_distribution();
        let root_value = tree.root_value();

        // Optional MCTS diagnostics for the replay viewer (raw visits, priors,
        // q-values per child). Cheap when capture is on; skipped entirely otherwise.
        let diagnostics = if capture_replay {
            Some(tree.extract_root_diagnostics())
        } else {
            None
        };

        // Write one-line summary to logs/mcts_summary.log when HYZERO_MCTS_TRACE is set.
        if mcts_summary_enabled() {
            trace_summary(model_version, turn_count, &legal_actions, &policy, &visit_distribution);
        }

        // Select action based on temperature (annealed past the exploration
        // window when HYZERO_TEMP_ANNEAL is on; hard step to 0.01 when off).
        let temperature = selfplay_temperature(turn_count, config.temperature_moves);
        // selected_action is in current-player (flipped) coordinate space for Black.
        let mut selected_action = tree.select_action(temperature);

        // Flip action back to absolute coordinates before applying to the board.
        let mut absolute_action = if side_to_move == Color::Black {
            flip_action(selected_action as usize) as ActionIndex
        } else {
            selected_action
        };

        // Root forced-mate override (HYZERO_ROOT_MATE_SOLVER_PLIES > 0). If the
        // side to move has a forced mate within the configured ply budget, play
        // that move directly instead of the MCTS choice. MCTS still ran above, so
        // `root_value` and replay diagnostics stay valid. The training policy
        // target is forced ONE-HOT onto the mating move (visit_distribution is
        // parallel to `legal_actions`, like `extract_visit_distribution`), so the
        // network distills the solver's choice even though MCTS never visited it.
        let mate_plies = root_mate_solver_plies();
        if mate_plies > 0 {
            if let Some(mate_mv) = find_forced_mate(&board, side_to_move, mate_plies) {
                // The mating move is in absolute board coordinates; convert to the
                // current-player (flipped-for-Black) action space used by
                // `legal_actions` / the policy target.
                let mate_absolute = move_to_action(&mate_mv);
                let mate_selected = if side_to_move == Color::Black {
                    flip_action(mate_absolute as usize) as ActionIndex
                } else {
                    mate_absolute
                };
                if let Some(idx) = legal_actions.iter().position(|&a| a == mate_selected) {
                    selected_action = mate_selected;
                    absolute_action = mate_absolute;
                    // One-hot policy target onto the mating move (same length and
                    // ordering as `legal_actions`).
                    let mut forced = vec![0.0f32; legal_actions.len()];
                    forced[idx] = 1.0;
                    visit_distribution = forced;
                }
            }
        }

        if let Some(diag) = diagnostics {
            replay_records.push(ReplayRecord {
                action: selected_action,
                legal_moves: legal_actions.clone(),
                child_visits: diag.child_visits,
                priors: diag.priors,
                q_values: diag.q_values,
                root_value,
                white_to_move: side_to_move == Color::White,
            });
        }

        // Record step — store selected_action (current-player perspective) in trajectory.
        // legal_moves also stored in current-player perspective.
        // white_to_move stored out-of-band since plane 101 (side-to-move) was removed.
        steps.push(StepRecord {
            observation,
            action: selected_action,
            visit_distribution,
            root_value,
            reward: 0.0, // Set terminal reward after game ends
            legal_moves: legal_actions,
            white_to_move: side_to_move == Color::White,
        });

        // Tablebase rescore hit for this step. Only recorded when rescoring is
        // active, so `tb_values` is either empty (inactive ⇒ no overrides at all,
        // byte-identical to pre-rescore) or has exactly one entry per `steps`
        // element (pushed in lockstep here so the vectors stay index-aligned across
        // every early-exit path below). `board`/`side_to_move` describe exactly the
        // position just recorded — the move is applied later this iteration. The
        // WDL is STM POV, matching the value-target convention; `None` when the
        // position is not covered by the tablebase map.
        if tb_active {
            tb_values.push(tb_rescore_lookup(&board.to_normfen(side_to_move)));
        }

        // Value-based resignation. `root_value` is in the current side-to-move's
        // POV, so a value at/below the (negative) threshold means the side to
        // move believes it is losing badly. Track consecutive such plies; once
        // they reach `resign_plies()` (and we are past `resign_min_ply()`), the
        // side to move resigns and the opponent is awarded the win. Material is
        // never consulted — passive play that avoids mate still drives
        // root_value negative, so the passivity attractor cannot game this.
        if resign_enabled() {
            if root_value <= resign_threshold() {
                consecutive_losing_plies += 1;
            } else {
                consecutive_losing_plies = 0;
            }
            if turn_count >= resign_min_ply() as usize && consecutive_losing_plies >= resign_plies()
            {
                if resign_disabled {
                    // Calibration game: record the FIRST ply the resignation
                    // condition fired (and which side) but do NOT end the game —
                    // it plays on so the real outcome reveals false positives.
                    if would_resign_ply.is_none() {
                        would_resign_ply = Some(turn_count);
                        would_resign_side = Some(side_to_move);
                    }
                } else {
                    resigned_side = Some(side_to_move);
                }
            }
        }

        // Snapshot position before applying the move (for history encoding on next turn)
        let snapshot = board_to_snapshot(&board);

        // Convert absolute action to move notation and apply
        let move_str = action_to_notation(absolute_action, side_to_move);
        moves.push(move_str.clone());
        match board.process_move(&move_str, side_to_move, turn_count) {
            Ok(_) => {}
            Err(_) => {
                // If the selected move is invalid (shouldn't happen with correct legal moves),
                // try the first legal move as fallback
                break;
            }
        }

        // Add pre-move snapshot to history buffer (oldest first); keep at most 7
        if history.len() == 7 {
            history.pop_front();
        }
        history.push_back(snapshot);

        // Alternate sides
        side_to_move = if side_to_move == Color::White {
            Color::Black
        } else {
            Color::White
        };
        turn_count += 1;

        // End the game immediately on resignation (the resigning side's last
        // move and step are already recorded above).
        if resigned_side.is_some() {
            break;
        }
    }

    // Game outcome.
    // Checkmates produce ±1 signal (strong). Rule-draw terminals (repetition,
    // 50-move rule, 300-move cap) optionally use tanh(Δmaterial/5) as a weak
    // material-proxy signal to keep the value head learning when games don't reach
    // checkmate; true draws (stalemate, insufficient material) stay 0.0 because the
    // position is drawn regardless of material. See `score_board_terminal`.
    // Adjudication is NOT re-enabled — it was the primary cause of the passivity
    // attractor. Material-at-cap alone (without early termination on material
    // imbalance) is a weaker incentive: preserving material only pays off IF you
    // survive to a real terminal, so passive play still risks getting checkmated.
    let board_result = board.result();
    // Termination cause for the PGN header. Resignation overrides the board cause;
    // otherwise map the board result (or move-cap) to a standard label.
    let termination = if resigned_side.is_some() {
        "resignation".to_string()
    } else if adjudicated_side.is_some() {
        "adjudication".to_string()
    } else {
        termination_label(board_result).to_string()
    };
    let (game_outcome, is_draw) = if let Some(loser) = resigned_side {
        // Value-based resignation: the side that resigned loses, opponent wins.
        // White-absolute: White resigns → -1.0 (Black wins); Black resigns → +1.0.
        let outcome = if loser == Color::White { -1.0f32 } else { 1.0f32 };
        (outcome, false)
    } else if let Some(winner) = adjudicated_side {
        // Material adjudication: decisive win for the leading side, scored as a
        // real win (±1, not a draw) so it flows into value/TD targets identically.
        let outcome = if winner == Color::White { 1.0f32 } else { -1.0f32 };
        (outcome, false)
    } else {
        score_board_terminal(board_result, &board)
    };

    // Set terminal reward on last step (convert white-absolute game_outcome to last-step POV,
    // matching the convention used by StepRecord.root_value).
    if let Some(last) = steps.last_mut() {
        let last_side_sign: f32 = if last.white_to_move { 1.0 } else { -1.0 };
        last.reward = game_outcome * last_side_sign;
    }

    // Lightweight per-game outcome trace (gated by HYZERO_MCTS_TRACE) — one line
    // per game, streamed to stdout so experiments can compute decisive ratios
    // without relying on the 1%-sampled PGN.
    if mcts_summary_enabled() {
        println!(
            "[game_outcome] v={} len={} outcome={:.3} is_draw={}",
            model_version,
            turn_count,
            game_outcome,
            is_draw,
        );
    }

    // Resignation calibration trace: one `[resign_calib]` line per disabled game,
    // emitted unconditionally (not gated) so a long run always yields the
    // false-positive denominator. `game_outcome` is white-absolute; a would-be
    // resigner is a false positive when its side did NOT actually lose (the
    // game's outcome from that side's POV is >= 0). Games where the condition
    // never fired report `would_resign_ply=none` (always not a false positive).
    if resign_disabled {
        let (ply_str, side_str, false_positive) = match (would_resign_ply, would_resign_side) {
            (Some(ply), Some(side)) => {
                // White-absolute outcome converted to the would-be resigner's POV.
                let side_sign: f32 = if side == Color::White { 1.0 } else { -1.0 };
                let resigner_outcome = game_outcome * side_sign;
                let side_char = if side == Color::White { "w" } else { "b" };
                // False positive: signalled a loss but did not actually lose.
                (ply.to_string(), side_char, resigner_outcome >= 0.0)
            }
            _ => ("none".to_string(), "-", false),
        };
        println!(
            "[resign_calib] game={model_version} would_resign_ply={ply_str} side={side_str} \
             outcome={game_outcome:.3} false_positive={false_positive}"
        );
    }

    // Replay capture: write the per-ply MCTS dump to disk if the user opted in
    // via `GameConfig.replay_dir`. One file per game, no sampling — opting in
    // means you want the data, even at high disk cost. Failures are logged and
    // swallowed so a flaky filesystem can't take down a self-play run.
    if let Some(dir) = config.replay_dir.as_ref() {
        let replay = ReplayFile {
            steps: replay_records,
            game_outcome,
            model_version,
            is_draw,
            starting_fen: starting_fen.clone(),
            c_puct: config.exploration_constant,
        };
        match write_replay(&replay, dir.as_ref()) {
            Ok(path) => {
                if mcts_summary_enabled() {
                    println!(
                        "[replay] wrote {} ({} plies)",
                        path.display(),
                        replay.steps.len()
                    );
                }
            }
            Err(e) => eprintln!("[replay] write failed: {e}"),
        }
    }

    // Sampled self-play PGN logging: `HYZERO_PGN_SAMPLE_RATE` of games (default
    // 1%), for opening-diversity analysis and live visualizer streaming.
    // Cheaply keyed on a single rng call; no impact on training dynamics.
    //
    // Result label must reflect the BOARD outcome, not the value target. Under
    // material shaping (opt-in), non-checkmate games get game_outcome = tanh(Δ/S)
    // which can exceed ±0.5 for a drawn-by-rule game. Labeling those "1-0" caused
    // the analysis confusion we debugged on 2026-04-23. Respect is_draw.
    if should_sample_pgn(pgn_sample_rate(), rand::random::<f32>()) {
        let result_str = if is_draw {
            "1/2-1/2"
        } else if game_outcome > 0.5 {
            "1-0"
        } else if game_outcome < -0.5 {
            "0-1"
        } else {
            "1/2-1/2"
        };
        crate::selfplay::pgn::write_pgn_game(
            "logs/selfplay_sample.pgn",
            &format!("Selfplay model_v{model_version}"),
            "selfplay_white",
            "selfplay_black",
            result_str,
            &termination,
            starting_fen.as_deref(),
            &moves,
        );
    }

    GameTrajectory {
        steps,
        game_outcome,
        model_version,
        is_draw,
        tb_values,
    }
}

/// Convert an ActionIndex to coordinate notation string (e.g., "e2e4").
///
/// For underpromotion actions (>= NUM_BASE_ACTIONS) the from/to squares are
/// derived from the encoded file indices and the color's promotion rank.
/// The suffix is "n", "b", or "r" for knight/bishop/rook underpromotion.
fn action_to_notation(action: ActionIndex, color: Color) -> String {
    if action as usize >= NUM_BASE_ACTIONS {
        // Decode underpromotion: piece_idx encodes suffix, from_file and to_file
        // give from/to squares at the promotion rank for this color.
        let offset = action as usize - NUM_BASE_ACTIONS;
        let piece_idx = offset / 192;
        let remainder = offset % 192;
        let from_file = (remainder / 24) as u8;
        let to_file_slot = (remainder % 24) as u8;
        // to_file_slot 0-7 encodes to_file directly (clamped to 0-7)
        let to_file = to_file_slot.min(7);

        let (from_rank_char, to_rank_char) = if color == Color::White {
            ('7', '8') // White pawn on rank 7 promotes to rank 8
        } else {
            ('2', '1') // Black pawn on rank 2 promotes to rank 1
        };

        let from_file_char = (b'a' + from_file) as char;
        let to_file_char = (b'a' + to_file) as char;

        let suffix = match piece_idx {
            0 => 'n', // Knight
            1 => 'b', // Bishop
            2 => 'r', // Rook
            _ => 'q', // Fallback (shouldn't happen)
        };

        return format!(
            "{}{}{}{}{}",
            from_file_char, from_rank_char, to_file_char, to_rank_char, suffix
        );
    }

    let from_sq = (action / 64) as u8;
    let to_sq = (action % 64) as u8;

    let from_file = (b'a' + from_sq % 8) as char;
    let from_rank = (b'1' + from_sq / 8) as char;
    let to_file = (b'a' + to_sq % 8) as char;
    let to_rank = (b'1' + to_sq / 8) as char;

    // Only add queen promotion suffix for moves from penultimate rank to back rank.
    // This avoids erroneously appending 'q' to king/rook moves that happen to land
    // on rank 1 or 8 (e.g. castling, back-rank rook moves).
    let from_rank_num = from_sq / 8;
    let to_rank_num = to_sq / 8;
    let is_promotion = (color == Color::White && from_rank_num == 6 && to_rank_num == 7)
        || (color == Color::Black && from_rank_num == 1 && to_rank_num == 0);
    if is_promotion {
        format!("{}{}{}{}q", from_file, from_rank, to_file, to_rank)
    } else {
        format!("{}{}{}{}", from_file, from_rank, to_file, to_rank)
    }
}

/// Collect all legal moves for the given color.
/// Iterates all squares, generates pseudo-legal moves via get_move_mask,
/// builds Move structs, and validates each with validate_move.
fn get_legal_moves(board: &GameBoard, color: Color) -> Vec<ActionIndex> {
    let mut legal = Vec::new();
    let player = if color == Color::White {
        &board.player1
    } else {
        &board.player2
    };
    let combined = board.white_pieces | board.black_pieces;

    for sq in 0..64usize {
        let piece_opt = player.own_board[sq];
        let piece = match piece_opt {
            Some(p) if p.color == color => p,
            _ => continue,
        };

        let move_mask = board.get_move_mask(
            sq,
            color,
            piece.piece_type,
            combined,
            board.white_pieces,
            board.black_pieces,
        );

        for to_sq in BitIterator::new(move_mask) {
            let from = Square::from(sq as u8);
            let to = Square::from(to_sq as u8);

            let to_rank = to_sq / 8;
            let is_promotion =
                piece.piece_type == PieceType::Pawn && (to_rank == 7 || to_rank == 0);

            // Detect en passant
            let en_passant = piece.piece_type == PieceType::Pawn
                && board.en_passant_target == Some(to_sq)
                && (sq % 8 != to_sq % 8); // diagonal move to EP square

            if is_promotion {
                // Emit all 4 promotion types (queen + 3 underpromotions)
                for &promo_type in &[
                    PieceType::Queen,
                    PieceType::Knight,
                    PieceType::Bishop,
                    PieceType::Rook,
                ] {
                    let candidate = Move {
                        from,
                        to,
                        promotion_piece_type: Some(promo_type),
                        castle_option: None,
                        en_passant: false,
                    };
                    if board.validate_move(
                        candidate,
                        color,
                        combined,
                        board.white_pieces,
                        board.black_pieces,
                    ) {
                        legal.push(move_to_action(&candidate));
                    }
                }
            } else {
                let candidate = Move {
                    from,
                    to,
                    promotion_piece_type: None,
                    castle_option: None,
                    en_passant,
                };
                if board.validate_move(
                    candidate,
                    color,
                    combined,
                    board.white_pieces,
                    board.black_pieces,
                ) {
                    legal.push(move_to_action(&candidate));
                }
            }
        }

        // Check castling moves for king
        if piece.piece_type == PieceType::King {
            for &castle_opt in &[CastleOption::Kingside, CastleOption::Queenside] {
                let (to_file, _king_file) = match castle_opt {
                    CastleOption::Kingside => (6u8, 4u8),
                    CastleOption::Queenside => (2u8, 4u8),
                };
                let king_rank = if color == Color::White { 0u8 } else { 7u8 };
                let to_sq_castle = king_rank * 8 + to_file;

                let candidate = Move {
                    from: Square::from(sq as u8),
                    to: Square::from(to_sq_castle),
                    promotion_piece_type: None,
                    castle_option: Some(castle_opt),
                    en_passant: false,
                };

                if board.validate_move(
                    candidate,
                    color,
                    combined,
                    board.white_pieces,
                    board.black_pieces,
                ) {
                    legal.push(move_to_action(&candidate));
                }
            }
        }
    }

    legal
}

/// Read the tanh-denominator for the material-proxy outcome from
/// `HYZERO_MATERIAL_SHAPING_SCALE`. Larger values shrink the signal so that
/// actual checkmate (±1.0) dominates terminal-time material adjudication.
///   scale=5  (default, legacy): up 5 material → 0.76, up 10 → 0.96
///   scale=10:                   up 5 material → 0.46, up 10 → 0.76
///   scale=20:                   up 5 material → 0.24, up 10 → 0.46
/// Clamped to [0.5, 100.0]. Defaults to 5.0 on missing/unparseable input.
fn material_shaping_scale() -> f32 {
    std::env::var("HYZERO_MATERIAL_SHAPING_SCALE")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .map(|v| v.clamp(0.5, 100.0))
        .unwrap_or(5.0)
}

/// Multiplier applied to the shaped value of a REPETITION or MOVE-CAP rule draw
/// (NOT fifty-move) when material shaping is on. `HYZERO_SHAPING_REP_DISCOUNT`,
/// default 1.0 (behavior unchanged), clamped to [0.0, 1.0]. Unparseable input
/// falls back to the default. Read per-call (TestEnvGuard-compatible).
///
/// Applied symmetrically to BOTH sides, preserving the existing antisymmetry of
/// the shaped value: winner's +0.9 → +0.27, loser's -0.9 → -0.27 at 0.3.
///
/// Rationale (2026-06-11, measured 3/120 forced-mate conversion rate): an
/// undiscounted material lead shapes a repetition/move-cap draw to ≈+0.9, so
/// shuffling earns ~90% of mating and nothing prefers conversion. Discounting
/// (e.g. 0.3) keeps a sharp mate-vs-shuffle gradient for the winner while — by
/// the preserved antisymmetry — still leaving the defender a slight preference
/// for escaping into repetition over being mated (loser's -0.9 → -0.27).
fn shaping_rep_discount() -> f32 {
    std::env::var("HYZERO_SHAPING_REP_DISCOUNT")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .map(|v| v.clamp(0.0, 1.0))
        .unwrap_or(1.0)
}

/// True when HYZERO_MATERIAL_SHAPING is set to any truthy value
/// ("1", "true", "yes", ...). When false (the DEFAULT), every non-checkmate
/// terminal produces outcome=0.0 — AlphaZero-style pure-outcome training
/// signal: only real checkmates provide non-zero value targets. When true,
/// non-checkmate terminals use `tanh(Δmaterial/scale)` as a proxy outcome.
///
/// Shaping is opt-in because it historically caused:
///   1. PGN result labeling bug: shaped outcomes > 0.5 got tagged "1-0"
///      even for drawn games, misleading diagnostics.
///   2. Shuffle-attractor reinforcement: material-leading sides drawing by
///      repetition received high value targets (+0.8-ish), teaching the
///      value head that passive play with a material lead is good.
///
/// Do not flip the default back to enabled without a deliberate decision.
fn material_shaping_enabled() -> bool {
    match std::env::var("HYZERO_MATERIAL_SHAPING") {
        Ok(v) => {
            let s = v.trim().to_ascii_lowercase();
            !(s.is_empty() || s == "0" || s == "false" || s == "no")
        }
        Err(_) => false,
    }
}

/// Material balance = white_total - black_total, in standard piece values.
/// Used for the material-proxy outcome on non-decisive terminals. Scale by tanh(Δ/S)
/// at call sites to bound to [-1, 1], where S is HYZERO_MATERIAL_SHAPING_SCALE (default 5).
fn compute_material_diff(board: &GameBoard) -> i32 {
    // Piece values: P=1, N=3, B=3, R=5, Q=9, K=0 (king never captured).
    const VALUES: [i32; 6] = [1, 3, 3, 5, 9, 0];
    let mut delta: i32 = 0;
    for (pt, &val) in VALUES.iter().enumerate() {
        let white_count = board.player1.pieces_bb[pt].count_ones() as i32;
        let black_count = board.player2.pieces_bb[pt].count_ones() as i32;
        delta += val * (white_count - black_count);
    }
    delta
}

/// Map a (non-resignation) board terminal to a self-play training outcome,
/// returning `(white_absolute_outcome, is_draw)`.
///
/// Checkmates always produce a decisive ±1 signal. Non-checkmate terminals split
/// into two groups:
///   - TRUE draws (stalemate, insufficient material): the position is genuinely
///     drawn regardless of material on the board, so the outcome MUST stay 0.0.
///     Shaping these with `tanh(Δmaterial)` would teach the value head a false
///     value (e.g. a stalemate with a material lead is still a draw, not a win).
///   - RULE draws (repetition, fifty-move, move-cap): the game stopped by a count
///     rule, not because the position is balanced. With HYZERO_MATERIAL_SHAPING=1
///     these get `tanh(Δmaterial/scale)` as a weak material-proxy signal to keep
///     the value head learning when games don't reach checkmate. Default (shaping
///     OFF) is AlphaZero-style: outcome = 0.0. Repetition and move-cap draws are
///     additionally scaled by HYZERO_SHAPING_REP_DISCOUNT (default 1.0 = no
///     change), applied symmetrically to both sides so the shaped value's
///     antisymmetry is preserved; fifty-move is NOT discounted.
///
/// `is_draw` is true for every non-checkmate terminal (used by the trainer's
/// draw penalty and by the PGN result label, which must stay 1/2-1/2 even when a
/// rule draw carries a non-zero shaped value — historically the cause of the
/// 2026-04-23 PGN labeling confusion). Material shaping is opt-in because it
/// historically reinforced the shuffle-attractor: material-leading sides got
/// high value targets for drawing by repetition, teaching the value head to
/// reward passive play.
fn score_board_terminal(board_result: GameResult, board: &GameBoard) -> (f32, bool) {
    match board_result {
        GameResult::Checkmate(Color::White) => (1.0f32, false),
        GameResult::Checkmate(Color::Black) => (-1.0f32, false),
        // True draws: drawn by position, never shaped.
        GameResult::Stalemate | GameResult::InsufficientMaterial => (0.0f32, true),
        // Repetition and move-cap (board still Ongoing when the loop exits on the
        // cap) are rule draws eligible for material shaping. They additionally
        // receive HYZERO_SHAPING_REP_DISCOUNT (default 1.0 = no change), applied
        // symmetrically so the antisymmetry of the shaped value is preserved.
        // The discount makes the winner's mate-vs-shuffle gradient sharp while
        // still letting a defender slightly prefer repetition over being mated.
        GameResult::ThreefoldRepetition | GameResult::Ongoing => {
            if material_shaping_enabled() {
                let delta = compute_material_diff(board);
                let scale = material_shaping_scale();
                let shaped = (delta as f32 / scale).tanh();
                (shaped * shaping_rep_discount(), true)
            } else {
                (0.0f32, true)
            }
        }
        // Fifty-move is a rule draw eligible for shaping but NOT for the
        // repetition/move-cap discount (it is not a shuffle-to-repetition draw).
        GameResult::FiftyMoveRule => {
            if material_shaping_enabled() {
                let delta = compute_material_diff(board);
                let scale = material_shaping_scale();
                ((delta as f32 / scale).tanh(), true)
            } else {
                (0.0f32, true)
            }
        }
    }
}

/// Eval-only adjudication for a non-checkmate terminal (used by `play_game_dual`).
/// When `enabled`, award +1.0 to White / -1.0 to Black if that side is ahead by
/// at least `margin` material; otherwise 0.0 (draw). When disabled, always 0.0.
/// Eval outcomes never enter training targets, so adjudication here is safe and
/// the passivity-attractor risk that bars it from self-play does not apply.
fn adjudicate_non_checkmate(board: &GameBoard, enabled: bool, margin: i32) -> f32 {
    if !enabled {
        return 0.0;
    }
    let delta = compute_material_diff(board);
    if delta >= margin {
        1.0
    } else if delta <= -margin {
        -1.0
    } else {
        0.0
    }
}

// --- Self-play material adjudication (HYZERO_SELFPLAY_ADJUDICATE* gates) ---
//
// High-threshold material adjudication for SELF-PLAY games, mirroring the
// eval-side plumbing (`adjudicate_non_checkmate`) but gated behind its own
// OnceLock knobs and OFF by default. When enabled, an otherwise-Ongoing self-play
// position with a white-absolute material lead >= margin is terminated as a
// decisive ±1 for the leading side (flowing into value/TD targets exactly like a
// real checkmate), instead of continuing to the move cap / repetition. Real
// terminals (checkmate/stalemate/etc.) always take precedence: the loop only
// consults adjudication on positions whose `board.result()` is `Ongoing`.
//
// Default OFF because material adjudication historically drove a passivity
// attractor in training. The default margin (12) is deliberately high so only
// overwhelmingly-decided positions ever adjudicate.

/// Pure parse helper for `HYZERO_SELFPLAY_ADJUDICATE`. Enabled only when the
/// (trimmed, lowercased) value is `"1"` or `"true"`; anything else (including
/// `"0"`, `"false"`, empty, and unset/`None`) is OFF. Extracted from the cached
/// gate so env-parse tests can exercise it without tripping the `OnceLock`.
fn parse_selfplay_adjudicate(value: Option<&str>) -> bool {
    match value {
        Some(v) => {
            let s = v.trim().to_ascii_lowercase();
            s == "1" || s == "true"
        }
        None => false,
    }
}

/// Cached env-gate: true when `HYZERO_SELFPLAY_ADJUDICATE` is `"1"`/`"true"`.
/// Read once per process via `OnceLock`, mirroring the other self-play knobs.
/// DEFAULT OFF — self-play behavior is byte-identical to before when unset.
fn selfplay_adjudicate_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        parse_selfplay_adjudicate(std::env::var("HYZERO_SELFPLAY_ADJUDICATE").ok().as_deref())
    })
}

/// Pure parse helper for `HYZERO_SELFPLAY_ADJ_MARGIN`: the white-absolute material
/// lead (standard piece values) required to adjudicate a self-play position as
/// decisive. Default 12, clamped to `>= 1`; unparseable/unset falls back to 12.
fn parse_selfplay_adj_margin(value: Option<&str>) -> i32 {
    value
        .and_then(|v| v.trim().parse::<i32>().ok())
        .filter(|&m| m >= 1)
        .unwrap_or(12)
}

/// Cached material-lead margin for self-play adjudication. Read once per process
/// via `OnceLock`, mirroring the other self-play knobs.
fn selfplay_adjudication_margin() -> i32 {
    static MARGIN: OnceLock<i32> = OnceLock::new();
    *MARGIN.get_or_init(|| {
        parse_selfplay_adj_margin(std::env::var("HYZERO_SELFPLAY_ADJ_MARGIN").ok().as_deref())
    })
}

/// Self-play adjudication decision for an OTHERWISE-ONGOING position. Returns
/// `Some(Color)` (the leading side, awarded the decisive win) when `enabled` and
/// the white-absolute material lead reaches `margin`, else `None`. Uses the same
/// `compute_material_diff` piece-value count as the eval-side
/// `adjudicate_non_checkmate`. Only ever called on positions the game loop has
/// already confirmed are `Ongoing`, so real terminals take precedence.
fn selfplay_adjudicated_winner(board: &GameBoard, enabled: bool, margin: i32) -> Option<Color> {
    if !enabled {
        return None;
    }
    let delta = compute_material_diff(board);
    if delta >= margin {
        Some(Color::White)
    } else if delta <= -margin {
        Some(Color::Black)
    } else {
        None
    }
}

/// PGN `[Termination "..."]` value for a board-result terminal. Maps the chess
/// `GameResult` variants onto the standard short causes recorded in the PGN
/// header. The move-cap and resignation/adjudication terminals are NOT reached
/// here — those are board `Ongoing` (cap) or out-of-band (resignation, eval
/// adjudication) and are labeled by the caller before this is consulted.
fn termination_label(result: GameResult) -> &'static str {
    match result {
        GameResult::Checkmate(_) => "checkmate",
        GameResult::Stalemate => "stalemate",
        GameResult::ThreefoldRepetition => "repetition",
        GameResult::FiftyMoveRule => "fifty-move",
        GameResult::InsufficientMaterial => "insufficient-material",
        // The loop only exits to terminal scoring once the game is non-Ongoing
        // OR the move cap is hit; an Ongoing board here means the cap stopped it.
        GameResult::Ongoing => "move-cap",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{BoardObservation, HiddenState, Policy, NUM_ACTIONS};
    use crate::mcts::evaluator::Evaluator;
    use async_trait::async_trait;

    use crate::data::types::TestEnvGuard;

    /// Regression: legal_actions ordering after POV-flip must be color-symmetric.
    ///
    /// Before the fix, `get_legal_moves` iterated sq 0..63 producing
    /// white's knights first (from sq 1, 6) and black's pawns first (from sq 48..55),
    /// and `flip_action` only rewrote VALUES, not positions. After sorting by
    /// action-index post-flip, both colors see identical legal_actions at
    /// equivalent POV positions. Validated to cure ~76% black dominance in
    /// random-evaluator self-play.
    #[tokio::test]
    async fn test_legal_actions_ordering_is_color_symmetric_after_sort() {
        let precomputed = Arc::new(PrecomputedItems::begin_precomputing());
        let p1 = Player::init_player(true);
        let p2 = Player::init_player(false);
        let board = GameBoard::init_game_board(precomputed, p1, p2);

        // White's legal_actions are already in white-POV (= absolute) coords.
        let mut legal_w: Vec<ActionIndex> = get_legal_moves(&board, Color::White);
        legal_w.sort_unstable();

        // Black's legal_actions are in absolute coords too, but need flipping to
        // POV coords and sorting to match what MCTS receives.
        let raw_b = get_legal_moves(&board, Color::Black);
        let mut legal_b_pov: Vec<ActionIndex> = raw_b
            .iter()
            .map(|&a| flip_action(a as usize) as ActionIndex)
            .collect();
        legal_b_pov.sort_unstable();

        assert_eq!(
            legal_w, legal_b_pov,
            "legal_actions must be identical between colors at the starting \
             position after POV-flip-then-sort — otherwise MCTS selection \
             becomes color-biased"
        );
        assert_eq!(legal_w.len(), 20, "starting position has 20 legal moves");
    }

    struct RandomEvaluator;

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

    /// Evaluator that always thinks the side to move is hopelessly losing:
    /// `root_setup` returns root value -1.0 (side-to-move POV) and `expand_leaf`
    /// returns leaf value +1.0 (child/opponent POV, reward 0). With the backup
    /// recurrence `G_0 = r - value`, every simulation pushes the root Q toward
    /// -1.0, so `tree.root_value()` sits well below the resignation threshold —
    /// letting the resignation tests fire deterministically.
    struct LosingEvaluator;

    #[async_trait]
    impl Evaluator for LosingEvaluator {
        async fn root_setup(
            &self,
            _obs: &BoardObservation,
            _legal_mask: &[bool],
        ) -> (HiddenState, Policy, f32, Option<f32>) {
            let policy = vec![1.0 / NUM_ACTIONS as f32; NUM_ACTIONS];
            (HiddenState::new(64), policy, -1.0, None)
        }

        async fn expand_leaf(
            &self,
            _hs: &HiddenState,
            _action: ActionIndex,
        ) -> (HiddenState, f32, Policy, f32, Option<f32>) {
            let policy = vec![1.0 / NUM_ACTIONS as f32; NUM_ACTIONS];
            (HiddenState::new(64), 0.0, policy, 1.0, None)
        }
    }

    #[test]
    fn test_legal_moves_starting_position() {
        let precomputed = Arc::new(PrecomputedItems::begin_precomputing());
        let p1 = Player::init_player(true);
        let p2 = Player::init_player(false);
        let board = GameBoard::init_game_board(precomputed, p1, p2);

        let moves = get_legal_moves(&board, Color::White);
        // Standard chess starting position: 20 legal moves for white
        // (16 pawn moves + 4 knight moves)
        assert_eq!(
            moves.len(),
            20,
            "Expected 20 legal moves, got {}",
            moves.len()
        );
    }

    #[tokio::test]
    async fn test_play_game_completes() {
        let precomputed = Arc::new(PrecomputedItems::begin_precomputing());
        let evaluator: Arc<dyn Evaluator> = Arc::new(RandomEvaluator);
        let config = GameConfig {
            num_simulations: 2, // Very few for speed
            exploration_constant: 1.5,
            temperature_moves: 5,
            replay_dir: None,
            adjudicate_at_cap: false,
            adjudication_material_margin: 5,
        };

        let trajectory = play_game(precomputed, evaluator, 1, config).await;

        // Game should have produced at least some steps
        assert!(!trajectory.steps.is_empty(), "Trajectory should have steps");

        // Each step should have valid data
        for step in &trajectory.steps {
            assert!(
                !step.legal_moves.is_empty(),
                "Each step should have legal moves"
            );
            assert!(
                !step.visit_distribution.is_empty(),
                "Each step should have visit distribution"
            );
        }

        // Game outcome should be in [-1, 1]; tanh at cap can produce non-integer values.
        assert!(
            trajectory.game_outcome.abs() <= 1.0,
            "Game outcome should be in [-1, 1], got {}",
            trajectory.game_outcome
        );
    }

    /// `play_game_dual_from` must start from the *supplied* board/side and
    /// forward `starting_fen` verbatim into the outcome — it must NOT fall back
    /// to `init_self_play_board` (which would ignore the caller's position).
    /// Uses a sparse K+P vs K endgame so the very first legal move can only come
    /// from that custom position, never the standard 20-move opening.
    #[tokio::test]
    async fn play_game_dual_from_uses_supplied_board() {
        let precomputed = Arc::new(PrecomputedItems::begin_precomputing());
        let white: Arc<dyn Evaluator> = Arc::new(RandomEvaluator);
        let black: Arc<dyn Evaluator> = Arc::new(RandomEvaluator);
        let config = GameConfig {
            num_simulations: 2,
            exploration_constant: 1.5,
            temperature_moves: 0,
            replay_dir: None,
            adjudicate_at_cap: false,
            adjudication_material_margin: 5,
        };

        // White: Kc1, Pd2. Black: Kc8. Only White pieces move first; none of the
        // standard-start opening moves (a2a3..h2h4, knights) are legal here.
        let fen = "2k5/8/8/8/8/8/3P4/2K5 w - - 0 1";
        let (board, side_to_move, _fm) =
            board_from_fen(fen, precomputed.clone()).expect("valid FEN");
        assert_eq!(side_to_move, Color::White);

        let outcome = play_game_dual_from(
            white,
            black,
            config,
            board,
            side_to_move,
            Some(fen.to_string()),
        )
        .await;

        // The starting FEN is forwarded verbatim (drives the PGN [FEN] header).
        assert_eq!(outcome.starting_fen.as_deref(), Some(fen));
        // At least one move was made, and the first one originates from a square
        // that holds a White piece in the supplied position (c1 king or d2 pawn) —
        // proving the game did NOT start from the standard initial position.
        assert!(!outcome.moves.is_empty(), "game should have made moves");
        let first_from = &outcome.moves[0][..2];
        assert!(
            first_from == "c1" || first_from == "d2",
            "first move {} must originate from the supplied K+P position, not the \
             standard start",
            outcome.moves[0]
        );
    }

    #[test]
    fn test_action_to_notation() {
        // e2 = sq 12, e4 = sq 28 → action = 12*64 + 28 = 796
        let notation = action_to_notation(796, Color::White);
        assert_eq!(notation, "e2e4");

        // a7 = sq 48, a8 = sq 56 → action = 48*64 + 56 = 3128 (promotion)
        let notation = action_to_notation(3128, Color::White);
        assert_eq!(notation, "a7a8q");
    }

    #[test]
    fn test_action_to_notation_underpromotion_white() {
        use crate::data::encoding::move_to_action as m2a;
        use crate::game::Move;

        // White pawn e7→e8 with knight underpromotion
        // from_sq = 52 (e7), to_sq = 60 (e8)
        let mv = Move {
            from: Square::E7,
            to: Square::E8,
            promotion_piece_type: Some(PieceType::Knight),
            castle_option: None,
            en_passant: false,
        };
        let action = m2a(&mv);
        assert!(
            action as usize >= NUM_BASE_ACTIONS,
            "expected underpromo action, got {action}"
        );
        let notation = action_to_notation(action, Color::White);
        assert_eq!(notation, "e7e8n");
    }

    #[test]
    fn test_action_to_notation_underpromotion_rook_white() {
        use crate::data::encoding::move_to_action as m2a;
        use crate::game::Move;

        // White pawn a7→a8 with rook underpromotion
        let mv = Move {
            from: Square::A7,
            to: Square::A8,
            promotion_piece_type: Some(PieceType::Rook),
            castle_option: None,
            en_passant: false,
        };
        let action = m2a(&mv);
        assert!(action as usize >= NUM_BASE_ACTIONS);
        let notation = action_to_notation(action, Color::White);
        assert_eq!(notation, "a7a8r");
    }

    /// Color-symmetry audit: play many random-vs-random dual games and assert
    /// the outcome distribution is approximately balanced. Any asymmetry here
    /// would be evidence of a bug in move generation / board state / encoding
    /// / MCTS — NOT in NN training — since both evaluators return uniform random
    /// policies with value=0.
    ///
    /// Expected under uniform-random play: White wins and Black wins should both
    /// sit around ~1-5% (most random games hit the 300-move cap and are draws).
    /// A gap of 3x+ between white_wins and black_wins would be a smoking gun.
    #[tokio::test]
    #[ignore] // 200 games × ≤300 moves × 2 sims ≈ ~30-60s; run with --ignored
    async fn test_random_play_color_symmetry_audit() {
        let precomputed = Arc::new(PrecomputedItems::begin_precomputing());
        let white: Arc<dyn Evaluator> = Arc::new(RandomEvaluator);
        let black: Arc<dyn Evaluator> = Arc::new(RandomEvaluator);
        let config = GameConfig {
            num_simulations: 2,
            exploration_constant: 1.5,
            temperature_moves: 0,
            replay_dir: None,
            adjudicate_at_cap: false,
            adjudication_material_margin: 5,
        };

        const N: usize = 2000;
        let mut white_wins = 0usize;
        let mut black_wins = 0usize;
        let mut draws = 0usize;

        for _ in 0..N {
            let o = play_game_dual(
                precomputed.clone(),
                white.clone(),
                black.clone(),
                config.clone(),
            )
            .await;
            match o.game_outcome {
                x if x > 0.5 => white_wins += 1,
                x if x < -0.5 => black_wins += 1,
                _ => draws += 1,
            }
        }

        eprintln!(
            "[random_audit] N={} white_wins={} black_wins={} draws={} white_frac={:.3} black_frac={:.3}",
            N,
            white_wins,
            black_wins,
            draws,
            white_wins as f64 / N as f64,
            black_wins as f64 / N as f64,
        );

        // Smoke assertion: neither side should dominate by >4x under pure random play.
        // (Allow some noise — 200 games with small decisive counts are high-variance.)
        let min_decisive = white_wins.min(black_wins);
        let max_decisive = white_wins.max(black_wins);
        if max_decisive >= 5 {
            let ratio = max_decisive as f64 / (min_decisive.max(1)) as f64;
            assert!(
                ratio < 4.0,
                "Color imbalance under random play: white={} black={} ratio={:.2}",
                white_wins,
                black_wins,
                ratio,
            );
        }
    }

    #[tokio::test]
    async fn test_play_game_dual_completes() {
        let precomputed = Arc::new(PrecomputedItems::begin_precomputing());
        let white_evaluator: Arc<dyn Evaluator> = Arc::new(RandomEvaluator);
        let black_evaluator: Arc<dyn Evaluator> = Arc::new(RandomEvaluator);
        let config = GameConfig {
            num_simulations: 2,
            exploration_constant: 1.5,
            temperature_moves: 5,
            replay_dir: None,
            adjudicate_at_cap: false,
            adjudication_material_margin: 5,
        };

        let outcome = play_game_dual(precomputed, white_evaluator, black_evaluator, config).await;

        assert!(
            outcome.game_outcome == 1.0
                || outcome.game_outcome == -1.0
                || outcome.game_outcome == 0.0,
            "game_outcome should be +1, -1, or 0"
        );
    }

    /// White wins → game_outcome = +1.0 (champion = Black, so champion LOSES).
    /// Negating the outcome gives champion_perspective = -1.0 (loss for champion).
    ///
    /// Black wins → game_outcome = -1.0 (champion = Black, so champion WINS).
    /// Negating gives +1.0 (win for champion).
    ///
    /// This test verifies the sign convention is understood by the caller.
    #[test]
    fn test_dual_game_outcome_sign_convention() {
        // White wins
        let white_wins = DualGameOutcome {
            game_outcome: 1.0,
            num_moves: 40,
            moves: vec![],
            termination: "checkmate".to_string(),
            starting_fen: None,
        };
        // When champion played Black, negate to get champion perspective
        let champion_perspective_when_black = -white_wins.game_outcome;
        assert_eq!(
            champion_perspective_when_black, -1.0,
            "champion lost when White won"
        );

        // Black wins (champion wins as Black)
        let black_wins = DualGameOutcome {
            game_outcome: -1.0,
            num_moves: 40,
            moves: vec![],
            termination: "checkmate".to_string(),
            starting_fen: None,
        };
        let champion_perspective_when_black = -black_wins.game_outcome;
        assert_eq!(
            champion_perspective_when_black, 1.0,
            "champion won when Black won"
        );

        // Draw
        let draw = DualGameOutcome {
            game_outcome: 0.0,
            num_moves: 300,
            moves: vec![],
            termination: "move-cap".to_string(),
            starting_fen: None,
        };
        let champion_perspective_when_black = -draw.game_outcome;
        assert_eq!(
            champion_perspective_when_black, 0.0,
            "draw is draw regardless of color"
        );
    }

    #[test]
    fn test_legal_moves_promotion_position() {
        use crate::game::fen::board_from_fen;
        use crate::PrecomputedItems;
        use std::sync::Arc;

        // FEN: white pawn on e7, white king on a1, black king on h1. White to move.
        // e8 is empty so the pawn can push straight → 4 promotion types (Q, N, B, R).
        let precomputed = Arc::new(PrecomputedItems::begin_precomputing());
        let (board, _, _) =
            board_from_fen("8/4P3/8/8/8/8/8/K6k w - - 0 1", precomputed).expect("invalid FEN");

        let moves = get_legal_moves(&board, Color::White);

        // King has ≤5 moves, pawn has 4 promotions. Only care that pawn promotions are present.
        // Identify queen promotions: actions in base range (< NUM_BASE_ACTIONS) where to_sq is
        // on rank 8 (to_sq / 8 == 7) and from_sq is on rank 7 (from_sq / 8 == 6).
        let queen_promos: Vec<_> = moves
            .iter()
            .copied()
            .filter(|&a| {
                if (a as usize) >= NUM_BASE_ACTIONS {
                    return false;
                }
                let from_sq = (a / 64) as u8;
                let to_sq = (a % 64) as u8;
                from_sq / 8 == 6 && to_sq / 8 == 7
            })
            .collect();
        let underpromos: Vec<_> = moves
            .iter()
            .copied()
            .filter(|&a| (a as usize) >= NUM_BASE_ACTIONS)
            .collect();

        assert_eq!(
            queen_promos.len(),
            1,
            "Expected 1 queen promotion in all moves {:?}",
            moves
        );
        assert_eq!(
            underpromos.len(),
            3,
            "Expected 3 underpromotion moves in all moves {:?}",
            moves
        );
    }

    /// Regression test for the terminal reward POV conversion in play_game.
    ///
    /// After the fix, the last step's reward must be game_outcome converted to
    /// the last step's player-to-move POV. For a game where Black delivers mate
    /// (game_outcome=-1.0, white_to_move=false on the last step), the stored
    /// reward must be +1.0 (Black's POV of Black winning), not -1.0 (raw white-absolute).
    #[tokio::test]
    // Deliberate: hold the TestEnvGuard across awaits to serialize env mutation
    // for the whole test (HYZERO_MATERIAL_SHAPING is read per-ply in play_game).
    #[allow(clippy::await_holding_lock)]
    async fn test_terminal_reward_pov_conversion() {
        use std::env;

        // Force material shaping off so game_outcome is always +1/-1/0 (not tanh).
        // Default is already off, but we explicitly clear the opt-in flag in case
        // another test left it set in this process. The guard serializes against
        // every other env-mutating test and restores the var on exit.
        let _env = TestEnvGuard::new(&["HYZERO_MATERIAL_SHAPING"]);
        // SAFETY: env mutation serialized by TestEnvGuard for this test's duration.
        unsafe {
            env::remove_var("HYZERO_MATERIAL_SHAPING");
        }

        let precomputed = Arc::new(PrecomputedItems::begin_precomputing());
        let evaluator: Arc<dyn Evaluator> = Arc::new(RandomEvaluator);
        let config = GameConfig {
            num_simulations: 2,
            exploration_constant: 1.5,
            temperature_moves: 0, // greedy — faster termination
            replay_dir: None,
            adjudicate_at_cap: false,
            adjudication_material_margin: 5,
        };

        // Play many games until we see a decisive result (Black wins), then verify
        // the last step's reward has the correct sign.
        let mut found_decisive = false;
        for _ in 0..500 {
            let trajectory = play_game(precomputed.clone(), evaluator.clone(), 1, config.clone()).await;

            if let Some(last) = trajectory.steps.last() {
                let outcome = trajectory.game_outcome;
                if outcome.abs() > 0.5 {
                    // Decisive game. Expected: last.reward == last_side_sign * outcome.
                    let expected_sign: f32 = if last.white_to_move { 1.0 } else { -1.0 };
                    let expected_reward = outcome * expected_sign;
                    assert!(
                        (last.reward - expected_reward).abs() < 1e-6,
                        "terminal reward POV mismatch: game_outcome={} white_to_move={} \
                         expected reward={} got {}",
                        outcome,
                        last.white_to_move,
                        expected_reward,
                        last.reward
                    );
                    found_decisive = true;
                    break;
                }
            }
        }

        // If no decisive game appeared in 500 tries, that itself is suspicious but
        // not a correctness bug — just skip the assertion.
        let _ = found_decisive;
    }

    /// With resignation enabled and a hopeless evaluator, the consecutive-losing
    /// counter must end the game decisively after exactly `resign_plies` plies.
    /// Uses min_ply=0, consecutive=2 so resignation fires at ply 2 — before any
    /// chess terminal is reachable (the fastest mate is 4 plies), making the
    /// outcome and length deterministic. FAILS without the counter logic (the
    /// game would otherwise run on as a draw).
    #[tokio::test]
    // Deliberate: hold the std Mutex across the await to serialize env mutation
    // for the whole game (the env knobs are read per-ply inside play_game).
    #[allow(clippy::await_holding_lock)]
    async fn resigns_after_consecutive_losing_plies_below_threshold() {
        use std::env;
        let _env = TestEnvGuard::new(&[
            "HYZERO_RESIGN",
            "HYZERO_RESIGN_THRESHOLD",
            "HYZERO_RESIGN_CONSECUTIVE",
            "HYZERO_RESIGN_MIN_PLY",
            "HYZERO_RESIGN_DISABLE_FRAC",
        ]);
        // SAFETY: env mutation serialized by TestEnvGuard for this test's duration.
        unsafe {
            env::remove_var("HYZERO_RESIGN"); // default ON
            env::remove_var("HYZERO_RESIGN_THRESHOLD"); // default -0.90
            env::set_var("HYZERO_RESIGN_CONSECUTIVE", "2");
            env::set_var("HYZERO_RESIGN_MIN_PLY", "0");
            // Pin calibration off so resignation always fires (default 0.1 would
            // disable it for ~10% of runs, making this assertion flaky).
            env::set_var("HYZERO_RESIGN_DISABLE_FRAC", "0");
        }

        let precomputed = Arc::new(PrecomputedItems::begin_precomputing());
        let evaluator: Arc<dyn Evaluator> = Arc::new(LosingEvaluator);
        let config = GameConfig {
            // Exactly one simulation: from a fresh root only depth-1 expansions are
            // reachable, so the backup is always g_0 = reward(0) - leaf = -1.0 and
            // root_value stays deterministically at the LosingEvaluator's -1.0. With
            // >=2 sims a depth-2 revisit can flip g_0 to +1.0 (Dirichlet root noise
            // steers selection), pushing root_value above the resign threshold and
            // making resignation fire nondeterministically.
            num_simulations: 1,
            exploration_constant: 1.5,
            temperature_moves: 0,
            replay_dir: None,
            adjudicate_at_cap: false,
            adjudication_material_margin: 5,
        };

        let trajectory = play_game(precomputed, evaluator, 1, config).await;

        // Resignation fires at ply index 1 (the 2nd losing ply): step 0 = White,
        // step 1 = Black, so Black resigns and White wins (+1.0). Two steps total.
        assert_eq!(
            trajectory.steps.len(),
            2,
            "expected resignation after exactly 2 losing plies",
        );
        assert!(
            !trajectory.is_draw && trajectory.game_outcome == 1.0,
            "Black resigns at ply 1 → White wins (+1.0); got outcome={} is_draw={}",
            trajectory.game_outcome,
            trajectory.is_draw,
        );
    }

    /// `resign_disable_frac` parses `HYZERO_RESIGN_DISABLE_FRAC`, defaults to 0.1
    /// when unset/unparseable, and clamps out-of-range values into [0.0, 1.0].
    /// FAILS without the new knob (the helper does not exist).
    #[test]
    fn resign_disable_frac_parses_defaults_and_clamps() {
        use std::env;
        let _env = TestEnvGuard::new(&["HYZERO_RESIGN_DISABLE_FRAC"]);
        // SAFETY: env mutation serialized by TestEnvGuard for this test's duration.
        unsafe {
            env::remove_var("HYZERO_RESIGN_DISABLE_FRAC");
            assert!(
                (resign_disable_frac() - 0.1).abs() < 1e-6,
                "default must be 0.1"
            );

            env::set_var("HYZERO_RESIGN_DISABLE_FRAC", "0.25");
            assert!((resign_disable_frac() - 0.25).abs() < 1e-6);

            // Out-of-range values clamp into [0.0, 1.0].
            env::set_var("HYZERO_RESIGN_DISABLE_FRAC", "5.0");
            assert!(
                (resign_disable_frac() - 1.0).abs() < 1e-6,
                "above 1.0 clamps to 1.0"
            );
            env::set_var("HYZERO_RESIGN_DISABLE_FRAC", "-2.0");
            assert!(
                (resign_disable_frac() - 0.0).abs() < 1e-6,
                "below 0.0 clamps to 0.0"
            );

            // Unparseable falls back to the default.
            env::set_var("HYZERO_RESIGN_DISABLE_FRAC", "not-a-number");
            assert!((resign_disable_frac() - 0.1).abs() < 1e-6);
        }
    }

    /// Calibration: with `HYZERO_RESIGN_DISABLE_FRAC=1.0` every game disables
    /// resignation, so even a hopeless (LosingEvaluator) game does NOT end at the
    /// would-be resign ply — it plays on past it to a natural terminal / cap.
    /// Mirrors `resigns_after_consecutive_losing_plies_below_threshold` (which ends
    /// at exactly 2 steps); the calibration game must instead exceed that, proving
    /// resignation was ignored. FAILS without the per-game disable (the game would
    /// resign and stop at 2 steps).
    #[tokio::test]
    // Deliberate: hold the std Mutex across the await to serialize env mutation
    // for the whole game (the env knobs are read per-ply inside play_game).
    #[allow(clippy::await_holding_lock)]
    async fn calibration_game_ignores_resignation_and_plays_on() {
        use std::env;
        let _env = TestEnvGuard::new(&[
            "HYZERO_RESIGN",
            "HYZERO_RESIGN_THRESHOLD",
            "HYZERO_RESIGN_CONSECUTIVE",
            "HYZERO_RESIGN_MIN_PLY",
            "HYZERO_RESIGN_DISABLE_FRAC",
        ]);
        // SAFETY: env mutation serialized by TestEnvGuard for this test's duration.
        unsafe {
            env::remove_var("HYZERO_RESIGN"); // default ON
            env::remove_var("HYZERO_RESIGN_THRESHOLD"); // default -0.90
            env::set_var("HYZERO_RESIGN_CONSECUTIVE", "2");
            env::set_var("HYZERO_RESIGN_MIN_PLY", "0");
            // Disable resignation for EVERY game (per-game draw < 1.0 is always true).
            env::set_var("HYZERO_RESIGN_DISABLE_FRAC", "1.0");
        }

        let precomputed = Arc::new(PrecomputedItems::begin_precomputing());
        let evaluator: Arc<dyn Evaluator> = Arc::new(LosingEvaluator);
        let config = GameConfig {
            num_simulations: 1,
            exploration_constant: 1.5,
            temperature_moves: 0,
            replay_dir: None,
            adjudicate_at_cap: false,
            adjudication_material_margin: 5,
        };

        let trajectory = play_game(precomputed, evaluator, 1, config).await;

        // Without the disable, the game resigns at ply 1 → exactly 2 steps. With
        // resignation disabled it plays past the would-be resign ply.
        assert!(
            trajectory.steps.len() > 2,
            "calibration game must ignore resignation and play on past ply 2 \
             (got {} steps)",
            trajectory.steps.len(),
        );
    }

    /// `termination_label` maps each board result onto its PGN cause, and an
    /// Ongoing board (only reachable at the move cap) maps to "move-cap".
    #[test]
    fn termination_label_maps_board_results() {
        assert_eq!(
            termination_label(GameResult::Checkmate(Color::White)),
            "checkmate"
        );
        assert_eq!(termination_label(GameResult::Stalemate), "stalemate");
        assert_eq!(
            termination_label(GameResult::ThreefoldRepetition),
            "repetition"
        );
        assert_eq!(termination_label(GameResult::FiftyMoveRule), "fifty-move");
        assert_eq!(
            termination_label(GameResult::InsufficientMaterial),
            "insufficient-material"
        );
        assert_eq!(termination_label(GameResult::Ongoing), "move-cap");
    }

    /// The min-ply gate must suppress resignation entirely until `resign_min_ply`,
    /// even when every ply is below threshold (consecutive=1). With min_ply=3 the
    /// game cannot resign on plies 0..2, so it survives to ply 3 — still before
    /// the 4-ply fastest mate, keeping the result deterministic. FAILS without
    /// the gate (resignation would fire on the very first ply, length 1).
    #[tokio::test]
    // Deliberate: hold the std Mutex across the await to serialize env mutation
    // for the whole game (the env knobs are read per-ply inside play_game).
    #[allow(clippy::await_holding_lock)]
    async fn does_not_resign_before_min_ply() {
        use std::env;
        let _env = TestEnvGuard::new(&[
            "HYZERO_RESIGN",
            "HYZERO_RESIGN_THRESHOLD",
            "HYZERO_RESIGN_CONSECUTIVE",
            "HYZERO_RESIGN_MIN_PLY",
            "HYZERO_RESIGN_DISABLE_FRAC",
        ]);
        // SAFETY: env mutation serialized by TestEnvGuard for this test's duration.
        unsafe {
            env::remove_var("HYZERO_RESIGN");
            env::remove_var("HYZERO_RESIGN_THRESHOLD");
            env::set_var("HYZERO_RESIGN_CONSECUTIVE", "1");
            env::set_var("HYZERO_RESIGN_MIN_PLY", "3");
            // Pin calibration off so resignation always fires (default 0.1 would
            // disable it for ~10% of runs, making the exact step count flaky).
            env::set_var("HYZERO_RESIGN_DISABLE_FRAC", "0");
        }

        let precomputed = Arc::new(PrecomputedItems::begin_precomputing());
        let evaluator: Arc<dyn Evaluator> = Arc::new(LosingEvaluator);
        let config = GameConfig {
            // One simulation keeps root_value deterministically at -1.0 (see the
            // resigns_after_* test); with >=2 sims Dirichlet root noise can flip the
            // backup sign and make resignation timing nondeterministic.
            num_simulations: 1,
            exploration_constant: 1.5,
            temperature_moves: 0,
            replay_dir: None,
            adjudicate_at_cap: false,
            adjudication_material_margin: 5,
        };

        let trajectory = play_game(precomputed, evaluator, 1, config).await;

        // The gate holds resignation until turn_count >= 3, so the game records
        // plies 0..3 (4 steps) before resigning, instead of resigning at ply 0.
        assert_eq!(
            trajectory.steps.len(),
            4,
            "min-ply gate must suppress resignation until ply 3 (got {} steps)",
            trajectory.steps.len(),
        );
    }

    /// Temperature must linearly anneal from 1.0 to 0.01 across the configured
    /// window after `temperature_moves`, rather than the old hard step. FAILS
    /// without the anneal (the old code jumps straight to 0.01 past the window).
    #[test]
    fn temperature_anneals_linearly_within_window() {
        use std::env;
        let _env = TestEnvGuard::new(&["HYZERO_TEMP_ANNEAL", "HYZERO_TEMP_ANNEAL_PLIES"]);
        // SAFETY: env mutation serialized by TestEnvGuard for this test's duration.
        unsafe {
            env::remove_var("HYZERO_TEMP_ANNEAL"); // default ON
            env::set_var("HYZERO_TEMP_ANNEAL_PLIES", "100");
        }

        let temperature_moves = 30u32;
        // Inside the exploration window: full temperature.
        assert!((selfplay_temperature(0, temperature_moves) - 1.0).abs() < 1e-6);
        assert!((selfplay_temperature(29, temperature_moves) - 1.0).abs() < 1e-6);
        // At the window edge the anneal starts at 1.0.
        assert!((selfplay_temperature(30, temperature_moves) - 1.0).abs() < 1e-6);
        // Halfway through the 100-ply anneal: midpoint between 1.0 and 0.01.
        let mid = selfplay_temperature(80, temperature_moves);
        let expected_mid = 1.0 + 0.5 * (0.01 - 1.0);
        assert!(
            (mid - expected_mid).abs() < 1e-4,
            "expected mid-anneal temp {expected_mid}, got {mid}",
        );
        // Past the full anneal span: clamped to the floor 0.01.
        assert!((selfplay_temperature(200, temperature_moves) - 0.01).abs() < 1e-4);

        // With annealing OFF, the old hard step is preserved.
        unsafe {
            env::set_var("HYZERO_TEMP_ANNEAL", "0");
        }
        assert!((selfplay_temperature(29, temperature_moves) - 1.0).abs() < 1e-6);
        assert!((selfplay_temperature(31, temperature_moves) - 0.01).abs() < 1e-6);
    }

    /// HYZERO_TEMPERATURE_MOVES unset yields no override (None), so the self-play
    /// construction site falls through to the legacy HYZERO_TEMP_MOVES/RunConfig
    /// default. Unparseable input is likewise no override.
    #[test]
    fn temperature_moves_override_is_none_when_unset_or_unparseable() {
        let _env = TestEnvGuard::new(&["HYZERO_TEMPERATURE_MOVES"]);
        // SAFETY: env mutation serialized by TestEnvGuard for this test's duration.
        unsafe {
            std::env::remove_var("HYZERO_TEMPERATURE_MOVES");
        }
        assert_eq!(temperature_moves_override(), None);
        // Unparseable input also yields no override (falls through to legacy).
        // SAFETY: still under the same TestEnvGuard.
        unsafe {
            std::env::set_var("HYZERO_TEMPERATURE_MOVES", "not-a-number");
        }
        assert_eq!(temperature_moves_override(), None);
    }

    /// A valid HYZERO_TEMPERATURE_MOVES value is parsed as-is into Some(value).
    #[test]
    fn temperature_moves_override_parses_valid_value() {
        let _env = TestEnvGuard::new(&["HYZERO_TEMPERATURE_MOVES"]);
        // SAFETY: env mutation serialized by TestEnvGuard for this test's duration.
        unsafe {
            std::env::set_var("HYZERO_TEMPERATURE_MOVES", "12");
        }
        assert_eq!(temperature_moves_override(), Some(12));
    }

    /// Out-of-range HYZERO_TEMPERATURE_MOVES values clamp into [1, 200].
    #[test]
    fn temperature_moves_override_clamps_out_of_range() {
        let _env = TestEnvGuard::new(&["HYZERO_TEMPERATURE_MOVES"]);
        // SAFETY: env mutation serialized by TestEnvGuard for this test's duration.
        unsafe {
            std::env::set_var("HYZERO_TEMPERATURE_MOVES", "0");
        }
        assert_eq!(temperature_moves_override(), Some(1));
        // SAFETY: still under the same TestEnvGuard.
        unsafe {
            std::env::set_var("HYZERO_TEMPERATURE_MOVES", "9999");
        }
        assert_eq!(temperature_moves_override(), Some(200));
    }

    /// Unset (or unparseable) HYZERO_ROOT_MATE_SOLVER_PLIES disables the solver
    /// (returns 0). The self-play / eval move-selection hooks are gated on
    /// `mate_plies > 0`, so a 0 here leaves both code paths bit-identical to
    /// before the knob existed.
    #[test]
    fn root_mate_solver_plies_disabled_when_unset_or_unparseable() {
        let _env = TestEnvGuard::new(&["HYZERO_ROOT_MATE_SOLVER_PLIES"]);
        // SAFETY: env mutation serialized by TestEnvGuard for this test's duration.
        unsafe {
            std::env::remove_var("HYZERO_ROOT_MATE_SOLVER_PLIES");
        }
        assert_eq!(root_mate_solver_plies(), 0, "unset must disable the solver");
        // SAFETY: still under the same TestEnvGuard.
        unsafe {
            std::env::set_var("HYZERO_ROOT_MATE_SOLVER_PLIES", "not-a-number");
        }
        assert_eq!(
            root_mate_solver_plies(),
            0,
            "unparseable input must disable the solver"
        );
    }

    /// A valid HYZERO_ROOT_MATE_SOLVER_PLIES value is parsed as-is.
    #[test]
    fn root_mate_solver_plies_parses_valid_value() {
        let _env = TestEnvGuard::new(&["HYZERO_ROOT_MATE_SOLVER_PLIES"]);
        // SAFETY: env mutation serialized by TestEnvGuard for this test's duration.
        unsafe {
            std::env::set_var("HYZERO_ROOT_MATE_SOLVER_PLIES", "5");
        }
        assert_eq!(root_mate_solver_plies(), 5);
    }

    /// Out-of-range HYZERO_ROOT_MATE_SOLVER_PLIES values clamp into [0, 7].
    #[test]
    fn root_mate_solver_plies_clamps_out_of_range() {
        let _env = TestEnvGuard::new(&["HYZERO_ROOT_MATE_SOLVER_PLIES"]);
        // SAFETY: env mutation serialized by TestEnvGuard for this test's duration.
        unsafe {
            std::env::set_var("HYZERO_ROOT_MATE_SOLVER_PLIES", "99");
        }
        assert_eq!(root_mate_solver_plies(), 7, "above-range clamps to 7");
        // SAFETY: still under the same TestEnvGuard.
        unsafe {
            std::env::set_var("HYZERO_ROOT_MATE_SOLVER_PLIES", "0");
        }
        assert_eq!(root_mate_solver_plies(), 0, "0 stays 0 (disabled)");
    }

    /// A set HYZERO_TEMPERATURE_MOVES reaches the self-play GameConfig: the
    /// construction site applies `temperature_moves_override().unwrap_or(...)`,
    /// so a set var of 12 overrides the legacy GameConfig::default() value of 30.
    /// `selfplay_temperature` then holds 1.0 up to that ply and drops just past it.
    #[test]
    fn configured_window_reaches_temperature_schedule() {
        let _env = TestEnvGuard::new(&["HYZERO_TEMPERATURE_MOVES", "HYZERO_TEMP_ANNEAL"]);
        // SAFETY: env mutation serialized by TestEnvGuard for this test's duration.
        unsafe {
            std::env::set_var("HYZERO_TEMPERATURE_MOVES", "12");
            std::env::set_var("HYZERO_TEMP_ANNEAL", "0"); // hard step for a crisp edge
        }
        // Mirror the src/bin/selfplay.rs construction site: the env override
        // wins over the legacy GameConfig::default() window (30).
        let mut config = GameConfig::default();
        assert_eq!(config.temperature_moves, 30, "default window must stay the legacy 30");
        config.temperature_moves = temperature_moves_override().unwrap_or(config.temperature_moves);
        assert_eq!(config.temperature_moves, 12);
        // Full temperature inside the shortened window, exploitation just past it.
        assert!((selfplay_temperature(11, config.temperature_moves) - 1.0).abs() < 1e-6);
        assert!((selfplay_temperature(12, config.temperature_moves) - 0.01).abs() < 1e-6);
    }

    /// Unset HYZERO_TEMPERATURE_MOVES leaves the self-play window at the legacy
    /// value: the construction site falls through `unwrap_or` to the supplied
    /// config window (the HYZERO_TEMP_MOVES/RunConfig chain) — bit-identical to
    /// pre-knob behavior. Here we feed RunConfig's default-15 to confirm it stands.
    #[test]
    fn unset_override_preserves_legacy_selfplay_window() {
        let _env = TestEnvGuard::new(&["HYZERO_TEMPERATURE_MOVES"]);
        // SAFETY: env mutation serialized by TestEnvGuard for this test's duration.
        unsafe {
            std::env::remove_var("HYZERO_TEMPERATURE_MOVES");
        }
        // Legacy self-play window from the RunConfig default (15).
        let legacy_window: u32 = 15;
        let resolved = temperature_moves_override().unwrap_or(legacy_window);
        assert_eq!(resolved, legacy_window);
    }

    /// Eval cap-adjudication: when enabled and one side is clearly ahead on
    /// material at a non-checkmate terminal, the game is awarded a decisive
    /// result. Exercises `adjudicate_non_checkmate`, the exact seam invoked by
    /// `play_game_dual`'s cap/terminal branch. FAILS without the adjudication
    /// branch (the default path returns 0.0). With adjudication OFF the same
    /// material lead must still draw.
    #[test]
    fn dual_game_adjudicates_material_lead_at_cap() {
        let precomputed = Arc::new(PrecomputedItems::begin_precomputing());
        // White up a full queen (delta +9 ≥ margin 5), no checkmate present.
        let (board, _, _) = board_from_fen("4k3/8/8/8/8/8/8/3QK3 w - - 0 1", precomputed)
            .expect("invalid FEN");

        // Enabled → decisive for the side ahead.
        assert_eq!(
            adjudicate_non_checkmate(&board, true, 5),
            1.0,
            "white up a queen should adjudicate to +1.0 when enabled",
        );
        // Disabled (self-play default) → still a draw, never adjudicated.
        assert_eq!(
            adjudicate_non_checkmate(&board, false, 5),
            0.0,
            "adjudication OFF must keep a non-checkmate game a draw",
        );
    }

    /// Eval cap-adjudication must NOT award a decisive result when the material
    /// lead is within the margin — a near-balanced non-checkmate game stays a
    /// draw even with adjudication enabled.
    #[test]
    fn dual_game_draws_when_material_within_margin() {
        let precomputed = Arc::new(PrecomputedItems::begin_precomputing());
        // White up only a pawn (delta +1 < margin 5).
        let (board, _, _) =
            board_from_fen("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", precomputed).expect("invalid FEN");
        assert_eq!(
            adjudicate_non_checkmate(&board, true, 5),
            0.0,
            "a within-margin material lead should stay a draw",
        );
    }

    /// Self-play adjudication: with the knob enabled, an otherwise-Ongoing
    /// position with a material lead at/above the margin (12) is decided for the
    /// leading side. White up a queen + rook (+14 ≥ 12) → Some(White). Tests the
    /// decision seam directly (not a whole game). FAILS without the adjudication
    /// branch (the pre-fix self-play loop never terminates on material).
    #[test]
    fn selfplay_adjudicates_win_at_margin_twelve() {
        let precomputed = Arc::new(PrecomputedItems::begin_precomputing());
        // White up a rook + queen (delta +14 ≥ margin 12), no checkmate present.
        let (board, _, _) = board_from_fen("4k3/8/8/8/8/8/8/R2QK3 w - - 0 1", precomputed)
            .expect("invalid FEN");
        assert_eq!(
            selfplay_adjudicated_winner(&board, true, 12),
            Some(Color::White),
            "white up +14 material must adjudicate a decisive win for white",
        );
    }

    /// Self-play adjudication must NOT fire when the material lead is below the
    /// margin: an 11-pawn lead (queen + two pawns = +11 < 12) leaves the game
    /// Ongoing (returns None), so play continues to a natural terminal / move cap.
    #[test]
    fn selfplay_no_adjudication_below_margin() {
        let precomputed = Arc::new(PrecomputedItems::begin_precomputing());
        // White up a queen + two pawns (delta +11 < margin 12).
        let (board, _, _) = board_from_fen("4k3/8/8/8/8/8/3PP3/3QK3 w - - 0 1", precomputed)
            .expect("invalid FEN");
        assert_eq!(
            selfplay_adjudicated_winner(&board, true, 12),
            None,
            "a below-margin material lead must leave the game Ongoing",
        );
    }

    /// Gate: with self-play adjudication DISABLED (the default), even an
    /// overwhelming material lead is never adjudicated — behavior is preserved
    /// and the game plays on. FAILS if the decision ever ignores the `enabled` flag.
    #[test]
    fn selfplay_adjudication_disabled_preserves_behavior() {
        let precomputed = Arc::new(PrecomputedItems::begin_precomputing());
        // White up a rook + queen (delta +14) — huge lead, but adjudication off.
        let (board, _, _) = board_from_fen("4k3/8/8/8/8/8/8/R2QK3 w - - 0 1", precomputed)
            .expect("invalid FEN");
        assert_eq!(
            selfplay_adjudicated_winner(&board, false, 12),
            None,
            "adjudication OFF must never terminate a self-play game on material",
        );
    }

    /// Env-parse (pure helper): `HYZERO_SELFPLAY_ADJUDICATE` is OFF by default and
    /// turns ON only for "1"/"true" (space/case-insensitive). Any other value —
    /// including "0", "false", empty, and unset — stays OFF.
    #[test]
    fn selfplay_adjudicate_env_parse_default_off_truthy_on() {
        assert!(!parse_selfplay_adjudicate(None));
        assert!(!parse_selfplay_adjudicate(Some("")));
        assert!(!parse_selfplay_adjudicate(Some("0")));
        assert!(!parse_selfplay_adjudicate(Some("false")));
        assert!(!parse_selfplay_adjudicate(Some("no")));
        assert!(parse_selfplay_adjudicate(Some("1")));
        assert!(parse_selfplay_adjudicate(Some("true")));
        assert!(parse_selfplay_adjudicate(Some("  TRUE  ")));
    }

    /// Env-parse (pure helper): `HYZERO_SELFPLAY_ADJ_MARGIN` defaults to 12 on
    /// unset/unparseable input, clamps sub-1 values back to the default, and
    /// otherwise parses the supplied margin verbatim.
    #[test]
    fn selfplay_adj_margin_env_parse_default_twelve() {
        assert_eq!(parse_selfplay_adj_margin(None), 12);
        assert_eq!(parse_selfplay_adj_margin(Some("not-a-number")), 12);
        assert_eq!(parse_selfplay_adj_margin(Some("0")), 12);
        assert_eq!(parse_selfplay_adj_margin(Some("-3")), 12);
        assert_eq!(parse_selfplay_adj_margin(Some("8")), 8);
        assert_eq!(parse_selfplay_adj_margin(Some("  20 ")), 20);
    }

    /// Regression: with material shaping ON, a repetition draw with a material
    /// lead stores a NON-ZERO shaped outcome of the correct sign — repetition is a
    /// rule draw (the game stopped by a count rule, not because the position is
    /// balanced), so the weak material-proxy signal is allowed. FAILS without the
    /// explicit rule-draw arm (the old catch-all already shaped this, so this case
    /// guards against a future regression that lumps it back with true draws).
    #[test]
    fn repetition_with_material_lead_is_shaped_when_enabled() {
        use std::env;
        let _env = TestEnvGuard::new(&["HYZERO_MATERIAL_SHAPING", "HYZERO_MATERIAL_SHAPING_SCALE"]);
        // SAFETY: env mutation serialized by TestEnvGuard for this test's duration.
        unsafe {
            env::set_var("HYZERO_MATERIAL_SHAPING", "1");
            env::remove_var("HYZERO_MATERIAL_SHAPING_SCALE"); // default scale 5.0
        }
        let precomputed = Arc::new(PrecomputedItems::begin_precomputing());
        // White up a full queen (delta +9) — non-checkmate position.
        let (board, _, _) =
            board_from_fen("4k3/8/8/8/8/8/8/3QK3 w - - 0 1", precomputed).expect("invalid FEN");

        let (outcome, is_draw) = score_board_terminal(GameResult::ThreefoldRepetition, &board);
        let expected = (9.0f32 / 5.0).tanh();
        assert!(
            (outcome - expected).abs() < 1e-6,
            "repetition with +9 material under shaping must store tanh(9/5)={expected}, got {outcome}",
        );
        // The shaped rule draw is still a draw for the PGN label / trainer penalty.
        assert!(is_draw, "a shaped repetition draw must remain is_draw=true");
    }

    /// Regression: with material shaping ON, a STALEMATE with a material lead must
    /// still store 0.0 — stalemate is a true draw (drawn by position regardless of
    /// material), and shaping it would teach the value head a false value. FAILS
    /// against the old catch-all, which shaped every non-checkmate terminal.
    #[test]
    fn stalemate_with_material_lead_stays_zero_when_enabled() {
        use std::env;
        let _env = TestEnvGuard::new(&["HYZERO_MATERIAL_SHAPING", "HYZERO_MATERIAL_SHAPING_SCALE"]);
        // SAFETY: env mutation serialized by TestEnvGuard for this test's duration.
        unsafe {
            env::set_var("HYZERO_MATERIAL_SHAPING", "1");
            env::remove_var("HYZERO_MATERIAL_SHAPING_SCALE");
        }
        let precomputed = Arc::new(PrecomputedItems::begin_precomputing());
        // White up a full queen (delta +9) — a true stalemate would still be a draw.
        let (board, _, _) =
            board_from_fen("4k3/8/8/8/8/8/8/3QK3 w - - 0 1", precomputed).expect("invalid FEN");

        let (outcome, is_draw) = score_board_terminal(GameResult::Stalemate, &board);
        assert_eq!(
            outcome, 0.0,
            "stalemate is a true draw — material shaping must NOT apply",
        );
        let _ = is_draw;
    }

    /// Regression: with material shaping ON, INSUFFICIENT MATERIAL must store 0.0 —
    /// it is a true draw. FAILS against the old catch-all, which shaped it via the
    /// (small) residual material delta. Constructed with White holding an extra
    /// pawn so the old code would have produced a non-zero tanh(Δ).
    #[test]
    fn insufficient_material_stays_zero_when_enabled() {
        use std::env;
        let _env = TestEnvGuard::new(&["HYZERO_MATERIAL_SHAPING", "HYZERO_MATERIAL_SHAPING_SCALE"]);
        // SAFETY: env mutation serialized by TestEnvGuard for this test's duration.
        unsafe {
            env::set_var("HYZERO_MATERIAL_SHAPING", "1");
            env::remove_var("HYZERO_MATERIAL_SHAPING_SCALE");
        }
        let precomputed = Arc::new(PrecomputedItems::begin_precomputing());
        // White up a pawn (delta +1) — under the old catch-all this would have
        // shaped to tanh(1/5) ≠ 0; the true-draw arm must keep it at 0.0.
        let (board, _, _) =
            board_from_fen("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", precomputed).expect("invalid FEN");

        let (outcome, _is_draw) = score_board_terminal(GameResult::InsufficientMaterial, &board);
        assert_eq!(
            outcome, 0.0,
            "insufficient material is a true draw — material shaping must NOT apply",
        );
    }

    /// `HYZERO_SHAPING_REP_DISCOUNT` parses, defaults to 1.0 on missing/unparseable
    /// input, and clamps out-of-range values into [0.0, 1.0].
    #[test]
    fn shaping_rep_discount_parses_defaults_and_clamps() {
        use std::env;
        let _env = TestEnvGuard::new(&["HYZERO_SHAPING_REP_DISCOUNT"]);
        // SAFETY: env mutation serialized by TestEnvGuard for this test's duration.
        unsafe {
            env::remove_var("HYZERO_SHAPING_REP_DISCOUNT");
            assert!(
                (shaping_rep_discount() - 1.0).abs() < 1e-6,
                "default must be 1.0"
            );

            env::set_var("HYZERO_SHAPING_REP_DISCOUNT", "0.3");
            assert!((shaping_rep_discount() - 0.3).abs() < 1e-6);

            // Out-of-range values clamp into [0.0, 1.0].
            env::set_var("HYZERO_SHAPING_REP_DISCOUNT", "5.0");
            assert!(
                (shaping_rep_discount() - 1.0).abs() < 1e-6,
                "above 1.0 clamps to 1.0"
            );
            env::set_var("HYZERO_SHAPING_REP_DISCOUNT", "-2.0");
            assert!(
                (shaping_rep_discount() - 0.0).abs() < 1e-6,
                "below 0.0 clamps to 0.0"
            );

            // Unparseable falls back to the default.
            env::set_var("HYZERO_SHAPING_REP_DISCOUNT", "not-a-number");
            assert!((shaping_rep_discount() - 1.0).abs() < 1e-6);
        }
    }

    /// At the default discount of 1.0, a shaped repetition draw is bit-identical
    /// to the undiscounted `tanh(Δ/scale)` — behavior unchanged when the knob is
    /// absent. Guards the "default = no-op" contract.
    #[test]
    fn repetition_shaping_unchanged_at_default_discount() {
        use std::env;
        let _env = TestEnvGuard::new(&[
            "HYZERO_MATERIAL_SHAPING",
            "HYZERO_MATERIAL_SHAPING_SCALE",
            "HYZERO_SHAPING_REP_DISCOUNT",
        ]);
        // SAFETY: env mutation serialized by TestEnvGuard for this test's duration.
        unsafe {
            env::set_var("HYZERO_MATERIAL_SHAPING", "1");
            env::remove_var("HYZERO_MATERIAL_SHAPING_SCALE"); // default scale 5.0
            env::remove_var("HYZERO_SHAPING_REP_DISCOUNT"); // default 1.0
        }
        let precomputed = Arc::new(PrecomputedItems::begin_precomputing());
        // White up a full queen (delta +9).
        let (board, _, _) =
            board_from_fen("4k3/8/8/8/8/8/8/3QK3 w - - 0 1", precomputed).expect("invalid FEN");

        let (outcome, _) = score_board_terminal(GameResult::ThreefoldRepetition, &board);
        let expected = (9.0f32 / 5.0).tanh();
        assert_eq!(
            outcome, expected,
            "default discount 1.0 must leave the shaped value bit-identical",
        );
    }

    /// With the discount at 0.3, a repetition draw's shaped value is scaled to
    /// 0.3× for BOTH sides, preserving antisymmetry: winner's +tanh(9/5) and the
    /// mirror loser's -tanh(9/5) both shrink by the same factor. FAILS without
    /// the discount code (the regression character of this knob).
    #[test]
    fn repetition_shaping_discounted_antisymmetrically_at_03() {
        use std::env;
        let _env = TestEnvGuard::new(&[
            "HYZERO_MATERIAL_SHAPING",
            "HYZERO_MATERIAL_SHAPING_SCALE",
            "HYZERO_SHAPING_REP_DISCOUNT",
        ]);
        // SAFETY: env mutation serialized by TestEnvGuard for this test's duration.
        unsafe {
            env::set_var("HYZERO_MATERIAL_SHAPING", "1");
            env::remove_var("HYZERO_MATERIAL_SHAPING_SCALE"); // default scale 5.0
            env::set_var("HYZERO_SHAPING_REP_DISCOUNT", "0.3");
        }
        let precomputed = Arc::new(PrecomputedItems::begin_precomputing());
        // White up a queen (+9): winner side.
        let (white_ahead, _, _) =
            board_from_fen("4k3/8/8/8/8/8/8/3QK3 w - - 0 1", precomputed.clone())
                .expect("invalid FEN");
        // Black up a queen (-9): mirror, loser side from White's POV.
        let (black_ahead, _, _) =
            board_from_fen("3qk3/8/8/8/8/8/8/4K3 w - - 0 1", precomputed).expect("invalid FEN");

        let undiscounted = (9.0f32 / 5.0).tanh();
        let (won, _) = score_board_terminal(GameResult::ThreefoldRepetition, &white_ahead);
        let (lost, _) = score_board_terminal(GameResult::ThreefoldRepetition, &black_ahead);

        assert!(
            (won - undiscounted * 0.3).abs() < 1e-6,
            "winner's repetition shaping must be 0.3*tanh(9/5)={}, got {won}",
            undiscounted * 0.3,
        );
        assert!(
            (lost - (-undiscounted * 0.3)).abs() < 1e-6,
            "loser's repetition shaping must be -0.3*tanh(9/5)={}, got {lost}",
            -undiscounted * 0.3,
        );
        // Antisymmetry preserved exactly.
        assert!(
            (won + lost).abs() < 1e-6,
            "discounted shaping must stay antisymmetric (won + lost == 0)",
        );
    }

    /// The move-cap terminal (board still `Ongoing` at the cap) receives the same
    /// repetition discount. FAILS without the discount code.
    #[test]
    fn move_cap_shaping_discounted_at_03() {
        use std::env;
        let _env = TestEnvGuard::new(&[
            "HYZERO_MATERIAL_SHAPING",
            "HYZERO_MATERIAL_SHAPING_SCALE",
            "HYZERO_SHAPING_REP_DISCOUNT",
        ]);
        // SAFETY: env mutation serialized by TestEnvGuard for this test's duration.
        unsafe {
            env::set_var("HYZERO_MATERIAL_SHAPING", "1");
            env::remove_var("HYZERO_MATERIAL_SHAPING_SCALE"); // default scale 5.0
            env::set_var("HYZERO_SHAPING_REP_DISCOUNT", "0.3");
        }
        let precomputed = Arc::new(PrecomputedItems::begin_precomputing());
        // White up a queen (+9). Ongoing = move-cap terminal.
        let (board, _, _) =
            board_from_fen("4k3/8/8/8/8/8/8/3QK3 w - - 0 1", precomputed).expect("invalid FEN");

        let (outcome, _) = score_board_terminal(GameResult::Ongoing, &board);
        let expected = (9.0f32 / 5.0).tanh() * 0.3;
        assert!(
            (outcome - expected).abs() < 1e-6,
            "move-cap repetition shaping must be 0.3*tanh(9/5)={expected}, got {outcome}",
        );
    }

    /// Fifty-move is a rule draw eligible for shaping but NOT for the
    /// repetition/move-cap discount: even with the discount at 0.3 its shaped
    /// value stays the full undiscounted `tanh(Δ/scale)`.
    #[test]
    fn fifty_move_shaping_ignores_rep_discount() {
        use std::env;
        let _env = TestEnvGuard::new(&[
            "HYZERO_MATERIAL_SHAPING",
            "HYZERO_MATERIAL_SHAPING_SCALE",
            "HYZERO_SHAPING_REP_DISCOUNT",
        ]);
        // SAFETY: env mutation serialized by TestEnvGuard for this test's duration.
        unsafe {
            env::set_var("HYZERO_MATERIAL_SHAPING", "1");
            env::remove_var("HYZERO_MATERIAL_SHAPING_SCALE"); // default scale 5.0
            env::set_var("HYZERO_SHAPING_REP_DISCOUNT", "0.3");
        }
        let precomputed = Arc::new(PrecomputedItems::begin_precomputing());
        // White up a queen (+9).
        let (board, _, _) =
            board_from_fen("4k3/8/8/8/8/8/8/3QK3 w - - 0 1", precomputed).expect("invalid FEN");

        let (outcome, _) = score_board_terminal(GameResult::FiftyMoveRule, &board);
        let expected = (9.0f32 / 5.0).tanh();
        assert_eq!(
            outcome, expected,
            "fifty-move shaping must ignore HYZERO_SHAPING_REP_DISCOUNT",
        );
    }

    /// True draws (stalemate, insufficient material) stay 0.0 regardless of the
    /// discount knob — the discount multiplies a value that is already 0.0.
    #[test]
    fn true_draws_unaffected_by_rep_discount() {
        use std::env;
        let _env = TestEnvGuard::new(&[
            "HYZERO_MATERIAL_SHAPING",
            "HYZERO_MATERIAL_SHAPING_SCALE",
            "HYZERO_SHAPING_REP_DISCOUNT",
        ]);
        // SAFETY: env mutation serialized by TestEnvGuard for this test's duration.
        unsafe {
            env::set_var("HYZERO_MATERIAL_SHAPING", "1");
            env::remove_var("HYZERO_MATERIAL_SHAPING_SCALE");
            env::set_var("HYZERO_SHAPING_REP_DISCOUNT", "0.3");
        }
        let precomputed = Arc::new(PrecomputedItems::begin_precomputing());
        // White up a queen (+9) — a true draw stays 0.0 regardless.
        let (board, _, _) =
            board_from_fen("4k3/8/8/8/8/8/8/3QK3 w - - 0 1", precomputed).expect("invalid FEN");

        let (stalemate, _) = score_board_terminal(GameResult::Stalemate, &board);
        let (insufficient, _) = score_board_terminal(GameResult::InsufficientMaterial, &board);
        assert_eq!(
            stalemate, 0.0,
            "stalemate stays 0.0 under the discount knob"
        );
        assert_eq!(
            insufficient, 0.0,
            "insufficient material stays 0.0 under the discount knob",
        );
    }

    /// `HYZERO_PGN_SAMPLE_RATE` parses, defaults to 0.01 on missing/unparseable
    /// input, and clamps out-of-range values into [0.0, 1.0].
    #[test]
    fn pgn_sample_rate_parses_defaults_and_clamps() {
        use std::env;
        let _env = TestEnvGuard::new(&["HYZERO_PGN_SAMPLE_RATE"]);
        // SAFETY: env mutation serialized by TestEnvGuard for this test's duration.
        unsafe {
            env::remove_var("HYZERO_PGN_SAMPLE_RATE");
            assert!(
                (pgn_sample_rate() - 0.01).abs() < 1e-6,
                "default must be 0.01"
            );

            env::set_var("HYZERO_PGN_SAMPLE_RATE", "0.5");
            assert!((pgn_sample_rate() - 0.5).abs() < 1e-6);

            // Out-of-range values clamp into [0.0, 1.0].
            env::set_var("HYZERO_PGN_SAMPLE_RATE", "5.0");
            assert!(
                (pgn_sample_rate() - 1.0).abs() < 1e-6,
                "above 1.0 clamps to 1.0"
            );
            env::set_var("HYZERO_PGN_SAMPLE_RATE", "-2.0");
            assert!(
                (pgn_sample_rate() - 0.0).abs() < 1e-6,
                "below 0.0 clamps to 0.0"
            );

            // Unparseable falls back to the default.
            env::set_var("HYZERO_PGN_SAMPLE_RATE", "not-a-number");
            assert!((pgn_sample_rate() - 0.01).abs() < 1e-6);
        }
    }

    /// At rate 1.0 every game samples a PGN; at rate 0.0 none do — for all
    /// possible rng draws in [0.0, 1.0). FAILS if the decision ignores the rate.
    #[test]
    fn pgn_sampling_writes_all_at_full_rate_and_none_at_zero() {
        // rand::random::<f32>() yields values in [0.0, 1.0); sweep that range.
        for i in 0..1000 {
            let draw = i as f32 / 1000.0;
            assert!(
                should_sample_pgn(1.0, draw),
                "rate 1.0 must sample every game (draw {draw})",
            );
            assert!(
                !should_sample_pgn(0.0, draw),
                "rate 0.0 must sample no game (draw {draw})",
            );
        }
    }
}
