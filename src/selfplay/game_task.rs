use std::collections::VecDeque;
use std::io::{BufWriter, Write};
use std::sync::{Arc, Mutex, OnceLock};

use crate::data::{
    board_to_snapshot, encode_board, move_to_action, ActionIndex, BoardSnapshot, GameTrajectory,
    StepRecord, NUM_BASE_ACTIONS,
};
use crate::data::encoding::flip_action;
use crate::game::board::GameResult;
use crate::game::{GameBoard, Move, Player};
use crate::mcts::evaluator::Evaluator;
use crate::mcts::tree::{MCTSConfig, MCTSTree};
use crate::{BitIterator, CastleOption, Color, PieceType, PrecomputedItems, Square};

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

/// Configuration for a self-play game.
#[derive(Debug, Clone)]
pub struct GameConfig {
    pub num_simulations: u32,
    pub exploration_constant: f32,
    /// Use temperature=1.0 for the first N moves, then near 0.
    pub temperature_moves: u32,
}

impl Default for GameConfig {
    fn default() -> Self {
        Self {
            num_simulations: 800,
            exploration_constant: 1.5,
            temperature_moves: 30,
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
}

/// Play a game with two distinct evaluators (challenger = White, champion = Black).
///
/// Returns the White-perspective outcome. Callers that want to count champion wins
/// when champion played Black must **negate** `game_outcome`.
///
/// No `GameTrajectory` is produced — eval games don't go to the training buffer.
pub async fn play_game_dual(
    precomputed: Arc<PrecomputedItems>,
    white_evaluator: Arc<dyn Evaluator>,
    black_evaluator: Arc<dyn Evaluator>,
    config: GameConfig,
) -> DualGameOutcome {
    let player1 = Player::init_player(true);
    let player2 = Player::init_player(false);
    let mut board = GameBoard::init_game_board(precomputed.clone(), player1, player2);

    let mut turn_count: usize = 0;
    let mut side_to_move = Color::White;
    let mut history: VecDeque<BoardSnapshot> = VecDeque::with_capacity(7);
    let mut moves: Vec<String> = Vec::new();

    const MAX_GAME_LENGTH: usize = 300;

    let mcts_config = MCTSConfig {
        num_simulations: config.num_simulations,
        exploration_constant: config.exploration_constant,
        add_root_noise: false, // eval: no exploration noise
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
        let legal_actions: Vec<ActionIndex> = if side_to_move == Color::Black {
            raw_legal
                .iter()
                .map(|&a| flip_action(a as usize) as ActionIndex)
                .collect()
        } else {
            raw_legal
        };

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

        let (hidden_state, policy, value) = evaluator.root_setup(&observation, &legal_mask).await;

        let mut tree = MCTSTree::new(
            hidden_state,
            &policy,
            value,
            legal_actions.clone(),
            mcts_config.clone(),
        );
        tree.run_simulations(evaluator.as_ref()).await;

        // Use near-greedy temperature for eval games (no exploration needed).
        let selected_action = tree.select_action(0.01);

        // Flip action back to absolute coordinates before applying to the board.
        let absolute_action = if side_to_move == Color::Black {
            flip_action(selected_action as usize) as ActionIndex
        } else {
            selected_action
        };

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

    let game_outcome = match board.result() {
        GameResult::Checkmate(Color::White) => 1.0,
        GameResult::Checkmate(Color::Black) => -1.0,
        _ => 0.0,
    };

    DualGameOutcome {
        game_outcome,
        num_moves: turn_count,
        moves,
    }
}

/// Play a complete game using MCTS, producing a GameTrajectory for training.
pub async fn play_game(
    precomputed: Arc<PrecomputedItems>,
    evaluator: Arc<dyn Evaluator>,
    model_version: u64,
    config: GameConfig,
) -> GameTrajectory {
    let player1 = Player::init_player(true);
    let player2 = Player::init_player(false);
    let mut board = GameBoard::init_game_board(precomputed.clone(), player1, player2);

    let mut steps: Vec<StepRecord> = Vec::new();
    let mut turn_count: usize = 0;
    let mut side_to_move = Color::White;
    // History buffer for encode_board: stores up to 7 past snapshots (oldest first).
    let mut history: VecDeque<BoardSnapshot> = VecDeque::with_capacity(7);

    const MAX_GAME_LENGTH: usize = 300;

    let mcts_config = MCTSConfig {
        num_simulations: config.num_simulations,
        exploration_constant: config.exploration_constant,
        add_root_noise: true, // self-play: inject exploration
    };

    let mut moves: Vec<String> = Vec::new();

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
        let legal_actions: Vec<ActionIndex> = if side_to_move == Color::Black {
            raw_legal
                .iter()
                .map(|&a| flip_action(a as usize) as ActionIndex)
                .collect()
        } else {
            raw_legal
        };

        if legal_actions.is_empty() {
            break;
        }

        // Build legal mask: true for each action that is legal in this position.
        let mut legal_mask = vec![false; crate::data::NUM_ACTIONS];
        for &a in &legal_actions {
            legal_mask[a as usize] = true;
        }

        // Root setup: encode board into latent space
        let (hidden_state, policy, value) = evaluator.root_setup(&observation, &legal_mask).await;

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
        let visit_distribution = tree.extract_visit_distribution();
        let root_value = tree.root_value();

        // Write one-line summary to logs/mcts_summary.log when HYZERO_MCTS_TRACE is set.
        if mcts_summary_enabled() {
            trace_summary(model_version, turn_count, &legal_actions, &policy, &visit_distribution);
        }

        // Select action based on temperature
        let temperature = if turn_count < config.temperature_moves as usize {
            1.0
        } else {
            0.01
        };
        // selected_action is in current-player (flipped) coordinate space for Black.
        let selected_action = tree.select_action(temperature);

        // Flip action back to absolute coordinates before applying to the board.
        let absolute_action = if side_to_move == Color::Black {
            flip_action(selected_action as usize) as ActionIndex
        } else {
            selected_action
        };

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
    }

    // Game outcome.
    // Checkmates produce ±1 signal (strong). Non-decisive terminals (stalemate, repetition,
    // 50-move rule, 300-move cap) use tanh(Δmaterial/5) as a weak material-proxy signal to
    // keep the value head learning when games don't reach checkmate. Adjudication is NOT
    // re-enabled — it was the primary cause of the passivity attractor. Material-at-cap
    // alone (without early termination on material imbalance) is a weaker incentive:
    // preserving material only pays off IF you survive to a real terminal, so passive
    // play still risks getting checkmated.
    let (game_outcome, is_draw) = match board.result() {
        GameResult::Checkmate(Color::White) => (1.0f32, false),
        GameResult::Checkmate(Color::Black) => (-1.0f32, false),
        _ => {
            // All non-checkmate terminals: stalemate, repetition, 50-move, cap, insufficient material.
            let delta = compute_material_diff(&board);
            ((delta as f32 / 5.0).tanh(), true)
        }
    };

    // Set terminal reward on last step
    if let Some(last) = steps.last_mut() {
        last.reward = game_outcome;
    }

    // Sampled self-play PGN logging: 1% of games, for opening-diversity analysis.
    // Cheaply keyed on a single rng call; no impact on training dynamics.
    if rand::random::<f32>() < 0.01 {
        let result_str = match game_outcome {
            x if x > 0.5 => "1-0",
            x if x < -0.5 => "0-1",
            _ => "1/2-1/2",
        };
        crate::selfplay::pgn::write_pgn_game(
            "logs/selfplay_sample.pgn",
            &format!("Selfplay model_v{model_version}"),
            "selfplay_white",
            "selfplay_black",
            result_str,
            &moves,
        );
    }

    GameTrajectory {
        steps,
        game_outcome,
        model_version,
        is_draw,
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

/// Material balance = white_total - black_total, in standard piece values.
/// Used for the material-proxy outcome on non-decisive terminals. Scale by tanh(Δ/5)
/// at call sites to bound to [-1, 1].
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{BoardObservation, HiddenState, Policy, NUM_ACTIONS};
    use crate::mcts::evaluator::Evaluator;
    use async_trait::async_trait;

    struct RandomEvaluator;

    #[async_trait]
    impl Evaluator for RandomEvaluator {
        async fn root_setup(
            &self,
            _obs: &BoardObservation,
            _legal_mask: &[bool],
        ) -> (HiddenState, Policy, f32) {
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

    #[tokio::test]
    async fn test_play_game_dual_completes() {
        let precomputed = Arc::new(PrecomputedItems::begin_precomputing());
        let white_evaluator: Arc<dyn Evaluator> = Arc::new(RandomEvaluator);
        let black_evaluator: Arc<dyn Evaluator> = Arc::new(RandomEvaluator);
        let config = GameConfig {
            num_simulations: 2,
            exploration_constant: 1.5,
            temperature_moves: 5,
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

}
