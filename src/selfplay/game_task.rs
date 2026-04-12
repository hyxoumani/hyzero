use std::sync::Arc;

use crate::{PrecomputedItems, Color, PieceType, BitIterator, CastleOption, Square};
use crate::data::{
    ActionIndex, StepRecord, GameTrajectory,
    encode_board, move_to_action,
};
use crate::game::{Move, GameBoard, Player};
use crate::game::board::GameResult;
use crate::mcts::evaluator::Evaluator;
use crate::mcts::tree::{MCTSTree, MCTSConfig};

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

    const MAX_GAME_LENGTH: usize = 150;

    let mcts_config = MCTSConfig {
        num_simulations: config.num_simulations,
        exploration_constant: config.exploration_constant,
    };

    loop {
        if board.result() != GameResult::Ongoing {
            break;
        }

        if turn_count >= MAX_GAME_LENGTH {
            // Treat as draw — prevents runaway games
            break;
        }

        let observation = encode_board(&board, side_to_move);
        let legal_actions = get_legal_moves(&board, side_to_move);

        if legal_actions.is_empty() {
            break;
        }

        // Root setup: encode board into latent space
        let (hidden_state, policy, value) = evaluator.root_setup(&observation).await;

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

        // Select action based on temperature
        let temperature = if turn_count < config.temperature_moves as usize {
            1.0
        } else {
            0.01
        };
        let action = tree.select_action(temperature);

        // Record step before applying move
        steps.push(StepRecord {
            observation,
            action,
            visit_distribution,
            root_value,
            reward: 0.0, // Set terminal reward after game ends
            legal_moves: legal_actions,
        });

        // Convert action to move notation and apply
        let move_str = action_to_notation(action);
        match board.process_move(&move_str, side_to_move, turn_count) {
            Ok(_) => {}
            Err(_) => {
                // If the selected move is invalid (shouldn't happen with correct legal moves),
                // try the first legal move as fallback
                break;
            }
        }

        // Alternate sides
        side_to_move = if side_to_move == Color::White { Color::Black } else { Color::White };
        turn_count += 1;
    }

    // Determine game outcome
    let game_outcome = match board.result() {
        GameResult::Checkmate(Color::White) => 1.0,  // White won
        GameResult::Checkmate(Color::Black) => -1.0, // Black won
        _ => 0.0, // Draw or other
    };

    // Set terminal reward on last step
    if let Some(last) = steps.last_mut() {
        last.reward = game_outcome;
    }

    GameTrajectory {
        steps,
        game_outcome,
        model_version,
    }
}

/// Convert an ActionIndex (from*64 + to) to coordinate notation string (e.g., "e2e4").
fn action_to_notation(action: ActionIndex) -> String {
    let from_sq = (action / 64) as u8;
    let to_sq = (action % 64) as u8;

    let from_file = (b'a' + from_sq % 8) as char;
    let from_rank = (b'1' + from_sq / 8) as char;
    let to_file = (b'a' + to_sq % 8) as char;
    let to_rank = (b'1' + to_sq / 8) as char;

    // Add queen promotion suffix if pawn reaches back rank
    let to_rank_num = to_sq / 8;
    if to_rank_num == 7 || to_rank_num == 0 {
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
    let player = if color == Color::White { &board.player1 } else { &board.player2 };
    let combined = board.white_pieces | board.black_pieces;

    for sq in 0..64usize {
        let piece_opt = player.own_board[sq];
        let piece = match piece_opt {
            Some(p) if p.color == color => p,
            _ => continue,
        };

        let move_mask = board.get_move_mask(
            sq, color, piece.piece_type,
            combined, board.white_pieces, board.black_pieces,
        );

        for to_sq in BitIterator::new(move_mask) {
            let from = Square::from(sq as u8);
            let to = Square::from(to_sq as u8);

            // Detect promotion
            let to_rank = to_sq / 8;
            let promotion = if piece.piece_type == PieceType::Pawn && (to_rank == 7 || to_rank == 0) {
                Some(PieceType::Queen)
            } else {
                None
            };

            // Detect en passant
            let en_passant = piece.piece_type == PieceType::Pawn
                && board.en_passant_target == Some(to_sq)
                && (sq % 8 != to_sq % 8); // diagonal move to EP square

            let candidate = Move {
                from,
                to,
                promotion_piece_type: promotion,
                castle_option: None,
                en_passant,
            };

            if board.validate_move(candidate, color, combined, board.white_pieces, board.black_pieces) {
                legal.push(move_to_action(&candidate));
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

                if board.validate_move(candidate, color, combined, board.white_pieces, board.black_pieces) {
                    legal.push(move_to_action(&candidate));
                }
            }
        }
    }

    legal
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
        async fn root_setup(&self, _obs: &BoardObservation) -> (HiddenState, Policy, f32) {
            let policy = vec![1.0 / NUM_ACTIONS as f32; NUM_ACTIONS];
            (HiddenState::new(64), policy, 0.0)
        }

        async fn expand_leaf(&self, _hs: &HiddenState, _action: ActionIndex) -> (HiddenState, f32, Policy, f32) {
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
        assert_eq!(moves.len(), 20, "Expected 20 legal moves, got {}", moves.len());
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
            assert!(!step.legal_moves.is_empty(), "Each step should have legal moves");
            assert!(!step.visit_distribution.is_empty(), "Each step should have visit distribution");
        }

        // Game outcome should be set
        assert!(
            trajectory.game_outcome == 1.0
                || trajectory.game_outcome == -1.0
                || trajectory.game_outcome == 0.0,
            "Game outcome should be +1, -1, or 0"
        );
    }

    #[test]
    fn test_action_to_notation() {
        // e2 = sq 12, e4 = sq 28 → action = 12*64 + 28 = 796
        let notation = action_to_notation(796);
        assert_eq!(notation, "e2e4");

        // a7 = sq 48, a8 = sq 56 → action = 48*64 + 56 = 3128 (promotion)
        let notation = action_to_notation(3128);
        assert_eq!(notation, "a7a8q");
    }
}
