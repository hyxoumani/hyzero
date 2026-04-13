use super::types::{ActionIndex, BoardObservation, NUM_ACTIONS};
use crate::game::{GameBoard, Move};
use crate::{BitIterator, CastleOption, Color, Piece, PieceType, Square};

/// Encode a GameBoard into a BoardObservation for the representation network.
pub fn encode_board(board: &GameBoard, side_to_move: Color) -> BoardObservation {
    let mut obs = BoardObservation::default();

    // Planes 0-5: White pieces (Pawn=0, Knight=1, Bishop=2, Rook=3, Queen=4, King=5)
    for pt in 0..6 {
        let bb = board.player1.pieces_bb[pt];
        for sq in BitIterator::new(bb) {
            obs.planes[pt * 64 + sq] = 1.0;
        }
    }

    // Planes 6-11: Black pieces
    for pt in 0..6 {
        let bb = board.player2.pieces_bb[pt];
        for sq in BitIterator::new(bb) {
            obs.planes[(pt + 6) * 64 + sq] = 1.0;
        }
    }

    // Planes 12-15: Castling rights (constant plane — all 64 squares set to 1.0 if right available)
    let castling = [
        board.white_kingside,
        board.white_queenside,
        board.black_kingside,
        board.black_queenside,
    ];
    for (i, &has_right) in castling.iter().enumerate() {
        if has_right {
            let plane_offset = (12 + i) * 64;
            for sq in 0..64 {
                obs.planes[plane_offset + sq] = 1.0;
            }
        }
    }

    // Plane 16: En passant target (one-hot)
    if let Some(ep_sq) = board.en_passant_target {
        obs.planes[16 * 64 + ep_sq] = 1.0;
    }

    // Plane 17: Side to move (all 1.0 if white, all 0.0 if black)
    if side_to_move == Color::White {
        let plane_offset = 17 * 64;
        for sq in 0..64 {
            obs.planes[plane_offset + sq] = 1.0;
        }
    }

    // Plane 18: Halfmove clock (normalized by 100)
    let clock_val = board.halfmove_clock as f32 / 100.0;
    let plane_offset = 18 * 64;
    for sq in 0..64 {
        obs.planes[plane_offset + sq] = clock_val;
    }

    obs
}

/// Encode a Move as an ActionIndex (from_square * 64 + to_square).
/// Default queen promotion — underpromotion support added later.
pub fn move_to_action(mv: &Move) -> ActionIndex {
    (mv.from as u16) * 64 + (mv.to as u16)
}

/// Decode an ActionIndex back to a Move, using board context to detect castling and en passant.
/// Promotion defaults to Queen when a pawn reaches the back rank.
pub fn action_to_move(action: ActionIndex, board: &GameBoard, _color: Color) -> Move {
    let from_sq = (action / 64) as u8;
    let to_sq = (action % 64) as u8;
    let from_file = (from_sq % 8) as i8;
    let to_file = (to_sq % 8) as i8;
    let to_rank = to_sq / 8;

    let piece: Option<Piece> = board.board_arr[from_sq as usize];

    // Castling detection: King moves two files
    let castle_option = if piece.map(|p| p.piece_type) == Some(PieceType::King)
        && (to_file - from_file).abs() == 2
    {
        if to_file > from_file {
            Some(CastleOption::Kingside)
        } else {
            Some(CastleOption::Queenside)
        }
    } else {
        None
    };

    // En passant detection: Pawn moves diagonally to the ep target square
    let en_passant = castle_option.is_none()
        && piece.map(|p| p.piece_type) == Some(PieceType::Pawn)
        && (to_file - from_file).abs() == 1
        && board.en_passant_target == Some(to_sq as usize);

    // Promotion detection: Pawn reaches back rank (only when not castling or ep)
    let promotion_piece_type = if castle_option.is_none()
        && !en_passant
        && piece.map(|p| p.piece_type) == Some(PieceType::Pawn)
        && (to_rank == 7 || to_rank == 0)
    {
        Some(PieceType::Queen)
    } else {
        None
    };

    Move {
        from: Square::from(from_sq),
        to: Square::from(to_sq),
        promotion_piece_type,
        castle_option,
        en_passant,
    }
}

/// Encode an action as 3 spatial planes for the dynamics network.
/// Plane 0: source square one-hot (8x8)
/// Plane 1: destination square one-hot (8x8)
/// Plane 2: promotion flag (all 1s if promotion, all 0s otherwise)
pub fn encode_action_spatial(action: ActionIndex) -> [f32; 3 * 64] {
    let mut planes = [0.0f32; 3 * 64];
    let from_sq = (action / 64) as usize;
    let to_sq = (action % 64) as usize;

    // Plane 0: source square
    planes[from_sq] = 1.0;
    // Plane 1: destination square
    planes[64 + to_sq] = 1.0;
    // Plane 2: promotion flag
    let to_rank = to_sq / 8;
    if to_rank == 7 || to_rank == 0 {
        for sq in 0..64 {
            planes[128 + sq] = 1.0;
        }
    }

    planes
}

/// Total action space size.
pub fn num_actions() -> usize {
    NUM_ACTIONS
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::{GameBoard, Player};
    use crate::{Color, Piece, PieceType, PrecomputedItems, Square};
    use std::sync::Arc;

    fn make_board() -> GameBoard {
        let precomputed = Arc::new(PrecomputedItems::begin_precomputing());
        let p1 = Player::init_player(true);
        let p2 = Player::init_player(false);
        GameBoard::init_game_board(precomputed, p1, p2)
    }

    #[test]
    fn test_action_to_move_normal() {
        // e2e4: from=12 (E2), to=28 (E4), action = 12*64+28 = 796
        let board = make_board();
        let mv = action_to_move(796, &board, Color::White);
        assert_eq!(mv.from, Square::E2);
        assert_eq!(mv.to, Square::E4);
        assert!(mv.castle_option.is_none());
        assert!(!mv.en_passant);
        assert!(mv.promotion_piece_type.is_none());
    }

    #[test]
    fn test_action_to_move_castling_kingside() {
        // King e1->g1: from=4 (E1), to=6 (G1), action = 4*64+6 = 262
        let mut board = make_board();
        // Clear f1 (5) and g1 (6) for the castling path
        board.board_arr[5] = None;
        board.board_arr[6] = None;
        let mv = action_to_move(262, &board, Color::White);
        assert_eq!(mv.castle_option, Some(crate::CastleOption::Kingside));
        assert!(!mv.en_passant);
        assert!(mv.promotion_piece_type.is_none());
    }

    #[test]
    fn test_action_to_move_castling_queenside() {
        // King e1->c1: from=4 (E1), to=2 (C1), action = 4*64+2 = 258
        let mut board = make_board();
        // Clear b1 (1), c1 (2), d1 (3)
        board.board_arr[1] = None;
        board.board_arr[2] = None;
        board.board_arr[3] = None;
        let mv = action_to_move(258, &board, Color::White);
        assert_eq!(mv.castle_option, Some(crate::CastleOption::Queenside));
        assert!(!mv.en_passant);
        assert!(mv.promotion_piece_type.is_none());
    }

    #[test]
    fn test_action_to_move_en_passant() {
        // White pawn on e5 (36) captures to d6 (43), ep target = 43
        let mut board = make_board();
        board.board_arr[36] = Some(Piece {
            piece_type: PieceType::Pawn,
            color: Color::White,
        });
        board.en_passant_target = Some(43);
        let action = 36 * 64 + 43; // 2347
        let mv = action_to_move(action, &board, Color::White);
        assert!(mv.en_passant);
        assert!(mv.castle_option.is_none());
        assert!(mv.promotion_piece_type.is_none());
    }

    #[test]
    fn test_action_to_move_promotion() {
        // White pawn on e7 (52) moves to e8 (60), action = 52*64+60 = 3388
        let mut board = make_board();
        board.board_arr[52] = Some(Piece {
            piece_type: PieceType::Pawn,
            color: Color::White,
        });
        let mv = action_to_move(52 * 64 + 60, &board, Color::White);
        assert_eq!(mv.promotion_piece_type, Some(PieceType::Queen));
        assert!(mv.castle_option.is_none());
        assert!(!mv.en_passant);
    }

    #[test]
    fn test_action_to_move_no_ep_when_square_differs() {
        // White pawn on e5 (36) moves straight to e6 (44), ep target = 43 (d6) — no ep
        let mut board = make_board();
        board.board_arr[36] = Some(Piece {
            piece_type: PieceType::Pawn,
            color: Color::White,
        });
        board.en_passant_target = Some(43);
        let mv = action_to_move(36 * 64 + 44, &board, Color::White);
        assert!(!mv.en_passant);
    }
}
