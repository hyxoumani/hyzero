use crate::game::{Move, GameBoard};
use crate::{Color, PieceType, Square, BitIterator};
use super::types::{BoardObservation, ActionIndex, NUM_ACTIONS};

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

/// Decode an ActionIndex back to a Move.
/// Sets promotion to Queen if the move reaches the back rank.
pub fn action_to_move(action: ActionIndex) -> Move {
    let from_sq = (action / 64) as u8;
    let to_sq = (action % 64) as u8;
    let to_rank = to_sq / 8;

    // Check if this could be a pawn promotion (reaching rank 0 or 7)
    let promotion = if to_rank == 7 || to_rank == 0 {
        Some(PieceType::Queen)
    } else {
        None
    };

    Move {
        from: Square::from(from_sq),
        to: Square::from(to_sq),
        promotion_piece_type: promotion,
        castle_option: None,
        en_passant: false,
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
