use crate::{Bitboard, Color, PieceType, Square};
use super::Piece;

pub struct Knight {
    pub color: Color,
}

impl Piece for Knight {
    fn get_piece_type(&self) -> PieceType {
        PieceType::Knight
    }

    fn get_color(&self) -> Color {
        self.color
    }

    fn generate_pseudo_legal_moves_mask(&self, square: Square, _occupied_squares: Bitboard, friendly_pieces: Bitboard) -> Bitboard {
        let mut moves = 0u64;
        let current_square_bb = 1u64 << (square as u8);

        // All possible knight moves relative to a square
        let mut knight_attacks = 0u64;

        // Shift current_square_bb in all 8 knight move directions
        // 2 up, 1 left/right
        if square as u8 % 8 > 0 { // Not on A file
            knight_attacks |= current_square_bb.checked_shl(17).unwrap_or(0); // Up 2, Left 1
            knight_attacks |= current_square_bb.checked_shr(15).unwrap_or(0); // Down 2, Left 1
        }
        if square as u8 % 8 < 7 { // Not on H file
            knight_attacks |= current_square_bb.checked_shl(15).unwrap_or(0); // Up 2, Right 1
            knight_attacks |= current_square_bb.checked_shr(17).unwrap_or(0); // Down 2, Right 1
        }

        // 1 up/down, 2 left/right
        if square as u8 % 8 > 1 { // Not on A or B file
            knight_attacks |= current_square_bb.checked_shl(10).unwrap_or(0); // Up 1, Left 2
            knight_attacks |= current_square_bb.checked_shr(6).unwrap_or(0); // Down 1, Left 2
        }
        if square as u8 % 8 < 6 { // Not on G or H file
            knight_attacks |= current_square_bb.checked_shl(6).unwrap_or(0); // Up 1, Right 2
            knight_attacks |= current_square_bb.checked_shr(10).unwrap_or(0); // Down 1, Right 2
        }
        
        moves = knight_attacks & !friendly_pieces;

        moves
    }
}
