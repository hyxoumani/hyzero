use crate::{Bitboard, Color, PieceType, Square};
use super::Piece;

pub struct King {
    pub color: Color,
}

impl Piece for King {
    fn get_piece_type(&self) -> PieceType {
        PieceType::King
    }

    fn get_color(&self) -> Color {
        self.color
    }

    fn generate_pseudo_legal_moves_mask(&self, square: Square, _occupied_squares: Bitboard, friendly_pieces: Bitboard) -> Bitboard {
        let mut moves = 0u64;
        let current_square_bb = 1u64 << (square as u8);

        // King moves are one square in any direction
        let mut king_attacks = 0u64;

        // North
        king_attacks |= current_square_bb.checked_shl(8).unwrap_or(0);
        // South
        king_attacks |= current_square_bb.checked_shr(8).unwrap_or(0);

        // East (not on H file)
        if square as u8 % 8 < 7 {
            king_attacks |= current_square_bb.checked_shl(1).unwrap_or(0);
            // North-East
            king_attacks |= current_square_bb.checked_shl(9).unwrap_or(0);
            // South-East
            king_attacks |= current_square_bb.checked_shr(7).unwrap_or(0);
        }

        // West (not on A file)
        if square as u8 % 8 > 0 {
            king_attacks |= current_square_bb.checked_shr(1).unwrap_or(0);
            // North-West
            king_attacks |= current_square_bb.checked_shl(7).unwrap_or(0);
            // South-West
            king_attacks |= current_square_bb.checked_shr(9).unwrap_or(0);
        }
        
        moves = king_attacks & !friendly_pieces;

        // TODO: Castling

        moves
    }
}
