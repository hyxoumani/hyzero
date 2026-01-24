use crate::{Bitboard, Color, PieceType, Square};
use super::Piece;

pub struct Pawn {
    pub color: Color,
}

impl Piece for Pawn {
    fn get_piece_type(&self) -> PieceType {
        PieceType::Pawn
    }

    fn get_color(&self) -> Color {
        self.color
    }

    fn generate_pseudo_legal_moves_mask(&self, square: Square, occupied_squares: Bitboard, friendly_pieces: Bitboard) -> Bitboard {
        let mut moves = 0u64;
        let current_square_bb = 1u64 << (square as u8);

        match self.color {
            Color::White => {
                // Single push
                let single_push = (current_square_bb << 8) & !occupied_squares;
                moves |= single_push;

                // Double push
                if (square as u8 / 8) == 1 && single_push != 0 { // Check if on 2nd rank and single push is possible
                    moves |= (single_push << 8) & !occupied_squares;
                }

                // Captures
                let left_capture = (current_square_bb << 7) & 0xFEFEFEFEFEFEFEFEu64; // Not on H file
                let right_capture = (current_square_bb << 9) & 0x7F7F7F7F7F7F7F7Fu64; // Not on A file

                moves |= (left_capture | right_capture) & !friendly_pieces & occupied_squares;

                // TODO: En passant
            },
            Color::Black => {
                // Single push
                let single_push = (current_square_bb >> 8) & !occupied_squares;
                moves |= single_push;

                // Double push
                if (square as u8 / 8) == 6 && single_push != 0 { // Check if on 7th rank and single push is possible
                    moves |= (single_push >> 8) & !occupied_squares;
                }

                // Captures
                let left_capture = (current_square_bb >> 9) & 0xFEFEFEFEFEFEFEFEu64; // Not on H file
                let right_capture = (current_square_bb >> 7) & 0x7F7F7F7F7F7F7F7Fu64; // Not on A file

                moves |= (left_capture | right_capture) & !friendly_pieces & occupied_squares;

                // TODO: En passant
            },
        }

        moves
    }
}
