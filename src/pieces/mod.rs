use crate::{Bitboard, Color, PieceType, Square};

pub trait Piece {
    fn get_piece_type(&self) -> PieceType;
    fn get_color(&self) -> Color;
    fn generate_pseudo_legal_moves_mask(&self, square: Square, occupied_squares: Bitboard, friendly_pieces: Bitboard) -> Bitboard;
}

pub mod pawn;
pub mod knight;
pub mod bishop;
pub mod mod_rook;
pub mod queen;
pub mod king;
