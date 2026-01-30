use crate::{Bitboard, Color, PieceType, Square};
use super::Piece;

pub struct Queen {
    pub color: Color,
}

impl Piece for Queen {
    fn get_piece_type(&self) -> PieceType {
        PieceType::Queen
    }

    fn get_color(&self) -> Color {
        self.color
    }
}
