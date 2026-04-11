use crate::{Color, PieceType};
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
}
