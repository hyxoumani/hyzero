use crate::{Color, PieceType};
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

    
}
