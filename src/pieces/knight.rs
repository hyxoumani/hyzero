use crate::{Color, PieceType};
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

}
