use crate::{Color, PieceType};

pub trait Piece {
    fn get_piece_type(&self) -> PieceType;
    fn get_color(&self) -> Color;
}

pub mod pawn;
pub mod knight;
pub mod bishop;
pub mod mod_rook;
pub mod queen;
pub mod king;
