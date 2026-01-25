pub type Bitboard = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Square {
    A1, B1, C1, D1, E1, F1, G1, H1,
    A2, B2, C2, D2, E2, F2, G2, H2,
    A3, B3, C3, D3, E3, F3, G3, H3,
    A4, B4, C4, D4, E4, F4, G4, H4,
    A5, B5, C5, D5, E5, F5, G5, H5,
    A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7,
    A8, B8, C8, D8, E8, F8, G8, H8,
}

impl From<u8> for Square {
    fn from(s: u8) -> Self {
        debug_assert!(s < 64);
        unsafe { std::mem::transmute(s) }
    }
}

impl From<Square> for u8 {
    fn from(s: Square) -> Self {
        s as u8
    }
}

impl From<Square> for usize {
    fn from (s:Square) -> Self {
        s as usize
    }
}

impl From<PieceType> for usize {
    fn from (p: PieceType) -> Self {
        p as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum File {
    A, B, C, D, E, F, G, H,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Rank {
    R1, R2, R3, R4, R5, R6, R7, R8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Color {
    White,
    Black,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PieceType {
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Piece {
    pub piece_type: PieceType,
    pub color: Color,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Move {
    pub from_square: Square,
    pub to_square: Square,
    pub piece_moved: Piece,
    pub is_capture: bool,
    pub is_castle: bool,
    pub is_en_passant: bool,
    pub promotion_piece_type: Option<PieceType>,
}



pub struct precomputed_items {
    sig_indexes: [[0; len(PieceType)]; len(Square)]
    bit_mask: [[0; len(PieceType)]; len(Square)]
    //using sig_indexes and bit_mask use mask to pre_compute things 
}

impl precomputed_items {
    pub fn precompue_masks (&self) {

    }
}



pub mod game;
pub mod pieces;

