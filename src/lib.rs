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








use crate::pieces::bishop::BishopEntry;
use crate::pieces::mod_rook::RookEntry;

pub struct precomputed_items {
    pub knight_moves: [u64; 64],
    pub king_moves: [u64; 64],
    pub white_pawn_moves: [u64; 64],
    pub black_pawn_moves: [u64; 64],
    pub rook_moves: [RookEntry; 64],
    pub bishop_moves: [BishopEntry; 64]
    //using sig_indexes and bit_mask use mask to pre_compute things 
}

impl precomputed_items {

    pub fn begin_precomputing() -> Self {
        let knight_moves = std::array::from_fn(|i| {
            let mut moves = 0u64;
            let r = (i / 8) as i32;
            let c = (i % 8) as i32;
            let offsets = [(2,1),(2,-1),(-2,1),(-2,-1),(1,2),(1,-2),(-1,2),(-1,-2)];
            for (dr, dc) in offsets {
                let (nr, nc) = (r + dr, c + dc);
                if nr >= 0 && nr < 8 && nc >= 0 && nc < 8 {
                    moves |= 1 << (nr * 8 + nc);
                }
            }
            moves
        });

        // --- 2. Kings ---
        let king_moves = std::array::from_fn(|i| {
            let mut moves = 0u64;
            let r = (i / 8) as i32;
            let c = (i % 8) as i32;
            for dr in -1..=1 {
                for dc in -1..=1 {
                    if dr == 0 && dc == 0 { continue; }
                    let (nr, nc) = (r + dr, c + dc);
                    if nr >= 0 && nr < 8 && nc >= 0 && nc < 8 {
                        moves |= 1 << (nr * 8 + nc);
                    }
                }
            }
            moves
        });

        // --- 3. Pawn Attacks (The "V" mask you mentioned) ---
        let white_pawn_moves = std::array::from_fn(|i| {
            let mut attacks = 0u64;
            let r = (i / 8) as i32;
            let c = (i % 8) as i32;
            // White attacks "up" (higher rank)
            if r < 7 { 
                if c > 0 { attacks |= 1 << ((r + 1) * 8 + (c - 1)); }
                if c < 7 { attacks |= 1 << ((r + 1) * 8 + (c + 1)); }
            }
            attacks
        });

        let black_pawn_moves = std::array::from_fn(|i| {
            let mut attacks = 0u64;
            let r = (i / 8) as i32;
            let c = (i % 8) as i32;
            // Black attacks "down" (lower rank)
            if r > 0 {
                if c > 0 { attacks |= 1 << ((r - 1) * 8 + (c - 1)); }
                if c < 7 { attacks |= 1 << ((r - 1) * 8 + (c + 1)); }
            }
            attacks
        });

        // --- 4. Sliders (Rooks & Bishops) ---
        // Here 'i' is the square index (0..63) passed to your init functions
        let rook_moves = std::array::from_fn(|i| RookEntry::init_rook(i as u8));
        let bishop_moves = std::array::from_fn(|i| BishopEntry::init_bishop(i as u8));

        precomputed_items {
            knight_moves,
            king_moves,
            white_pawn_moves,
            black_pawn_moves,
            rook_moves,
            bishop_moves,
        }
    }





}



pub mod game;
pub mod pieces;

