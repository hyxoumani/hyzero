pub type Bitboard = u64;

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Square {
    #[default]
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
    fn from(s: Square) -> Self {
        s as usize
    }
}

impl From<PieceType> for usize {
    fn from(p: PieceType) -> Self {
        p as usize
    }
}

pub struct BitIterator {
    bits: u64,
}

impl Iterator for BitIterator {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.bits == 0 {
            None
        } else {
            let sq = self.bits.trailing_zeros() as usize;
            self.bits &= self.bits - 1;
            Some(sq)
        }
    }
}

impl BitIterator {
    pub fn new(bits: u64) -> Self {
        Self { bits }
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
pub enum CastleOption {
    Kingside,
    Queenside
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
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

use crate::pieces::bishop::BishopEntry;
use crate::pieces::mod_rook::RookEntry;
#[derive(Debug)]
pub struct PrecomputedItems {
    pub knight_moves: [u64; 64],
    pub king_moves: [u64; 64],
    pub white_pawn_attacks: [u64; 64],
    pub white_pawn_pushes: [u64; 64],
    pub black_pawn_pushes: [u64; 64],
    pub black_pawn_attacks: [u64; 64],
    pub rook_moves: [RookEntry; 64],
    pub bishop_moves: [BishopEntry; 64],
    pub rays: [[u64; 64]; 64],
    pub lines: [[u64; 64]; 64],
    pub castle_empty_squares: [[u64; 2]; 2],
    pub castle_path_squares: [[u64; 2]; 2],
}

impl PrecomputedItems {
    pub fn precompute_castle_squares() -> ([[u64; 2]; 2], [[u64; 2]; 2]) {
        let mut empty = [[0u64; 2]; 2];
        let mut path = [[0u64; 2]; 2];

        // Empty squares: between king and rook (must be unoccupied)
        empty[Color::White as usize][CastleOption::Kingside as usize] =
            (1u64 << 5) | (1u64 << 6);           // f1, g1
        empty[Color::White as usize][CastleOption::Queenside as usize] =
            (1u64 << 1) | (1u64 << 2) | (1u64 << 3); // b1, c1, d1
        empty[Color::Black as usize][CastleOption::Kingside as usize] =
            (1u64 << 61) | (1u64 << 62);          // f8, g8
        empty[Color::Black as usize][CastleOption::Queenside as usize] =
            (1u64 << 57) | (1u64 << 58) | (1u64 << 59); // b8, c8, d8

        // Path squares: king passes through these (must not be under attack)
        path[Color::White as usize][CastleOption::Kingside as usize] =
            (1u64 << 4) | (1u64 << 5) | (1u64 << 6);  // e1, f1, g1
        path[Color::White as usize][CastleOption::Queenside as usize] =
            (1u64 << 4) | (1u64 << 3) | (1u64 << 2);  // e1, d1, c1
        path[Color::Black as usize][CastleOption::Kingside as usize] =
            (1u64 << 60) | (1u64 << 61) | (1u64 << 62); // e8, f8, g8
        path[Color::Black as usize][CastleOption::Queenside as usize] =
            (1u64 << 60) | (1u64 << 59) | (1u64 << 58); // e8, d8, c8

        (empty, path)
    }

    pub fn begin_precomputing() -> Self {

        let mut rays = [[0u64; 64]; 64];

        for from in 0..64 {
            let from_rank = from / 8;
            let from_file = from % 8;

            for to in 0..64 {
                let to_rank = to / 8;
                let to_file = to % 8;

                let rank_diff = (to_rank as i8) - (from_rank as i8);
                let file_diff = (to_file as i8) - (from_file as i8);

                let is_on_line = from_rank == to_rank ||
                                from_file == to_file ||
                                rank_diff.abs() == file_diff.abs();

                if is_on_line && from != to {
                    let rank_step = rank_diff.signum();
                    let file_step = file_diff.signum();

                    let mut current_rank = from_rank as i8 + rank_step;
                    let mut current_file = from_file as i8 + file_step;

                    while (current_rank as usize) != to_rank || (current_file as usize) != to_file {
                        let current_sq = (current_rank * 8 + current_file) as usize;
                        rays[from][to] |= 1u64 << current_sq;

                        current_rank += rank_step;
                        current_file += file_step;
                    }
                }
            }
        }

        let mut line_masks = [[0u64; 64]; 64];

        for sq1 in 0..64 {
            let r1 = (sq1 / 8) as i32;
            let c1 = (sq1 % 8) as i32;

            for sq2 in 0..64 {
                if sq1 == sq2 { continue; }

                let r2 = (sq2 / 8) as i32;
                let c2 = (sq2 % 8) as i32;

                let dr = r2 - r1;
                let dc = c2 - c1;

                if dr == 0 || dc == 0 || dr.abs() == dc.abs() {
                    let step_r = dr.signum();
                    let step_c = dc.signum();

                    let mut line = 0u64;

                    let mut curr_r = r1;
                    let mut curr_c = c1;

                    while curr_r >= 0 && curr_r < 8 && curr_c >= 0 && curr_c < 8 {
                        curr_r -= step_r;
                        curr_c -= step_c;
                    }

                    curr_r += step_r;
                    curr_c += step_c;

                    while curr_r >= 0 && curr_r < 8 && curr_c >= 0 && curr_c < 8 {
                        line |= 1u64 << (curr_r * 8 + curr_c);
                        curr_r += step_r;
                        curr_c += step_c;
                    }

                    line_masks[sq1][sq2] = line;
                }
            }
        }

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

        let white_pawn_attacks: [u64; 64] = std::array::from_fn(|i| {
            let mut mask = 0u64;
            let (r, c) = (i / 8, i % 8);
            if r < 7 {
                if c > 0 { mask |= 1u64 << ((r + 1) * 8 + (c - 1)); }
                if c < 7 { mask |= 1u64 << ((r + 1) * 8 + (c + 1)); }
            }
            mask
        });

        let white_pawn_pushes: [u64; 64] = std::array::from_fn(|i| {
            let mut mask = 0u64;
            let (r, c) = (i / 8, i % 8);
            if r < 7 {
                mask |= 1u64 << ((r + 1) * 8 + c);
                if r == 1 { mask |= 1u64 << ((r + 2) * 8 + c); }
            }
            mask
        });

        let black_pawn_attacks: [u64; 64] = std::array::from_fn(|i| {
            let mut mask = 0u64;
            let (r, c) = (i / 8, i % 8);
            if r > 0 {
                if c > 0 { mask |= 1u64 << ((r - 1) * 8 + (c - 1)); }
                if c < 7 { mask |= 1u64 << ((r - 1) * 8 + (c + 1)); }
            }
            mask
        });

        let black_pawn_pushes: [u64; 64] = std::array::from_fn(|i| {
            let mut mask = 0u64;
            let (r, c) = (i / 8, i % 8);
            if r > 0 {
                mask |= 1u64 << ((r - 1) * 8 + c);
                if r == 6 { mask |= 1u64 << ((r - 2) * 8 + c); }
            }
            mask
        });

        let rook_moves = std::array::from_fn(|i| RookEntry::init_rook(i as u8));
        let bishop_moves = std::array::from_fn(|i| BishopEntry::init_bishop(i as u8));

        PrecomputedItems {
            knight_moves,
            king_moves,
            white_pawn_pushes,
            white_pawn_attacks,
            black_pawn_attacks,
            black_pawn_pushes,
            rook_moves,
            bishop_moves,
            rays,
            lines: line_masks,
            castle_empty_squares: {
                let (empty, _) = PrecomputedItems::precompute_castle_squares();
                empty
            },
            castle_path_squares: {
                let (_, path) = PrecomputedItems::precompute_castle_squares();
                path
            }
        }
    }
}

pub fn create_game_board() -> [Option<Piece>; 64] {
    let mut board = [None; 64];
    let white = |t:PieceType| Some(Piece{piece_type: t, color: Color::White});
    let black = |t:PieceType| Some(Piece{piece_type:t, color: Color::Black});
    board[0] = white(PieceType::Rook);
    board[1] = white(PieceType::Knight);
    board[2] = white(PieceType::Bishop);
    board[3] = white(PieceType::Queen);
    board[4] = white(PieceType::King);
    board[5] = white(PieceType::Bishop);
    board[6] = white(PieceType::Knight);
    board[7] = white(PieceType::Rook);
    for i in 8..16 { board[i] = white(PieceType::Pawn); }
    for i in 48..56 { board[i] = black(PieceType::Pawn); }
    board[56] = black(PieceType::Rook);
    board[57] = black(PieceType::Knight);
    board[58] = black(PieceType::Bishop);
    board[59] = black(PieceType::Queen);
    board[60] = black(PieceType::King);
    board[61] = black(PieceType::Bishop);
    board[62] = black(PieceType::Knight);
    board[63] = black(PieceType::Rook);

    board
}

pub fn create_own_board(color:Color) -> [Option<Piece>; 64] {
    let mut board = [None; 64];
    let white = |t:PieceType| Some(Piece{piece_type: t, color: Color::White});
    let black = |t:PieceType| Some(Piece{piece_type:t, color: Color::Black});
    if color == Color::White {
        board[0] = white(PieceType::Rook);
        board[1] = white(PieceType::Knight);
        board[2] = white(PieceType::Bishop);
        board[3] = white(PieceType::Queen);
        board[4] = white(PieceType::King);
        board[5] = white(PieceType::Bishop);
        board[6] = white(PieceType::Knight);
        board[7] = white(PieceType::Rook);
        for i in 8..16 { board[i] = white(PieceType::Pawn); }
    } else {
        for i in 48..56 { board[i] = black(PieceType::Pawn); }
        board[56] = black(PieceType::Rook);
        board[57] = black(PieceType::Knight);
        board[58] = black(PieceType::Bishop);
        board[59] = black(PieceType::Queen);
        board[60] = black(PieceType::King);
        board[61] = black(PieceType::Bishop);
        board[62] = black(PieceType::Knight);
        board[63] = black(PieceType::Rook);
    }
    board
}

pub mod game;
pub mod pieces;
pub mod session;
