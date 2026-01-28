use crate::Bitboard;
use crate::game::Move;
use rand::Rng;

#[derive(Debug, Clone)]
pub struct Player {
    pub (crate)color: bool,
    pub (crate)pieces_bb: [u64; 6], // [Pawn, Knight, Bishop, Rook, Queen, King]
    pub (crate)pieces: u64,         // All pieces combined (occupancy)
}

impl Player {
    pub fn new_white() -> Self {
        let bbs = [
            0x000000000000FF00, // 0: Pawns
            0x0000000000000042, // 1: Knights
            0x0000000000000024, // 2: Bishops
            0x0000000000000081, // 3: Rooks
            0x0000000000000008, // 4: Queen
            0x0000000000000010, // 5: King
        ];

        Self {
            color: true,
            pieces_bb: bbs,
            pieces: 0x000000000000FFFF, 
        }
    }

    pub fn make_move(&mut self) -> Move{
        //randomly generate
        let mut rng = rand::thread_rng();
        Move{
            from: rng.gen_range(0..64).into(),
            to: rng.gen_range(0..64).into(),
            promotion_piece_type: None
        }
    }
    
    pub fn new_black() -> Self {
        let bbs = [
            0x00FF000000000000, // 0: Pawns
            0x4200000000000000, // 1: Knights
            0x2400000000000000, // 2: Bishops
            0x8100000000000000, // 3: Rooks
            0x0800000000000000, // 4: Queen
            0x1000000000000000, // 5: King
        ];

        Self {
            color: false,
            pieces_bb: bbs,
            pieces: 0xFFFF000000000000,
        }
    }
}