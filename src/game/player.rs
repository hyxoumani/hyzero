use crate::Bitboard;

pub struct Player {
    color: bool,
    pawns_bb: Bitboard,
    bishops_bb: Bitboard,
    queen_bb: Bitboard,
    rooks_bb: Bitboard,
    king_bb: Bitboard,
    knights_bb:Bitboard,
    pieces: Bitboard
}
impl Player {
    pub fn new_white() -> Self {
        Self {
            color: true,  // true = white
            pawns_bb: 0x000000000000FF00,      // Rank 2
            rooks_bb: 0x0000000000000081,      // a1, h1
            knights_bb: 0x0000000000000042,    // b1, g1
            bishops_bb: 0x0000000000000024,    // c1, f1
            queen_bb: 0x0000000000000008,      // d1
            king_bb: 0x0000000000000010,       // e1
            pieces: 0x000000000000FFFF,        // All white pieces (ranks 1-2)
        }
    }
    
    pub fn new_black() -> Self {
        Self {
            color: false,  // false = black
            pawns_bb: 0x00FF000000000000,      // Rank 7
            rooks_bb: 0x8100000000000000,      // a8, h8
            knights_bb: 0x4200000000000000,    // b8, g8
            bishops_bb: 0x2400000000000000,    // c8, f8
            queen_bb: 0x0800000000000000,      // d8
            king_bb: 0x1000000000000000,       // e8
            pieces: 0xFFFF000000000000,        // All black pieces (ranks 7-8)
        }
    }
}