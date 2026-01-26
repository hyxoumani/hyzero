use crate::{Bitboard, Color, PieceType, Square};
use super::Piece;

pub struct Rook {
    pub color: Color,
}

impl Piece for Rook {
    fn get_piece_type(&self) -> PieceType {
        PieceType::Rook
    }

    fn get_color(&self) -> Color {
        self.color
    }

    fn generate_pseudo_legal_moves_mask(&self, square: Square, occupied_squares: Bitboard, friendly_pieces: Bitboard) -> Bitboard {
        // This is a placeholder. A proper sliding piece attack generator will be more complex.
        // For now, we'll use a simplified (and incorrect) approach.
        let mut moves = 0u64;
        let current_square_bb = 1u64 << (square as u8);

        // Helper for rank and file attacks (simplified for now)
        // TODO: Implement proper sliding attacks with blockers
        // These are just a few squares to demonstrate, not full rank/file logic
        if square as u8 / 8 < 7 { // North
            moves |= current_square_bb.checked_shl(8).unwrap_or(0);
        }
        if square as u8 / 8 > 0 { // South
            moves |= current_square_bb.checked_shr(8).unwrap_or(0);
        }
        if square as u8 % 8 < 7 { // East
            moves |= current_square_bb.checked_shl(1).unwrap_or(0);
        }
        if square as u8 % 8 > 0 { // West
            moves |= current_square_bb.checked_shr(1).unwrap_or(0);
        }

        moves & !friendly_pieces
    }
}

pub struct RookEntry {
    pub mask: u64,
    pub magic_num: u64,
    pub sig_bits: u8,
    pub magic_table: Vec<u64>,
    pub pos: u8
}

use rand::Rng; 

impl RookEntry {
    //first calculate mask
    //using mask calc sig_bits
    //then find magic number
    //then compute magic_table

    pub fn init_rook(pos: u8) -> Self{
        let mut rook_entry = RookEntry {
            mask: 0,
            magic_num: 0,
            sig_bits: 0,
            magic_table: Vec::new(),
            pos: pos
        };
        
        rook_entry.mask = rook_entry.calculate_mask(pos);
        rook_entry.sig_bits = rook_entry.mask.count_ones() as u8;
        rook_entry.calculate_magic_num();


        rook_entry
    }

    pub fn calculate_mask(&self, pos:u8) -> u64 {
        let mut mask = 0u64;
        let r = (pos / 8) as i32;
        let c = (pos % 8) as i32;
        for i in (r + 1)..7 { mask |= 1 << (i * 8 + c); }
        for i in 1..r { mask |= 1 << (i * 8 + c); }
        for i in (c + 1)..7 { mask |= 1 << (r * 8 + i); }
        for i in 1..c { mask |= 1 << (r * 8 + i); }
        mask
    }

    pub fn calculate_magic_num(&mut self){
        let mut rng = rand::thread_rng();
        let num_var = 1 << self.sig_bits;

        let combinations = self.generate_combinations();
        let real_moves:Vec<u64> = combinations.iter().map(|&combo| self.get_moves(combo)).collect();

        loop {
            let candidates = rng.random::<u64>() & rng.random::<u64>() & rng.random::<u64>();
            let mut test_vec = vec![0u64; num_var];
            let mut fail = false;

            for (i, &combo) in combinations.iter().enumerate(){
                let ind = (combo.wrapping_mul(candidates) >> (64-self.sig_bits)) as usize;
                if test_vec[ind] == 0{
                    test_vec[ind] = real_moves[i];
                } else{
                    fail = true;
                    break;
                }
            }

            if !fail {
                self.magic_num = candidates;
                self.magic_table = test_vec;
                return;
            }
        }
    }

    pub fn generate_combinations(&self) -> Vec<u64> {
        let mut mask_ind = Vec::new();
        for i in 0..64 {
            if (self.mask & (1 << i)) != 0{
                mask_ind.push(i);
            }
        }

        //generate the combos
        (0 .. (1 << self.sig_bits)).map(|i| {
            let mut combo = 0u64;
            for j in 0..self.sig_bits {
                if (i & (1 << j)) != 0 {
                    combo |= 1 << mask_ind[j as usize];
                }
            } combo 
        }).collect()
    }

    fn get_moves(&self, occupied: u64) -> u64 {
        let mut moves = 0u64;
        let r = (self.pos / 8) as i32;
        let c = (self.pos % 8) as i32;
        let dirs = [(1,0), (-1,0), (0,1), (0,-1)];

        for (dr, dc) in dirs {
            for len in 1..8 {
                let (nr, nc) = (r + dr * len, c + dc * len);
                if nr < 0 || nr >= 8 || nc < 0 || nc >= 8 { break; }
                let bit = 1 << (nr * 8 + nc);
                moves |= bit;
                if (occupied & bit) != 0 { break; } // Hit a blocker
            }
        }
        moves
    }

}