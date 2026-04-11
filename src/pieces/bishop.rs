use crate::{Color, PieceType};
use super::Piece;

pub struct Bishop {
    pub color: Color,
}

impl Piece for Bishop {
    fn get_piece_type(&self) -> PieceType {
        PieceType::Bishop
    }

    fn get_color(&self) -> Color {
        self.color
    }

}

use rand::Rng;
#[derive(Debug)]
pub struct BishopEntry {
    pub mask: u64,
    pub magic_num: u64,
    pub sig_bits: u8,
    pub magic_table: Vec<u64>,
    pub pos: u8
}

impl BishopEntry {
    pub fn init_bishop(pos: u8) -> Self {
        let mut entry = BishopEntry {
            mask: 0,
            magic_num: 0,
            sig_bits: 0,
            magic_table: Vec::new(),
            pos,
        };
        
        entry.mask = entry.calculate_mask(pos);
        entry.sig_bits = entry.mask.count_ones() as u8;
        entry.calculate_magic_num();

        entry
    }

    pub fn calculate_mask(&self, pos: u8) -> u64 {
        let mut mask = 0u64;
        let r = (pos / 8) as i32;
        let c = (pos % 8) as i32;

        // Diagonals: Stop 1 square before the edge
        // Top-Right, Top-Left, Bottom-Right, Bottom-Left
        for (dr, dc) in [(1, 1), (1, -1), (-1, 1), (-1, -1)] {
            let mut nr = r + dr;
            let mut nc = c + dc;
            // Check boundaries: 1 to 6 (excluding 0 and 7)
            while nr > 0 && nr < 7 && nc > 0 && nc < 7 {
                mask |= 1 << (nr * 8 + nc);
                nr += dr;
                nc += dc;
            }
        }
        mask
    }

    pub fn calculate_magic_num(&mut self) {
        let mut rng = rand::rng();
        let num_var = 1 << self.sig_bits;

        let combinations = self.generate_combinations();
        let real_moves: Vec<u64> = combinations.iter()
            .map(|&combo| self.get_moves(combo))
            .collect();

        loop {
            // Sparse candidate for minimal carry-bit interference
            let candidates = rng.random::<u64>() & rng.random::<u64>() & rng.random::<u64>();
            let mut test_vec = vec![0u64; num_var];
            let mut fail = false;

            for (i, &combo) in combinations.iter().enumerate() {
                let ind = (combo.wrapping_mul(candidates) >> (64 - self.sig_bits)) as usize;
                
                if test_vec[ind] == 0 {
                    test_vec[ind] = real_moves[i];
                } else if test_vec[ind] != real_moves[i] {
                    // Fail only on "Destructive Collisions"
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
            if (self.mask & (1 << i)) != 0 {
                mask_ind.push(i);
            }
        }

        (0..(1 << self.sig_bits)).map(|i| {
            let mut combo = 0u64;
            for j in 0..self.sig_bits {
                if (i & (1 << j)) != 0 {
                    combo |= 1 << mask_ind[j as usize];
                }
            } 
            combo 
        }).collect()
    }

    fn get_moves(&self, occupied: u64) -> u64 {
        let mut moves = 0u64;
        let r = (self.pos / 8) as i32;
        let c = (self.pos % 8) as i32;
        let dirs = [(1, 1), (1, -1), (-1, 1), (-1, -1)];

        for (dr, dc) in dirs {
            for len in 1..8 {
                let (nr, nc) = (r + dr * len, c + dc * len);
                if !(0..8).contains(&nr) || !(0..8).contains(&nc) { break; }
                let bit = 1 << (nr * 8 + nc);
                moves |= bit;
                if (occupied & bit) != 0 { break; }
            }
        }
        moves
    }
}
