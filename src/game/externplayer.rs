use crate::{Bitboard, Color};
use crate::game::Move;
use crate::PieceType;
use rand::Rng;

#[derive(Debug, Clone)]
pub struct ExternPlayer {
    board: [u64; 6],
    color: Color
}

impl ExternPlayer {
    pub fn new(board: [u64; 6], color: Color) -> Self {
        Self{
            board,
            color
        }
    }

    pub fn get_move(&self) -> String {
        let mut rng = rand::thread_rng();
        
        // Generate random squares
        let from_file = rng.gen_range(0..8);
        let from_rank = rng.gen_range(0..8);
        let to_file = rng.gen_range(0..8);
        let to_rank = rng.gen_range(0..8);
        
        // Convert to chess notation (e.g., "e2e4")
        let chess_notation = format!("{}{}{}{}", 
            (b'a' + from_file) as char, 
            from_rank + 1,
            (b'a' + to_file) as char, 
            to_rank + 1
        );
        
        // Optionally add promotion character randomly
        let promotions = ['q', 'r', 'b', 'n', ' '];
        let promotion_char = promotions[rng.gen_range(0..5)];
        
        let full_notation = if promotion_char != ' ' {
            format!("{}{}", chess_notation, promotion_char)
        } else {
            chess_notation
        };
        
        full_notation
    }


    pub fn update_board(&mut self){

    }


        

}