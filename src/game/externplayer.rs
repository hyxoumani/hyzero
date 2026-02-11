use crate::{Bitboard, Color};
use crate::game::Move;
use crate::PieceType;
use rand::Rng;
use std::io;

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
        let mut input_string = String::new();
        io::stdin().read_line(&mut input_string).expect("failed");
        
        input_string
    }


    pub fn update_board(&mut self){

    }


        

}