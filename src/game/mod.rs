pub mod board;
pub mod playerobj;
pub mod externplayer;

pub use externplayer::ExternPlayer;
pub use playerobj::Player;
pub use board::GameBoard;
use crate::{Bitboard, Color, PieceType, Square, Piece, CastleOption};
use crate::PrecomputedItems;
use std::sync::Arc;

#[derive(Default, Copy, Debug, Clone)]
pub struct Move{
    pub from: Square,
    pub to: Square,
    pub promotion_piece_type: Option<PieceType>,
    pub castle_option: Option<CastleOption>,
    pub en_passant: Option<bool>
}

#[derive(Debug, Clone)]
pub struct GameState {
    game_state: GameBoard,
    game_over: bool
}

impl GameState {
    pub fn init_game_state(precomputed_items: Arc<PrecomputedItems>) -> Self {
        Self{        
            game_state: GameBoard::init_game_board(precomputed_items),
            game_over: false
        }
    }

    pub fn start_game(&mut self){
        self.game_state.start_game();
    }
}