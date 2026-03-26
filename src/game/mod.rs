pub mod board;
pub mod playerobj;
pub mod externplayer;
pub mod history;

pub use playerobj::Player;
pub use board::GameBoard;
use crate::{PieceType, Square, CastleOption};
use crate::PrecomputedItems;
use std::sync::Arc;

#[derive(Default, Copy, Debug, Clone)]
pub struct Move {
    pub from: Square,
    pub to: Square,
    pub promotion_piece_type: Option<PieceType>,
    pub castle_option: Option<CastleOption>,
    pub en_passant: bool,
}

#[derive(Debug, Clone)]
pub struct GameState {
    game_state: GameBoard,
}

impl GameState {
    pub fn init_game_state(precomputed_items: Arc<PrecomputedItems>, player1: Player, player2: Player) -> Self {
        Self {
            game_state: GameBoard::init_game_board(precomputed_items, player1, player2),
        }
    }

    pub fn start_game(&mut self) {
        self.game_state.start_game();
    }
}
