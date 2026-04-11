use crate::Piece;

pub struct GameHistory {
    pub move_history: Vec<String>,
    pub board_snapshots: Vec<[Option<Piece>; 64]>,
}

impl Default for GameHistory {
    fn default() -> Self {
        Self::new()
    }
}

impl GameHistory {
    pub fn new() -> Self {
        Self {
            move_history: Vec::new(),
            board_snapshots: Vec::new(),
        }
    }

    pub fn record_move(&mut self, color_prefix: &str, notation: &str, board_state: [Option<Piece>; 64]) {
        self.move_history.push(format!("{}: {}", color_prefix, notation));
        self.board_snapshots.push(board_state);
    }
}
