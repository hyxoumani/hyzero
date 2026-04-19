/// Index into the 4672 action space (from_square * 64 + to_square, or underpromotion range)
pub type ActionIndex = u16;

/// Probability distribution over the action space
pub type Policy = Vec<f32>;

/// Total number of actions in the action space: 4096 base + 576 underpromotion slots.
/// Base encoding: from_sq * 64 + to_sq for queen promotions and all non-promotion moves.
/// Underpromotion range: 4096 + piece_idx * 192 + from_file * 24 + to_file_offset * 8 + 0
/// where piece_idx is 0=Knight, 1=Bishop, 2=Rook, and from_file is 0-7.
/// Total: 4096 + 3 * 192 = 4096 + 576 = 4672.
pub const NUM_ACTIONS: usize = 4672;

/// Number of base (non-underpromotion) actions (from_sq * 64 + to_sq encoding).
pub const NUM_BASE_ACTIONS: usize = 4096;

/// Number of underpromotion action slots (3 piece types * 8 from-files * 24 slots per file).
pub const NUM_UNDERPROMO_ACTIONS: usize = 576;

/// Number of history positions (current + 7 past) encoded in board observations.
pub const NUM_HISTORY_POSITIONS: usize = 8;

/// Number of observation planes for the representation network.
/// Layout: 8 positions * 12 piece planes each = 96, plus 6 game-state planes.
/// Current position: planes 0-11 (pieces), planes 96-101 (castling x4, EP, halfmove).
/// Past positions 1-7: planes 12-95 (12 piece planes each, no castling/EP for history).
/// Side-to-move is NOT encoded (removed in Phase 3b for color-invariant observations).
pub const NUM_OBS_PLANES: usize = 102;

/// Board observation encoded as 102 float planes (8x8 each).
///
/// Plane layout:
///   0-11:   Current position pieces (my pawn..king, opp pawn..king)
///   12-23:  Past position 1 pieces (oldest in window)
///   24-35:  Past position 2 pieces
///   ...
///   84-95:  Past position 7 pieces
///   96-99:  Castling rights (my KS, my QS, opp KS, opp QS) — current position only
///   100:    En passant target square (one-hot, rank-mirrored for Black)
///   101:    Halfmove clock (all squares = clock / 100.0)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BoardObservation {
    pub planes: Vec<f32>,
}

impl Default for BoardObservation {
    fn default() -> Self {
        Self {
            planes: vec![0.0; NUM_OBS_PLANES * 64],
        }
    }
}

/// Lightweight snapshot of a board position for history encoding.
/// Stores only the 12 piece-placement bitboards (6 types × 2 colors).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BoardSnapshot {
    /// Piece bitboards indexed by piece type (0-5: Pawn..King) for white.
    pub white_pieces_bb: [u64; 6],
    /// Piece bitboards indexed by piece type (0-5: Pawn..King) for black.
    pub black_pieces_bb: [u64; 6],
}

/// Latent state produced by the representation or dynamics network.
/// Shape: [channels, 8, 8] stored as a flat Vec<f32>.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HiddenState {
    pub data: Vec<f32>,
    pub channels: usize,
}

impl HiddenState {
    pub fn new(channels: usize) -> Self {
        Self {
            data: vec![0.0; channels * 64],
            channels,
        }
    }
}

/// One step of a played game — produced after each real move.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StepRecord {
    pub observation: BoardObservation,
    pub action: ActionIndex,
    pub visit_distribution: Vec<f32>,
    pub root_value: f32,
    pub reward: f32,
    pub legal_moves: Vec<ActionIndex>,
    /// White-to-move at the position this step records. Needed by the trainer
    /// to convert absolute-White game_outcome to step-k perspective. Stored
    /// out-of-band since plane 101 (side-to-move) was removed from observations.
    pub white_to_move: bool,
}

/// Complete trajectory of a played game — moved to replay buffer when game ends.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GameTrajectory {
    pub steps: Vec<StepRecord>,
    pub game_outcome: f32,
    pub model_version: u64,
    /// True if game ended non-decisively (stalemate, repetition, 50-move, cap, insufficient material).
    /// False only for actual checkmate. Used by trainer to apply non-zero-sum draw penalty.
    pub is_draw: bool,
}
