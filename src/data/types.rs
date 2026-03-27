/// Index into the 4096 action space (from_square * 64 + to_square)
pub type ActionIndex = u16;

/// Probability distribution over the action space
pub type Policy = Vec<f32>;

/// Total number of actions in the action space (64 * 64)
pub const NUM_ACTIONS: usize = 4096;

/// Number of observation planes for the representation network
pub const NUM_OBS_PLANES: usize = 19;

/// Board observation encoded as 19 float planes (8x8 each).
///
/// Plane layout:
///   0-5:   White pieces (Pawn, Knight, Bishop, Rook, Queen, King)
///   6-11:  Black pieces (same order)
///   12-15: Castling rights (WK, WQ, BK, BQ) — constant plane per right
///   16:    En passant target square (one-hot)
///   17:    Side to move (all 1.0 = white, all 0.0 = black)
///   18:    Halfmove clock (all squares = clock / 100.0)
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
}

/// Complete trajectory of a played game — moved to replay buffer when game ends.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GameTrajectory {
    pub steps: Vec<StepRecord>,
    pub game_outcome: f32,
    pub model_version: u64,
}
