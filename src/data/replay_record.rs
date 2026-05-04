use crate::data::ActionIndex;

/// One ply's MCTS dump for the replay viewer. Parallel to `StepRecord` but
/// strips the observation tensor and adds the per-child diagnostics that the
/// trainer doesn't need but the viewer does.
///
/// All per-child arrays (`legal_moves`, `child_visits`, `priors`, `q_values`)
/// are the same length and indexed identically: position `i` describes the
/// child reached by playing `legal_moves[i]`. Actions are stored in
/// current-player coordinate space (Black-flipped) — same convention as
/// `StepRecord.action`. The viewer un-flips for display.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ReplayRecord {
    pub action: ActionIndex,
    pub legal_moves: Vec<ActionIndex>,
    pub child_visits: Vec<u32>,
    pub priors: Vec<f32>,
    pub q_values: Vec<f32>,
    pub root_value: f32,
    pub white_to_move: bool,
}

/// Persisted replay of one game: per-ply MCTS dumps plus the header needed to
/// reconstruct the board and recompute PUCT scores.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ReplayFile {
    pub steps: Vec<ReplayRecord>,
    pub game_outcome: f32,
    pub model_version: u64,
    pub is_draw: bool,
    /// Starting FEN. `None` means the standard initial position.
    pub starting_fen: Option<String>,
    /// PUCT exploration constant used at search time, so the viewer can recompute
    /// the U term exactly: `U = c_puct * prior * sqrt(parent_visits) / (1 + child_visits)`.
    pub c_puct: f32,
}
