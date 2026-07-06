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
/// Layout: 8 positions * 12 piece planes each = 96, plus 6 game-state planes, plus
/// 8 lc0-style repetition planes (one per history position).
/// Current position: planes 0-11 (pieces), planes 96-101 (castling x4, EP, halfmove).
/// Past positions 1-7: planes 12-95 (12 piece planes each, no castling/EP for history).
/// Repetition: planes 102-109 (one per history position; constant-fill 1.0 if that
/// position had occurred before in the game, 0.0 otherwise).
/// Side-to-move is NOT encoded (removed in Phase 3b for color-invariant observations).
pub const NUM_OBS_PLANES: usize = 110;

/// Board observation encoded as 110 float planes (8x8 each).
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
///   102:    Current position repetition flag (all 1.0 if the position had occurred
///           before in the game, else 0.0)
///   103-109: Past positions 1-7 repetition flags (captured at snapshot time)
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
    /// True if this position had already occurred at least once earlier in the
    /// game at the time it was the current position (zobrist repeat count >= 2).
    /// Captured at snapshot time and fed into the lc0-style repetition planes.
    #[serde(default)]
    pub repeated: bool,
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
    /// Optional per-step tablebase WDL value overrides, one entry per `steps`
    /// element, in the SAME side-to-move POV as the computed TD targets. `Some(v)`
    /// on a step forces that step's value target to `v` (tablebase tail-rescoring);
    /// `None` keeps the normal TD/outcome target. Empty when tablebase rescoring is
    /// inactive (the common case) — a length-0 vec means "no overrides at all".
    /// `#[serde(default)]` keeps pre-rescore on-disk `ReplayBuffer.bin` loadable
    /// (a missing field deserializes to an empty vec ⇒ no override).
    #[serde(default)]
    pub tb_values: Vec<Option<f32>>,
}

/// Crate-wide lock serializing tests that mutate `std::env`.
///
/// The process environment is global, so a per-module mutex only serializes
/// env access *within* that module — tests in different modules still race.
/// Every test that calls `set_var`/`remove_var` must acquire this single lock
/// (via [`TestEnvGuard`]) so all such tests run mutually exclusive crate-wide.
#[cfg(test)]
pub(crate) static TEST_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// RAII guard for env-mutating tests.
///
/// Acquiring one holds the crate-wide [`TEST_ENV_LOCK`] (recovering from poison
/// so a panicking test cannot cascade `PoisonError` into others) and snapshots
/// the named env vars. On drop — including unwind on a failed assertion — every
/// named var is restored to its pre-test value (or removed if it was unset), so
/// test execution order cannot leak state between tests.
///
/// Usage:
/// ```ignore
/// let _env = TestEnvGuard::new(&["HYZERO_RESIGN_CONSECUTIVE", "HYZERO_RESIGN_MIN_PLY"]);
/// std::env::set_var("HYZERO_RESIGN_CONSECUTIVE", "2");
/// ```
#[cfg(test)]
pub(crate) struct TestEnvGuard {
    _lock: std::sync::MutexGuard<'static, ()>,
    saved: Vec<(String, Option<String>)>,
}

#[cfg(test)]
impl TestEnvGuard {
    /// Lock the crate-wide env mutex and snapshot the current values of `keys`.
    pub(crate) fn new(keys: &[&str]) -> Self {
        let _lock = TEST_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let saved = keys
            .iter()
            .map(|&k| (k.to_string(), std::env::var(k).ok()))
            .collect();
        Self { _lock, saved }
    }
}

#[cfg(test)]
impl Drop for TestEnvGuard {
    fn drop(&mut self) {
        // Restore each snapshotted var so order-independence holds even on panic.
        // Still serialized by the held TEST_ENV_LOCK guard.
        for (key, value) in &self.saved {
            match value {
                Some(v) => std::env::set_var(key, v),
                None => std::env::remove_var(key),
            }
        }
    }
}
