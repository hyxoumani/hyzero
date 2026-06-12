pub mod inference;
pub mod game_task;
pub mod coordinator;
pub mod elo;
pub mod pool;
pub mod evaluation;
pub mod champion;
pub mod pgn;
pub mod replay_writer;

pub use inference::{
    InferenceRequest, InferenceBackend, RandomBackend, SwappableBackend,
    InferenceBatcher, BatcherConfig, ChannelEvaluator, EvalError,
};
pub use game_task::{GameConfig, play_game, play_game_dual, play_game_dual_from, DualGameOutcome};
pub use coordinator::{SelfPlayConfig, SelfPlayCoordinator};
pub use elo::{expected_score, update_rating, INITIAL_RATING, K_FACTOR};
pub use pool::latest_archive_versions;
pub use evaluation::{EvaluationConfig, EvaluationTask, RandomEvaluator};
pub use champion::ChampionStore;
