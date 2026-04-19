pub mod inference;
pub mod game_task;
pub mod coordinator;
pub mod evaluation;
pub mod champion;
pub mod pgn;

pub use inference::{
    InferenceRequest, InferenceBackend, RandomBackend, SwappableBackend,
    InferenceBatcher, BatcherConfig, ChannelEvaluator,
};
pub use game_task::{GameConfig, play_game, DualGameOutcome};
pub use coordinator::{SelfPlayConfig, SelfPlayCoordinator};
pub use evaluation::{EvaluationConfig, EvaluationTask, RandomEvaluator};
pub use champion::ChampionStore;
