pub mod inference;
pub mod game_task;
pub mod coordinator;
pub mod training;

pub use inference::{
    InferenceRequest, InferenceBackend, RandomBackend,
    InferenceBatcher, BatcherConfig, ChannelEvaluator,
};
pub use game_task::{GameConfig, play_game};
pub use coordinator::{SelfPlayConfig, SelfPlayCoordinator};
pub use training::{TrainingConfig, TrainingThread};
