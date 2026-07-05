use async_trait::async_trait;
use crate::data::{BoardObservation, HiddenState, Policy, ActionIndex};

/// Abstraction over neural network calls for MCTS.
/// Implementations may call into Python via PyO3 or return random values for testing.
#[async_trait]
pub trait Evaluator: Send + Sync {
    /// Representation + prediction: encode real board into latent space, predict policy + value.
    /// Combines h() + f() into one call.
    ///
    /// `legal_mask` is a boolean mask of length NUM_ACTIONS; `true` entries are legal moves.
    /// Implementations may use this to zero out illegal logits before softmax.
    ///
    /// The trailing `Option<f32>` is the normalized moves-left estimate `m` in
    /// [0, 1] from the network's moves-left head (lc0-style MLH). It is `Some(m)`
    /// only when the backend produced one (HYZERO_MOVES_LEFT_HEAD=1); otherwise
    /// `None`, and callers keep the neutral node default (0.5).
    async fn root_setup(&self, observation: &BoardObservation, legal_mask: &[bool]) -> (HiddenState, Policy, f32, Option<f32>);

    /// Dynamics + prediction: advance hidden state by one action, predict policy + value.
    /// Combines g() + f() into one call.
    ///
    /// The trailing `Option<f32>` is the moves-left estimate `m`; see `root_setup`.
    async fn expand_leaf(&self, hidden_state: &HiddenState, action: ActionIndex) -> (HiddenState, f32, Policy, f32, Option<f32>);
}
