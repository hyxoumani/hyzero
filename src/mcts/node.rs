use crate::data::{HiddenState, ActionIndex};

/// A single node in the MCTS tree.
///
/// Children are indexed by position in `legal_actions` — `children[i]` corresponds
/// to taking action `legal_actions[i]`. Unexpanded children are `None`.
#[derive(Debug, Clone)]
pub struct MCTSNode {
    pub hidden_state: HiddenState,
    pub visit_count: u32,
    pub total_value: f32,
    pub reward: f32,
    pub priors: Vec<f32>,
    pub children: Vec<Option<Box<MCTSNode>>>,
    pub legal_actions: Vec<ActionIndex>,
}

impl MCTSNode {
    /// Create a new leaf node with the given hidden state, priors, and legal actions.
    pub fn new(
        hidden_state: HiddenState,
        policy: &[f32],
        legal_actions: Vec<ActionIndex>,
        reward: f32,
    ) -> Self {
        // Extract priors for legal actions only, then renormalize
        let mut priors: Vec<f32> = legal_actions
            .iter()
            .map(|&a| policy[a as usize].max(0.0))
            .collect();

        let sum: f32 = priors.iter().sum();
        if sum > 0.0 {
            for p in &mut priors {
                *p /= sum;
            }
        } else if !priors.is_empty() {
            // Uniform fallback if policy gives zero mass to all legal moves
            let uniform = 1.0 / priors.len() as f32;
            priors.fill(uniform);
        }

        let num_children = legal_actions.len();
        Self {
            hidden_state,
            visit_count: 0,
            total_value: 0.0,
            reward,
            priors,
            children: vec![None; num_children],
            legal_actions,
        }
    }

    /// Q-value: average backpropagated value. Returns 0.0 if unvisited.
    pub fn q_value(&self) -> f32 {
        if self.visit_count > 0 {
            self.total_value / self.visit_count as f32
        } else {
            0.0
        }
    }
}
