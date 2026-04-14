use crate::data::{HiddenState, Policy, ActionIndex};
use crate::mcts::evaluator::Evaluator;
use crate::mcts::node::MCTSNode;
use crate::mcts::puct::select_child;

/// Exploration fraction for Dirichlet noise at the root.
const NOISE_EPSILON: f32 = 0.25;
/// Dirichlet concentration parameter (alpha) for chess.
const NOISE_ALPHA: f32 = 0.03;

/// Sample from a Dirichlet(alpha, ..., alpha) distribution of length `n`.
///
/// Uses Gamma(alpha, 1) sampling via the Marsaglia-Tsang method for alpha < 1,
/// then normalizes. Returns a uniform distribution if `n == 0`.
fn dirichlet_noise(n: usize) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }

    use rand::Rng;
    let mut rng = rand::rng();

    // Gamma(alpha, 1) samples via Marsaglia-Tsang for alpha < 1:
    // Let Y ~ Gamma(alpha+1, 1) and U ~ Uniform(0,1).
    // Then X = Y * U^(1/alpha) ~ Gamma(alpha, 1).
    let alpha = NOISE_ALPHA;
    let d = alpha + 1.0 - 1.0 / 3.0;   // alpha+1 shifted for M-T
    let c = 1.0 / (9.0 * d).sqrt();

    let mut samples: Vec<f32> = (0..n)
        .map(|_| {
            // Marsaglia-Tsang for Gamma(alpha+1, 1)
            let y_gamma = loop {
                let x: f32 = rng.random::<f32>() * 6.0 - 3.0; // rough normal approx
                // Use Box-Muller for a proper normal sample
                let u1: f32 = rng.random::<f32>();
                let u2: f32 = rng.random::<f32>();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                let v = (1.0 + c * z).powi(3);
                if v <= 0.0 {
                    let _ = x; // suppress unused warning
                    continue;
                }
                let u: f32 = rng.random::<f32>();
                let z2 = z * z;
                if u < 1.0 - 0.0331 * z2 * z2 {
                    break d * v;
                }
                if u.ln() < 0.5 * z2 + d * (1.0 - v + v.ln()) {
                    break d * v;
                }
            };
            // Apply Y * U^(1/alpha) transformation
            let u: f32 = rng.random::<f32>();
            y_gamma * u.powf(1.0 / alpha)
        })
        .collect();

    let sum: f32 = samples.iter().sum();
    if sum > 0.0 {
        for s in &mut samples {
            *s /= sum;
        }
    } else {
        // Degenerate: return uniform
        let uniform = 1.0 / n as f32;
        samples.fill(uniform);
    }
    samples
}

/// Configuration for MCTS search.
#[derive(Debug, Clone)]
pub struct MCTSConfig {
    pub num_simulations: u32,
    pub exploration_constant: f32,
}

impl Default for MCTSConfig {
    fn default() -> Self {
        Self {
            num_simulations: 800,
            exploration_constant: 1.5,
        }
    }
}

/// Transient MCTS tree built fresh for each move.
/// After search, extract the visit distribution, then discard the tree.
/// Optionally carried forward via `reuse_subtree` to preserve search work.
pub struct MCTSTree {
    pub root: MCTSNode,
    config: MCTSConfig,
}

impl MCTSTree {
    /// Create a new tree with an already-evaluated root.
    pub fn new(
        root_hidden_state: HiddenState,
        root_policy: &Policy,
        root_value: f32,
        legal_actions: Vec<ActionIndex>,
        config: MCTSConfig,
    ) -> Self {
        let mut root = MCTSNode::new(root_hidden_state, root_policy, legal_actions, 0.0);
        // Root gets one visit with its initial value
        root.visit_count = 1;
        root.total_value = root_value;

        // Mix Dirichlet noise into root priors for exploration diversity.
        // P(a) = (1 - ε) * P(a) + ε * η_a, where η ~ Dir(α).
        let n = root.priors.len();
        if n > 0 {
            let noise = dirichlet_noise(n);
            for (prior, eta) in root.priors.iter_mut().zip(noise.iter()) {
                *prior = (1.0 - NOISE_EPSILON) * *prior + NOISE_EPSILON * eta;
            }
        }

        Self { root, config }
    }

    /// Run all simulations. Each simulation: select -> expand -> backpropagate.
    pub async fn run_simulations(&mut self, evaluator: &dyn Evaluator) {
        for _ in 0..self.config.num_simulations {
            // Collect the path of (child index) taken during selection
            let mut path: Vec<usize> = Vec::new();

            // Selection: walk down tree using PUCT until we hit an unexpanded child
            let mut current = &self.root as *const MCTSNode;
            loop {
                let node = unsafe { &*current };
                if node.legal_actions.is_empty() {
                    // Terminal node — backpropagate with current value
                    break;
                }

                let child_idx = select_child(node, self.config.exploration_constant);
                path.push(child_idx);

                match &node.children[child_idx] {
                    Some(child) => {
                        current = child.as_ref() as *const MCTSNode;
                    }
                    None => {
                        // Unexpanded — need to expand
                        break;
                    }
                }
            }

            // Determine if we need to expand or if we hit a terminal
            let value = if path.is_empty() {
                // Root is terminal
                self.root.q_value()
            } else {
                // Navigate to the parent of the leaf to expand
                let leaf_action_idx = *path.last().unwrap();
                let parent = self.navigate_to_parent_mut(&path);

                if parent.children[leaf_action_idx].is_some() {
                    // We selected an existing child (terminal node with no legal actions)
                    let child = parent.children[leaf_action_idx].as_ref().unwrap();
                    child.q_value()
                } else {
                    // Expand: call evaluator
                    let action = parent.legal_actions[leaf_action_idx];
                    let (new_hidden, reward, policy, value) =
                        evaluator.expand_leaf(&parent.hidden_state, action).await;

                    // For now, create leaf with empty legal actions (we don't have the game
                    // state to compute them). The self-play game task will provide legal actions.
                    // During pure MCTS testing with mock evaluators, leaves are terminal.
                    let child = MCTSNode::new(new_hidden, &policy, Vec::new(), reward);
                    parent.children[leaf_action_idx] = Some(Box::new(child));

                    value
                }
            };

            // Backpropagate: walk back up the path, updating visit counts and values
            self.backpropagate(&path, value);
        }
    }

    /// Navigate to the parent node of the last element in the path.
    /// path must have at least one element.
    fn navigate_to_parent_mut(&mut self, path: &[usize]) -> &mut MCTSNode {
        let mut node = &mut self.root;
        // Walk all but the last step
        for &idx in &path[..path.len() - 1] {
            node = node.children[idx].as_mut().unwrap();
        }
        node
    }

    /// Backpropagate value up the tree along the given path.
    fn backpropagate(&mut self, path: &[usize], value: f32) {
        // Update root
        self.root.visit_count += 1;
        self.root.total_value += value;

        // Walk down the path updating each node
        let mut node = &mut self.root;
        for &idx in path {
            let child = node.children[idx].as_mut().unwrap();
            child.visit_count += 1;
            // Negate value at each level: child is evaluated from the opponent's perspective
            child.total_value += -value;
            node = node.children[idx].as_mut().unwrap();
        }
    }

    /// Extract the normalized visit count distribution over legal actions.
    /// Returns a vector of the same length as root.legal_actions.
    pub fn extract_visit_distribution(&self) -> Vec<f32> {
        let total: u32 = self.root.children.iter()
            .map(|c| c.as_ref().map_or(0, |n| n.visit_count))
            .sum();

        if total == 0 {
            return vec![0.0; self.root.legal_actions.len()];
        }

        self.root.children.iter()
            .map(|c| c.as_ref().map_or(0, |n| n.visit_count) as f32 / total as f32)
            .collect()
    }

    /// Root Q-value estimate.
    pub fn root_value(&self) -> f32 {
        self.root.q_value()
    }

    /// Advance the tree by consuming the subtree rooted at the child for `action`.
    ///
    /// Returns `Some(new_tree)` if the child for `action` was already expanded
    /// (non-None). Returns `None` if `action` is not in `legal_actions` or the
    /// child was never expanded. Fresh Dirichlet noise is mixed into the reused
    /// root's priors to maintain exploration diversity.
    pub fn reuse_subtree(mut self, action: ActionIndex) -> Option<MCTSTree> {
        let idx = self.root.legal_actions.iter().position(|&a| a == action)?;
        let child_box = self.root.children[idx].take()?;
        let mut new_root = *child_box;
        // Mix fresh Dirichlet noise into the reused root priors.
        let n = new_root.priors.len();
        if n > 0 {
            let noise = dirichlet_noise(n);
            for (prior, eta) in new_root.priors.iter_mut().zip(noise.iter()) {
                *prior = (1.0 - NOISE_EPSILON) * *prior + NOISE_EPSILON * eta;
            }
        }
        Some(MCTSTree { root: new_root, config: self.config })
    }

    /// Select an action based on visit counts and temperature.
    /// temperature=0 picks the most-visited action deterministically.
    /// temperature>0 samples proportionally to visit_count^(1/temperature).
    pub fn select_action(&self, temperature: f32) -> ActionIndex {
        if self.root.legal_actions.is_empty() {
            return 0;
        }

        let visits: Vec<f32> = self.root.children.iter()
            .map(|c| c.as_ref().map_or(0, |n| n.visit_count) as f32)
            .collect();

        if temperature <= f32::EPSILON {
            // Deterministic: pick highest visit count
            let best_idx = visits.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            self.root.legal_actions[best_idx]
        } else {
            // Temperature-based sampling
            let inv_temp = 1.0 / temperature;
            let weights: Vec<f32> = visits.iter().map(|&v| v.powf(inv_temp)).collect();
            let total: f32 = weights.iter().sum();

            if total <= 0.0 {
                return self.root.legal_actions[0];
            }

            let threshold = rand::random::<f32>() * total;
            let mut cumulative = 0.0;
            for (i, &w) in weights.iter().enumerate() {
                cumulative += w;
                if cumulative >= threshold {
                    return self.root.legal_actions[i];
                }
            }
            *self.root.legal_actions.last().unwrap()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcts::evaluator::Evaluator;
    use crate::data::{BoardObservation, HiddenState, Policy, ActionIndex, NUM_ACTIONS};
    use async_trait::async_trait;

    /// Mock evaluator that returns uniform policy and 0.5 value.
    struct MockEvaluator;

    #[async_trait]
    impl Evaluator for MockEvaluator {
        async fn root_setup(&self, _obs: &BoardObservation, _legal_mask: &[bool]) -> (HiddenState, Policy, f32) {
            let policy = vec![1.0 / NUM_ACTIONS as f32; NUM_ACTIONS];
            (HiddenState::new(64), policy, 0.5)
        }

        async fn expand_leaf(&self, _hs: &HiddenState, _action: ActionIndex) -> (HiddenState, f32, Policy, f32) {
            let policy = vec![1.0 / NUM_ACTIONS as f32; NUM_ACTIONS];
            (HiddenState::new(64), 0.0, policy, 0.5)
        }
    }

    #[tokio::test]
    async fn test_tree_runs_simulations() {
        let policy = vec![1.0 / NUM_ACTIONS as f32; NUM_ACTIONS];
        let legal_actions: Vec<ActionIndex> = (0..20).collect();
        let config = MCTSConfig {
            num_simulations: 50,
            exploration_constant: 1.5,
        };

        let mut tree = MCTSTree::new(
            HiddenState::new(64),
            &policy,
            0.5,
            legal_actions,
            config,
        );

        let evaluator = MockEvaluator;
        tree.run_simulations(&evaluator).await;

        // Root should have 1 (initial) + 50 (simulations) = 51 visits
        assert_eq!(tree.root.visit_count, 51);
    }

    #[tokio::test]
    async fn test_visit_distribution_sums_to_one() {
        let policy = vec![1.0 / NUM_ACTIONS as f32; NUM_ACTIONS];
        let legal_actions: Vec<ActionIndex> = (0..10).collect();
        let config = MCTSConfig {
            num_simulations: 100,
            exploration_constant: 1.5,
        };

        let mut tree = MCTSTree::new(
            HiddenState::new(64),
            &policy,
            0.5,
            legal_actions,
            config,
        );

        tree.run_simulations(&MockEvaluator).await;

        let dist = tree.extract_visit_distribution();
        let sum: f32 = dist.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "Visit distribution sum: {}", sum);
    }

    #[tokio::test]
    async fn test_select_action_deterministic() {
        let policy = vec![1.0 / NUM_ACTIONS as f32; NUM_ACTIONS];
        let legal_actions: Vec<ActionIndex> = (0..5).collect();
        let config = MCTSConfig {
            num_simulations: 50,
            exploration_constant: 1.5,
        };

        let mut tree = MCTSTree::new(
            HiddenState::new(64),
            &policy,
            0.5,
            legal_actions,
            config,
        );

        tree.run_simulations(&MockEvaluator).await;

        // Temperature 0 should always pick the same action
        let action1 = tree.select_action(0.0);
        let action2 = tree.select_action(0.0);
        assert_eq!(action1, action2);
    }

    #[tokio::test]
    async fn test_reuse_subtree_returns_some_for_expanded_child() {
        let policy = vec![1.0 / NUM_ACTIONS as f32; NUM_ACTIONS];
        let legal_actions: Vec<ActionIndex> = (0..10).collect();
        let config = MCTSConfig {
            num_simulations: 10,
            exploration_constant: 1.5,
        };

        let mut tree = MCTSTree::new(
            HiddenState::new(64),
            &policy,
            0.5,
            legal_actions.clone(),
            config,
        );

        tree.run_simulations(&MockEvaluator).await;

        // Find any expanded child (non-None)
        let expanded_action = legal_actions
            .iter()
            .zip(tree.root.children.iter())
            .find(|(_, c)| c.is_some())
            .map(|(&a, _)| a)
            .expect("at least one child should be expanded after 10 sims");

        let reused = tree.reuse_subtree(expanded_action);
        assert!(reused.is_some(), "expected Some for an expanded child");
        let reused_tree = reused.unwrap();
        assert!(reused_tree.root.visit_count > 0, "reused root should have visits");
    }

    #[tokio::test]
    async fn test_reuse_subtree_returns_none_for_unexpanded_child() {
        let policy = vec![1.0 / NUM_ACTIONS as f32; NUM_ACTIONS];
        // Use many legal actions so not all will be expanded after very few sims.
        let legal_actions: Vec<ActionIndex> = (0..50).collect();
        let config = MCTSConfig {
            num_simulations: 2, // Very few — most children stay unexpanded
            exploration_constant: 1.5,
        };

        let mut tree = MCTSTree::new(
            HiddenState::new(64),
            &policy,
            0.5,
            legal_actions.clone(),
            config,
        );

        tree.run_simulations(&MockEvaluator).await;

        // Find an unexpanded child (None)
        let unexpanded_action = legal_actions
            .iter()
            .zip(tree.root.children.iter())
            .find(|(_, c)| c.is_none())
            .map(|(&a, _)| a)
            .expect("with only 2 sims and 50 children, some must remain unexpanded");

        let reused = tree.reuse_subtree(unexpanded_action);
        assert!(reused.is_none(), "expected None for an unexpanded child");
    }

    #[tokio::test]
    async fn test_reuse_subtree_returns_none_for_unknown_action() {
        let policy = vec![1.0 / NUM_ACTIONS as f32; NUM_ACTIONS];
        let legal_actions: Vec<ActionIndex> = vec![10, 20, 30];
        let config = MCTSConfig {
            num_simulations: 5,
            exploration_constant: 1.5,
        };

        let mut tree = MCTSTree::new(
            HiddenState::new(64),
            &policy,
            0.5,
            legal_actions,
            config,
        );

        tree.run_simulations(&MockEvaluator).await;

        // Action 999 is not in legal_actions
        let reused = tree.reuse_subtree(999);
        assert!(reused.is_none(), "expected None for unknown action");
    }
}
