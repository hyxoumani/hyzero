use crate::data::{ActionIndex, HiddenState, Policy};
use crate::mcts::evaluator::Evaluator;
use crate::mcts::node::MCTSNode;
use crate::mcts::puct::select_child;

/// Return the top-K action indices by policy weight, descending.
/// For MuZero's internal MCTS nodes we don't have a ground-truth legality mask,
/// so we approximate legality by taking the K highest-prior actions. K=64 for chess.
fn top_k_actions(policy: &[f32], k: usize) -> Vec<crate::data::ActionIndex> {
    let k = k.min(policy.len());
    if k == 0 {
        return Vec::new();
    }
    let mut indexed: Vec<(crate::data::ActionIndex, f32)> = policy
        .iter()
        .enumerate()
        .map(|(i, &p)| (i as crate::data::ActionIndex, p))
        .collect();
    // Partial sort: find the top-K by prior (descending). select_nth_unstable_by
    // partitions so the first k elements are the top-k (in arbitrary order).
    if k < indexed.len() {
        indexed.select_nth_unstable_by(k - 1, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        indexed.truncate(k);
    }
    indexed.into_iter().map(|(a, _)| a).collect()
}

/// Exploration fraction for Dirichlet noise at the root.
const NOISE_EPSILON: f32 = 0.25;
/// Dirichlet concentration parameter (alpha) for chess.
/// AlphaZero paper: 0.3 for chess, 0.15 for shogi, 0.03 for Go.
const NOISE_ALPHA: f32 = 0.3;

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
    let d = alpha + 1.0 - 1.0 / 3.0; // alpha+1 shifted for M-T
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
pub struct MCTSTree {
    root: MCTSNode,
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

                    // MuZero internal nodes: top-K=64 action candidates by prior policy.
                    // (Real legality isn't computable from the hidden state alone.)
                    let child_actions = top_k_actions(&policy, 64);
                    let child = MCTSNode::new(new_hidden, &policy, child_actions, reward);
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

    /// Backpropagate `value` up the tree along `path`.
    ///
    /// Sign convention (matching PUCT in puct.rs, which uses `child.q_value` directly):
    /// - Each node stores Q from its PARENT's perspective.
    /// - The root has no parent; it stores Q from its own player's perspective.
    /// - `value` is the leaf evaluation from the LEAF's player's perspective.
    ///
    /// For a two-player alternating game, the leaf's player matches the root's player
    /// iff the path length D is even. Walking down: root and depth-1 share the same
    /// POV (root's own player), and from depth 2 onwards the stored POV flips at each step.
    fn backpropagate(&mut self, path: &[usize], value: f32) {
        let d_path = path.len();
        // Leaf → root: flip D times. Root's own-POV sign: +1 if D even, -1 if D odd.
        let mut sign: f32 = if d_path.is_multiple_of(2) { 1.0 } else { -1.0 };

        self.root.visit_count += 1;
        self.root.total_value += sign * value;

        let mut node = &mut self.root;
        for (i, &idx) in path.iter().enumerate() {
            // Depth-1 child stores Q from root's POV (same as root's own).
            // Depth-d (d≥2) stores Q from depth-(d-1)'s POV, flipped each step.
            if i >= 1 {
                sign = -sign;
            }
            let child = node.children[idx].as_mut().unwrap();
            child.visit_count += 1;
            child.total_value += sign * value;
            node = node.children[idx].as_mut().unwrap();
        }
    }

    /// Extract the normalized visit count distribution over legal actions.
    /// Returns a vector of the same length as root.legal_actions.
    pub fn extract_visit_distribution(&self) -> Vec<f32> {
        let total: u32 = self
            .root
            .children
            .iter()
            .map(|c| c.as_ref().map_or(0, |n| n.visit_count))
            .sum();

        if total == 0 {
            return vec![0.0; self.root.legal_actions.len()];
        }

        self.root
            .children
            .iter()
            .map(|c| c.as_ref().map_or(0, |n| n.visit_count) as f32 / total as f32)
            .collect()
    }

    /// Root Q-value estimate.
    pub fn root_value(&self) -> f32 {
        self.root.q_value()
    }

    /// Select an action based on visit counts and temperature.
    /// temperature=0 picks the most-visited action deterministically.
    /// temperature>0 samples proportionally to visit_count^(1/temperature).
    pub fn select_action(&self, temperature: f32) -> ActionIndex {
        if self.root.legal_actions.is_empty() {
            return 0;
        }

        let visits: Vec<f32> = self
            .root
            .children
            .iter()
            .map(|c| c.as_ref().map_or(0, |n| n.visit_count) as f32)
            .collect();

        if temperature <= f32::EPSILON {
            // Deterministic: pick highest visit count
            let best_idx = visits
                .iter()
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
    use crate::data::{ActionIndex, BoardObservation, HiddenState, Policy, NUM_ACTIONS};
    use crate::mcts::evaluator::Evaluator;
    use async_trait::async_trait;

    /// Mock evaluator that returns uniform policy and 0.5 value.
    struct MockEvaluator;

    #[async_trait]
    impl Evaluator for MockEvaluator {
        async fn root_setup(
            &self,
            _obs: &BoardObservation,
            _legal_mask: &[bool],
        ) -> (HiddenState, Policy, f32) {
            let policy = vec![1.0 / NUM_ACTIONS as f32; NUM_ACTIONS];
            (HiddenState::new(64), policy, 0.5)
        }

        async fn expand_leaf(
            &self,
            _hs: &HiddenState,
            _action: ActionIndex,
        ) -> (HiddenState, f32, Policy, f32) {
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

        let mut tree = MCTSTree::new(HiddenState::new(64), &policy, 0.5, legal_actions, config);

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

        let mut tree = MCTSTree::new(HiddenState::new(64), &policy, 0.5, legal_actions, config);

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

        let mut tree = MCTSTree::new(HiddenState::new(64), &policy, 0.5, legal_actions, config);

        tree.run_simulations(&MockEvaluator).await;

        // Temperature 0 should always pick the same action
        let action1 = tree.select_action(0.0);
        let action2 = tree.select_action(0.0);
        assert_eq!(action1, action2);
    }

    #[tokio::test]
    async fn test_tree_descends_past_depth_one() {
        // With top-K children, at least one depth-1 child should itself have expanded
        // children after enough simulations. Depth-1-only trees fail this.
        let policy = vec![1.0 / NUM_ACTIONS as f32; NUM_ACTIONS];
        let legal_actions: Vec<ActionIndex> = (0..5).collect();
        let config = MCTSConfig {
            num_simulations: 200,
            exploration_constant: 1.5,
        };

        let mut tree = MCTSTree::new(HiddenState::new(64), &policy, 0.5, legal_actions, config);

        tree.run_simulations(&MockEvaluator).await;

        // Find any depth-1 child that has a depth-2 child expanded.
        let has_grandchild = tree.root.children.iter().any(|c| {
            c.as_ref()
                .map(|child| child.children.iter().any(|gc| gc.is_some()))
                .unwrap_or(false)
        });

        assert!(
            has_grandchild,
            "MCTS tree should descend past depth 1 after 200 sims; found no grandchildren"
        );
    }

    #[tokio::test]
    async fn test_backpropagate_alternates_signs() {
        // Custom evaluator: returns a fixed non-zero value so we can inspect signs.
        struct FixedValueEvaluator;

        #[async_trait]
        impl Evaluator for FixedValueEvaluator {
            async fn root_setup(
                &self,
                _obs: &BoardObservation,
                _legal_mask: &[bool],
            ) -> (HiddenState, Policy, f32) {
                let policy = vec![1.0 / NUM_ACTIONS as f32; NUM_ACTIONS];
                (HiddenState::new(64), policy, 0.0) // root starts at 0 to isolate backprop
            }

            async fn expand_leaf(
                &self,
                _hs: &HiddenState,
                _action: ActionIndex,
            ) -> (HiddenState, f32, Policy, f32) {
                let policy = vec![1.0 / NUM_ACTIONS as f32; NUM_ACTIONS];
                (HiddenState::new(64), 0.0, policy, 1.0) // every leaf evaluates to +1 (from leaf's POV)
            }
        }

        let policy = vec![1.0 / NUM_ACTIONS as f32; NUM_ACTIONS];
        let legal_actions: Vec<ActionIndex> = (0..3).collect();
        let config = MCTSConfig {
            num_simulations: 100,
            exploration_constant: 1.5,
        };

        let mut tree = MCTSTree::new(
            HiddenState::new(64),
            &policy,
            0.0, // root_value = 0, so root.total_value starts at 0
            legal_actions,
            config,
        );

        tree.run_simulations(&FixedValueEvaluator).await;

        // Every depth-1 child's Q should be NEGATIVE (leaf POV is opponent; stored
        // from root's POV = -leaf_value = -1). Q = total_value / visit_count.
        // With leaf_value = +1 always, every visit of a depth-1 child contributes
        // -1 to its stored total_value (convention B for d=1, d_path=1: sign at d=1 is -1).
        // BUT some visits came from depth-2 paths (d_path=2), where depth-1 child
        // gets +value. Net sign depends on mix.
        //
        // A cleaner invariant: depth-2 grandchildren's Q should have the OPPOSITE
        // sign of their depth-1 parent's Q contribution from those same visits.
        // Simplest check: find any expanded grandchild and verify its total_value
        // has a different average sign than a pure depth-1-only backprop would give.
        //
        // Concrete check: for at least one (parent, child) pair in the tree where
        // both have visit_count > 0, the stored total_values should not both match
        // the "all negative" pattern of the buggy code (which stored -value for every
        // descendant regardless of depth).

        // Count: if buggy (every child gets -value), all depth-≥1 nodes would have
        // total_value == -visit_count (i.e., mean -1). If fixed, depth-2 nodes'
        // contributions from depth-2 backups will be +value, pushing their mean above -1.
        let mut found_positive_contribution = false;
        for d1 in tree.root.children.iter().flatten() {
            for d2 in d1.children.iter().flatten() {
                if d2.visit_count > 0 {
                    let q = d2.total_value / d2.visit_count as f32;
                    // In convention B with leaf_value=+1, a depth-2 child (depth=2 from
                    // a d_path=2 backup) stores sign = -1. But the same node can also
                    // be visited via longer paths (d_path=3,4,...), producing varied signs.
                    // The key is: at least some non-(-1) contributions should appear.
                    if q > -0.99 {
                        found_positive_contribution = true;
                    }
                }
            }
        }
        assert!(
            found_positive_contribution,
            "Expected at least one depth-2 node whose Q is not stuck at -1 \
             (the buggy code's signature). All depth-2 Q values were ≤ -0.99, \
             which means backprop didn't alternate signs as intended."
        );
    }
}
