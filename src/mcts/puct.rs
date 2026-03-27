use crate::mcts::node::MCTSNode;

/// Compute the PUCT score for a child action.
///
/// score(a) = Q(s,a) + c * P(s,a) * sqrt(N_parent) / (1 + N(a))
///
/// - Q = average value = total_value / visit_count (0.0 if unvisited)
/// - P = prior probability from the policy network
/// - N_parent = total visits to the parent
/// - N(a) = visits to this child
/// - c = exploration constant
pub fn puct_score(q_value: f32, prior: f32, parent_visits: u32, child_visits: u32, c: f32) -> f32 {
    let exploration = c * prior * (parent_visits as f32).sqrt() / (1.0 + child_visits as f32);
    q_value + exploration
}

/// Select the child index with the highest PUCT score.
/// Only considers legal actions that have entries in the children array.
/// Returns the index into `node.children` / `node.legal_actions`.
pub fn select_child(node: &MCTSNode, c: f32) -> usize {
    let parent_visits = node.visit_count;
    let mut best_idx = 0;
    let mut best_score = f32::NEG_INFINITY;

    for (i, child_opt) in node.children.iter().enumerate() {
        let (q, child_visits) = match child_opt {
            Some(child) => {
                let q = if child.visit_count > 0 {
                    child.total_value / child.visit_count as f32
                } else {
                    0.0
                };
                (q, child.visit_count)
            }
            // Unexpanded child: Q=0, visits=0 — exploration term dominates
            None => (0.0, 0),
        };

        let prior = node.priors[i];
        let score = puct_score(q, prior, parent_visits, child_visits, c);

        if score > best_score {
            best_score = score;
            best_idx = i;
        }
    }

    best_idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_puct_score_unvisited_child() {
        // Unvisited child with high prior should have high exploration score
        let score = puct_score(0.0, 0.8, 100, 0, 1.5);
        // exploration = 1.5 * 0.8 * sqrt(100) / (1 + 0) = 1.5 * 0.8 * 10 = 12.0
        assert!((score - 12.0).abs() < 1e-5);
    }

    #[test]
    fn test_puct_score_visited_child() {
        // Visited child: exploitation + reduced exploration
        let score = puct_score(0.5, 0.3, 100, 10, 1.5);
        // exploration = 1.5 * 0.3 * 10 / 11 ≈ 0.4091
        // total ≈ 0.5 + 0.4091 = 0.9091
        assert!((score - 0.9091).abs() < 0.01);
    }

    #[test]
    fn test_puct_exploration_decreases_with_visits() {
        let score_low = puct_score(0.5, 0.5, 100, 1, 1.5);
        let score_high = puct_score(0.5, 0.5, 100, 50, 1.5);
        assert!(score_low > score_high);
    }

    #[test]
    fn test_select_child_picks_unvisited_high_prior() {
        // Create a node with two children: one visited, one not
        let node = MCTSNode {
            hidden_state: crate::data::HiddenState::new(64),
            visit_count: 10,
            total_value: 5.0,
            reward: 0.0,
            priors: vec![0.2, 0.8],
            children: vec![
                Some(Box::new(MCTSNode {
                    hidden_state: crate::data::HiddenState::new(64),
                    visit_count: 8,
                    total_value: 4.0,
                    reward: 0.0,
                    priors: vec![],
                    children: vec![],
                    legal_actions: vec![],
                })),
                None, // unvisited
            ],
            legal_actions: vec![0, 1],
        };

        let selected = select_child(&node, 1.5);
        // Unvisited child with prior=0.8 should win over visited child
        assert_eq!(selected, 1);
    }
}
