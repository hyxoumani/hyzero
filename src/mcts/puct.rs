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
///
/// Ties (scores within TIE_EPSILON of each other) are broken uniformly at random
/// to prevent systematic bias toward early-indexed children (which correspond to
/// low-numbered squares via get_legal_moves iteration order).
pub fn select_child(node: &MCTSNode, c: f32) -> usize {
    let parent_visits = node.visit_count;
    let mut best_score = f32::NEG_INFINITY;
    let mut tied: Vec<usize> = Vec::new();
    const TIE_EPSILON: f32 = 1e-6;

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

        if score > best_score + TIE_EPSILON {
            best_score = score;
            tied.clear();
            tied.push(i);
        } else if (score - best_score).abs() <= TIE_EPSILON {
            tied.push(i);
        }
    }

    if tied.len() == 1 {
        tied[0]
    } else if tied.is_empty() {
        // Should never happen with non-empty children, but defensively:
        0
    } else {
        use rand::Rng;
        let mut rng = rand::rng();
        tied[rng.random_range(0..tied.len())]
    }
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

    #[test]
    fn test_select_child_breaks_ties_uniformly() {
        // All 20 children with identical priors and no visits → all scores tied.
        // Over 10000 samples, each index should be picked roughly 500 times (uniform).
        let priors = vec![0.05; 20];
        let node = MCTSNode {
            hidden_state: crate::data::HiddenState::new(64),
            visit_count: 1,
            total_value: 0.0,
            reward: 0.0,
            priors,
            children: vec![None; 20],
            legal_actions: (0..20).map(|i| i as crate::data::ActionIndex).collect(),
        };

        let mut counts = vec![0usize; 20];
        for _ in 0..10_000 {
            counts[select_child(&node, 1.5)] += 1;
        }

        for (i, &c) in counts.iter().enumerate() {
            assert!(
                (350..=650).contains(&c),
                "index {i} selected {c} times — tie-breaking not uniform (expected 500 ± 150)"
            );
        }
    }
}
