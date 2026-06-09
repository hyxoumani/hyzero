use crate::mcts::node::{MCTSNode, MinMaxStats};

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

/// Compute the PUCT score components for a child action.
///
/// Returns `(q_value, exploration_term, total_score)` so callers can log
/// each component independently without re-implementing the formula.
/// Satisfies: `total_score == q_value + exploration_term`.
pub fn puct_score_detail(
    q_value: f32,
    prior: f32,
    parent_visits: u32,
    child_visits: u32,
    c: f32,
) -> (f32, f32, f32) {
    let exploration = c * prior * (parent_visits as f32).sqrt() / (1.0 + child_visits as f32);
    let score = q_value + exploration;
    (q_value, exploration, score)
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

/// Select the child index with the highest PUCT score using MinMaxStats Q
/// normalization (canonical AlphaZero/MuZero) instead of raw Q.
///
/// Differs from [`select_child`] only in the Q term fed to the exploration
/// formula; the prior/visit/exploration math, tie-breaking, and signature
/// shape are otherwise identical:
/// - VISITED child: `q = stats.normalize(child_q)`.
/// - UNVISITED child: First-Play Urgency. `q = parent_q_norm - fpu_reduction`,
///   where `parent_q_norm = stats.normalize(node.q_value())`. While the window
///   is degenerate (`stats.is_degenerate()` — the first visits of each search,
///   when ≤1 distinct Q has been seen) the reduction is suppressed and FPU
///   falls back to `parent_q_norm` (a pass-through of the parent's raw Q), so
///   unvisited children are not uniformly pushed negative and exploration does
///   not dominate everything.
///
/// `fpu_reduction` is the (already env-resolved, clamped) reduction constant.
pub fn select_child_normalized(
    node: &MCTSNode,
    c: f32,
    stats: &MinMaxStats,
    fpu_reduction: f32,
) -> usize {
    let parent_visits = node.visit_count;
    let parent_q_norm = stats.normalize(node.q_value());
    let degenerate = stats.is_degenerate();
    let mut best_score = f32::NEG_INFINITY;
    let mut tied: Vec<usize> = Vec::new();
    const TIE_EPSILON: f32 = 1e-6;

    for (i, child_opt) in node.children.iter().enumerate() {
        let (q, child_visits) = match child_opt {
            Some(child) if child.visit_count > 0 => {
                let child_q = child.total_value / child.visit_count as f32;
                (stats.normalize(child_q), child.visit_count)
            }
            // Unexpanded or zero-visit child: First-Play Urgency.
            _ => {
                let fpu = if degenerate {
                    // No scale yet — inherit the parent's value with no pessimism.
                    parent_q_norm
                } else {
                    parent_q_norm - fpu_reduction
                };
                (fpu, 0)
            }
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
    fn test_puct_score_detail_components_sum_to_score() {
        // Components (q, exploration, total) must satisfy: q + exploration == total,
        // and total must equal the value returned by puct_score with the same args.
        let cases: &[(f32, f32, u32, u32, f32)] = &[
            (0.0, 0.8, 100, 0, 1.5),
            (0.5, 0.3, 100, 10, 1.5),
            (0.5, 0.5, 100, 50, 1.5),
            (-0.3, 0.1, 25, 3, 2.0),
        ];
        for &(q, p, np, nc, c) in cases {
            let (q_out, expl, total) = puct_score_detail(q, p, np, nc, c);
            let reference = puct_score(q, p, np, nc, c);
            assert!(
                (q_out + expl - total).abs() < 1e-6,
                "components don't sum: q={q_out} expl={expl} total={total}"
            );
            assert!(
                (total - reference).abs() < 1e-6,
                "detail total {total} != puct_score {reference}"
            );
        }
    }

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

    /// Helper: a visited child with the given mean Q (visit_count=1).
    fn visited_child(q: f32) -> Option<Box<MCTSNode>> {
        Some(Box::new(MCTSNode {
            hidden_state: crate::data::HiddenState::new(64),
            visit_count: 1,
            total_value: q,
            reward: 0.0,
            priors: vec![],
            children: vec![],
            legal_actions: vec![],
        }))
    }

    #[test]
    fn fpu_pessimism_deprioritizes_unvisited_when_parent_low() {
        // Non-degenerate window [-1, 1]. Parent Q is low (-0.8 → norm 0.1), so an
        // unvisited child gets FPU = 0.1 - 0.25 = -0.15. The visited child's Q is
        // only modestly higher (-0.6 → norm 0.2). Constants are tuned so the
        // exploration term sits in the 0.15-wide band where FPU pessimism is
        // decisive:
        //   visited (norm 0.2, N=20): 0.2 + small_expl ≈ 0.214
        //   FPU unvisited:            -0.15 + 0.30      = 0.150  → visited wins
        // With legacy select_child (unvisited Q init = 0.0, no pessimism):
        //   raw unvisited:             0.0  + 0.30      = 0.300  → unvisited wins
        // So this case selects DIFFERENTLY under FPU than under raw-Q init.
        let mut stats = MinMaxStats::new();
        stats.update(-1.0);
        stats.update(1.0);

        let node = MCTSNode {
            hidden_state: crate::data::HiddenState::new(64),
            visit_count: 16,
            total_value: -12.8, // mean Q = -0.8 → normalize → 0.1
            reward: 0.0,
            priors: vec![0.5, 0.5],
            children: vec![
                {
                    // Modestly-positive visited child: mean Q = -0.6 → norm 0.2,
                    // with enough visits that its exploration term is tiny.
                    Some(Box::new(MCTSNode {
                        hidden_state: crate::data::HiddenState::new(64),
                        visit_count: 20,
                        total_value: -12.0, // mean Q = -0.6
                        reward: 0.0,
                        priors: vec![],
                        children: vec![],
                        legal_actions: vec![],
                    }))
                },
                None, // unvisited → FPU = 0.1 - 0.25 = -0.15
            ],
            legal_actions: vec![0, 1],
        };

        let selected = select_child_normalized(&node, 0.15, &stats, 0.25);
        assert_eq!(
            selected, 0,
            "FPU pessimism should keep search on the visited child over the \
             pessimistic unvisited sibling"
        );
    }

    #[test]
    fn fpu_falls_back_to_parent_value_when_window_degenerate() {
        // Degenerate window (no distinct Q seen). Unvisited children must inherit
        // parent_q_norm with NO reduction — so a high-prior unvisited child still
        // wins via the exploration term, exactly as in the legacy raw-Q path.
        let stats = MinMaxStats::new();
        assert!(stats.is_degenerate());

        let node = MCTSNode {
            hidden_state: crate::data::HiddenState::new(64),
            visit_count: 100,
            total_value: 50.0, // parent mean Q = 0.5; pass-through under degenerate
            reward: 0.0,
            priors: vec![0.2, 0.8],
            children: vec![None, None],
            legal_actions: vec![0, 1],
        };

        // Both children share FPU = parent_q_norm = 0.5 (no -0.25), so selection
        // is driven purely by prior → index 1 (the 0.8 prior) wins.
        let selected = select_child_normalized(&node, 1.5, &stats, 0.25);
        assert_eq!(
            selected, 1,
            "degenerate-window FPU must be parent value with no reduction"
        );

        // Confirm no pessimism was applied: with a reduction the FPU values would
        // still be equal, but they must equal the parent's pass-through value, so
        // a zero-prior tie-break is impossible here. Sanity-check the fallback by
        // making priors equal — the high reduction must NOT flip anything.
        let node_equal = MCTSNode {
            priors: vec![0.5, 0.5],
            ..node
        };
        // With equal FPU + equal prior, scores tie; selection is one of {0,1}.
        let sel = select_child_normalized(&node_equal, 1.5, &stats, 0.9);
        assert!(sel < 2);
    }

    #[test]
    fn normalized_selection_prefers_high_q_after_minmax() {
        // Two visited children with spread Q values; min-max normalization must
        // make the higher-Q child win. Raw select_child would agree here, but the
        // point is that the NEW fn ranks by normalized Q (FAILS to compile/exist
        // without select_child_normalized).
        let mut stats = MinMaxStats::new();
        stats.update(-0.5);
        stats.update(0.7);

        let node = MCTSNode {
            hidden_state: crate::data::HiddenState::new(64),
            visit_count: 200,
            total_value: 0.0,
            reward: 0.0,
            priors: vec![0.5, 0.5],
            children: vec![
                visited_child(-0.5), // normalize → 0.0
                visited_child(0.7),  // normalize → 1.0
            ],
            legal_actions: vec![0, 1],
        };

        let selected = select_child_normalized(&node, 1.5, &stats, 0.25);
        assert_eq!(
            selected, 1,
            "normalized selection should prefer the higher-Q child"
        );
    }
}
