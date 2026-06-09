use crate::data::{ActionIndex, HiddenState};

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

/// Running min/max of node Q-values observed during a single search, used to
/// normalize Q into `[0, 1]` before adding the PUCT exploration term (canonical
/// AlphaZero/MuZero MinMaxStats).
///
/// `normalize` only rescales once a non-degenerate window exists (two distinct
/// Q's seen). Before that — the common early-search case where every node shares
/// the same Q — it passes `q` through unchanged so the exploration term is added
/// to raw Q exactly as the legacy path did.
#[derive(Debug, Clone, Copy)]
pub struct MinMaxStats {
    pub min: f32,
    pub max: f32,
}

impl MinMaxStats {
    /// Epsilon below which the [min, max] window is treated as degenerate
    /// (no meaningful scale yet).
    const EPS: f32 = 1e-8;

    /// Create empty stats (min = +INF, max = -INF) so the first `update`
    /// seeds both bounds.
    pub fn new() -> Self {
        Self {
            min: f32::INFINITY,
            max: f32::NEG_INFINITY,
        }
    }

    /// Fold a newly observed Q-value into the running min/max.
    pub fn update(&mut self, q: f32) {
        if q < self.min {
            self.min = q;
        }
        if q > self.max {
            self.max = q;
        }
    }

    /// True while no meaningful scale exists yet: at most one distinct Q has
    /// been seen, so `max <= min + EPS`. The empty window (min=+INF, max=-INF)
    /// is degenerate since `max - min` is `-INF <= EPS`.
    pub fn is_degenerate(&self) -> bool {
        self.max - self.min <= Self::EPS
    }

    /// Normalize `q` to `[0, 1]` via `(q - min) / (max - min)` once the window
    /// has scale; otherwise pass `q` through unchanged.
    pub fn normalize(&self, q: f32) -> f32 {
        if self.is_degenerate() {
            q
        } else {
            (q - self.min) / (self.max - self.min)
        }
    }
}

impl Default for MinMaxStats {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minmax_normalizes_to_unit_range() {
        let mut stats = MinMaxStats::new();
        stats.update(-1.0);
        stats.update(1.0);
        stats.update(0.0);
        assert!(!stats.is_degenerate());
        // min=-1, max=1 → (q+1)/2.
        assert!((stats.normalize(-1.0) - 0.0).abs() < 1e-6);
        assert!((stats.normalize(1.0) - 1.0).abs() < 1e-6);
        assert!((stats.normalize(0.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn minmax_passes_through_when_degenerate() {
        // No updates: window is empty → degenerate → identity.
        let empty = MinMaxStats::new();
        assert!(empty.is_degenerate());
        assert!((empty.normalize(0.37) - 0.37).abs() < 1e-6);

        // A single distinct Q (even repeated) keeps the window degenerate.
        let mut single = MinMaxStats::new();
        single.update(0.5);
        single.update(0.5);
        assert!(single.is_degenerate());
        assert!((single.normalize(0.5) - 0.5).abs() < 1e-6);
        assert!((single.normalize(-0.2) - (-0.2)).abs() < 1e-6);
    }
}
