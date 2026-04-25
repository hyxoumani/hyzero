//! Gumbel AlphaZero root-level action selection.
//!
//! Replaces PUCT-based root sampling with Gumbel-Top-K + sequential halving,
//! per Danihelka et al. "Policy improvement by planning with Gumbel" (ICLR 2022).
//!
//! Mechanism: at the root, sample one Gumbel(0) value per legal action, take
//! the top-K candidates by `logit + g`, then run sequential halving — each
//! round simulates each surviving candidate, then halves the set by
//! `logit + g + sigma(q)`. The final survivor is the chosen action.
//!
//! Why: PUCT with sharp priors and small Q-gradients (symmetric self-play)
//! starves low-prior moves of visits; Gumbel guarantees every top-K candidate
//! gets baseline sims regardless of prior, breaking the prior-only feedback loop.

use rand::Rng;

/// Default sigma transformation constants (paper recommends c_visit=50,
/// c_scale=1.0 for chess at typical sim budgets).
const DEFAULT_C_VISIT: f32 = 50.0;
const DEFAULT_C_SCALE: f32 = 1.0;

/// Sample n values from the standard Gumbel(0, 1) distribution.
///
/// Uses inverse-CDF sampling: G = -log(-log(U)) where U ~ Uniform(0, 1).
pub fn sample_gumbel(n: usize) -> Vec<f32> {
    let mut rng = rand::rng();
    (0..n)
        .map(|_| {
            // Avoid log(0) — clamp the uniform sample away from 0 and 1.
            let u: f32 = rng.random::<f32>().clamp(1e-9, 1.0 - 1e-9);
            -(-u.ln()).ln()
        })
        .collect()
}

/// Q-value transformation `sigma(q)`. Scales by `c_visit + max_visits` so
/// that Q's influence grows with how much the tree has been explored — at
/// the start of search, sigma(q) ≈ 0 (priors dominate); after many sims,
/// sigma(q) reflects the actual Q gradient (Q dominates).
///
/// Per-paper formula: `sigma(q) = (c_visit + max_visits) * c_scale * q`.
pub fn sigma_q(q: f32, max_visits: u32) -> f32 {
    (DEFAULT_C_VISIT + max_visits as f32) * DEFAULT_C_SCALE * q
}

/// Choose the number of considered actions for sequential halving.
///
/// Defaults to 16 (paper-recommended for chess at sim_budget=200), capped
/// at the number of legal actions. Always returns at least 2 (or n if n<2).
pub fn default_top_k(n_legal: usize) -> usize {
    n_legal.min(16).max(2.min(n_legal))
}

/// Number of sequential-halving rounds for K candidates.
///
/// `ceil(log2(K))` rounds; at least 1.
pub fn num_rounds(k: usize) -> usize {
    if k <= 1 {
        return 1;
    }
    let lg = (k as f32).log2().ceil() as usize;
    lg.max(1)
}

/// Per-round visit budget for sequential halving.
///
/// Paper formula: `floor(total_sims / (num_rounds * k_round))` per candidate,
/// where k_round is the size of the surviving set in this round. We pre-compute
/// the per-round candidate count and let the caller derive sims-per-candidate.
pub fn sims_per_candidate(total_sims: u32, num_rounds: usize, k_round: usize) -> u32 {
    if num_rounds == 0 || k_round == 0 {
        return 1;
    }
    let per = total_sims as usize / (num_rounds * k_round);
    per.max(1) as u32
}

/// Compute the Gumbel-improved policy at the root.
///
/// Returns `softmax(logit + sigma(completed_q))` over the considered set,
/// with zero mass on non-considered actions. This is what the paper recommends
/// as the training target — it is provably ≥ the original policy under
/// MCTS regret bounds.
///
/// Currently unused (we use raw visit counts as the training target for
/// implementation simplicity); kept for future migration.
#[allow(dead_code)]
pub fn improved_policy(
    logits: &[f32],
    completed_q: &[f32],
    considered_mask: &[bool],
    max_visits: u32,
) -> Vec<f32> {
    let n = logits.len();
    let mut out = vec![0.0f32; n];
    let mut max_log = f32::NEG_INFINITY;
    for i in 0..n {
        if considered_mask[i] {
            let z = logits[i] + sigma_q(completed_q[i], max_visits);
            if z > max_log {
                max_log = z;
            }
        }
    }
    if max_log == f32::NEG_INFINITY {
        return out;
    }
    let mut sum = 0.0f32;
    for i in 0..n {
        if considered_mask[i] {
            let z = logits[i] + sigma_q(completed_q[i], max_visits);
            let e = (z - max_log).exp();
            out[i] = e;
            sum += e;
        }
    }
    if sum > 0.0 {
        for v in &mut out {
            *v /= sum;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_gumbel_distribution() {
        let n = 100_000;
        let samples = sample_gumbel(n);
        let mean: f32 = samples.iter().sum::<f32>() / n as f32;
        // Standard Gumbel mean ≈ 0.5772 (Euler-Mascheroni constant).
        assert!((mean - 0.5772).abs() < 0.05, "mean = {}", mean);
    }

    #[test]
    fn test_sigma_grows_with_visits() {
        let q = 0.5;
        let s_low = sigma_q(q, 0);
        let s_high = sigma_q(q, 200);
        assert!(s_high > s_low);
    }

    #[test]
    fn test_default_top_k() {
        assert_eq!(default_top_k(20), 16);
        assert_eq!(default_top_k(8), 8);
        assert_eq!(default_top_k(1), 1);
    }

    #[test]
    fn test_num_rounds() {
        assert_eq!(num_rounds(16), 4);
        assert_eq!(num_rounds(8), 3);
        assert_eq!(num_rounds(4), 2);
        assert_eq!(num_rounds(2), 1);
        assert_eq!(num_rounds(1), 1);
    }

    #[test]
    fn test_improved_policy_shape() {
        let logits = vec![0.5, 0.3, 0.1, 0.05];
        let q = vec![0.0, 0.0, 0.0, 0.0];
        let mask = vec![true, true, true, false];
        let p = improved_policy(&logits, &q, &mask, 10);
        // mass on first 3, none on 4th
        assert!(p[3] == 0.0);
        let sum: f32 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}
