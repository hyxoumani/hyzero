use crate::data::{ActionIndex, HiddenState, Policy};
use crate::mcts::evaluator::Evaluator;
use crate::mcts::node::{MCTSNode, MinMaxStats};
use crate::mcts::puct::{select_child, select_child_normalized};

// ---------------------------------------------------------------------------
// MCTS trace — enabled by HYZERO_MCTS_TRACE=1 (or any non-empty, non-"0" value).
//
// Design:
//   TRACE_ENABLED   — OnceLock<bool>: env read once, zero-overhead when off.
//   TRACE_WRITER_CLAIMED — AtomicUsize: CAS(0→1) lets the first tokio task claim
//                           the writer; all others short-circuit.
//   TRACE_FILE      — OnceLock<Mutex<Option<BufWriter<File>>>>: file handle that
//                     survives across root calls; truncated on first open.
//   TRACE_MOVE_CTR  — AtomicUsize: incremented each root call by the writer;
//                     private to the writer in practice (only one task writes).
//                     Capped at TRACE_MOVE_LIMIT to bound log size.
// ---------------------------------------------------------------------------
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

const TRACE_MOVE_LIMIT: usize = 500;

static TRACE_ENABLED: OnceLock<bool> = OnceLock::new();
/// 0 = unclaimed, 1 = claimed by writer.
static TRACE_WRITER_CLAIMED: AtomicUsize = AtomicUsize::new(0);
static TRACE_FILE: OnceLock<Mutex<Option<BufWriter<File>>>> = OnceLock::new();
static TRACE_MOVE_CTR: AtomicUsize = AtomicUsize::new(0);

/// Returns true if HYZERO_MCTS_TRACE is set to a non-empty value that isn't "0".
fn trace_enabled() -> bool {
    *TRACE_ENABLED.get_or_init(|| {
        std::env::var("HYZERO_MCTS_TRACE")
            .map(|v| !v.is_empty() && v != "0")
            .unwrap_or(false)
    })
}

/// Attempt to claim the writer role for this task. Returns true only for the
/// first caller (CAS 0→1 succeeds). All subsequent callers return false.
fn try_claim_writer() -> bool {
    TRACE_WRITER_CLAIMED
        .compare_exchange(0, 1, Ordering::SeqCst, Ordering::SeqCst)
        .is_ok()
}

/// Returns true if this task has already claimed the writer (CLAIMED == 1 and
/// we won the CAS at some earlier point). Since `try_claim_writer` is called
/// exactly once per `run_simulations` invocation before any writes, we instead
/// keep a thread-local sentinel to avoid re-CAS on subsequent calls.
///
/// Implementation note: tokio tasks run on OS threads; `thread_local!` is the
/// right scope for "did this task win the CAS". We record the result in a
/// thread-local so subsequent root calls from the same task skip the CAS.
fn is_writer() -> bool {
    WRITER_LOCAL.with(|w| *w.borrow())
}

use std::cell::RefCell;
thread_local! {
    static WRITER_LOCAL: RefCell<bool> = const { RefCell::new(false) };
}

/// Open (truncating) the trace log file and store the handle in TRACE_FILE.
/// Must only be called by the writer task on its first root call.
fn open_trace_file() -> bool {
    let file_lock = TRACE_FILE.get_or_init(|| Mutex::new(None));
    let mut guard = file_lock.lock().unwrap();
    if guard.is_some() {
        return true; // already open
    }
    match OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open("logs/mcts_trace.log")
    {
        Ok(f) => {
            *guard = Some(BufWriter::new(f));
            true
        }
        Err(e) => {
            eprintln!("[mcts_trace] failed to open logs/mcts_trace.log: {e}");
            false
        }
    }
}

/// Write a line to the trace file. Must only be called by the writer task.
/// The trailing newline is added here.
fn trace_write(line: &str) {
    let file_lock = TRACE_FILE.get().unwrap();
    let mut guard = file_lock.lock().unwrap();
    if let Some(w) = guard.as_mut() {
        let _ = writeln!(w, "{line}");
    }
}

/// Flush the trace file. Called after each `final_visit_dist` write.
fn trace_flush() {
    let file_lock = TRACE_FILE.get().unwrap();
    let mut guard = file_lock.lock().unwrap();
    if let Some(w) = guard.as_mut() {
        let _ = w.flush();
    }
}

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
/// Default 0.25 (AlphaZero chess). Overridable via `HYZERO_DIRICHLET_EPSILON`.
const DEFAULT_NOISE_EPSILON: f32 = 0.25;
/// Dirichlet concentration parameter (alpha) for chess.
/// AlphaZero paper: 0.3 for chess, 0.15 for shogi, 0.03 for Go.
/// Overridable via `HYZERO_DIRICHLET_ALPHA`.
const DEFAULT_NOISE_ALPHA: f32 = 0.3;

/// Read Dirichlet ε from env (cached) — fraction of root prior replaced by noise.
fn noise_epsilon() -> f32 {
    static CACHED: OnceLock<f32> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("HYZERO_DIRICHLET_EPSILON")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .filter(|v| (0.0..=1.0).contains(v))
            .unwrap_or(DEFAULT_NOISE_EPSILON)
    })
}

/// Read Dirichlet α from env (cached) — concentration of the noise distribution.
fn noise_alpha() -> f32 {
    static CACHED: OnceLock<f32> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("HYZERO_DIRICHLET_ALPHA")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .filter(|v| *v > 0.0)
            .unwrap_or(DEFAULT_NOISE_ALPHA)
    })
}

/// First-Play Urgency reduction subtracted from the parent's normalized Q for
/// unvisited children (canonical AlphaZero/MuZero). Default 0.25.
const DEFAULT_FPU_REDUCTION: f32 = 0.25;

/// True when MinMaxStats Q-normalization + FPU selection is enabled (cached).
/// Gated by `HYZERO_MCTS_QNORM`; default ON. OFF for any empty / "0" value.
fn qnorm_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("HYZERO_MCTS_QNORM")
            .map(|v| !v.is_empty() && v != "0")
            .unwrap_or(true)
    })
}

/// Read the FPU reduction from env (cached), clamped to `[0.0, 1.0]`.
/// Overridable via `HYZERO_FPU`; default `DEFAULT_FPU_REDUCTION` (0.25).
fn fpu_reduction() -> f32 {
    static CACHED: OnceLock<f32> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("HYZERO_FPU")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .map(|v| v.clamp(0.0, 1.0))
            .unwrap_or(DEFAULT_FPU_REDUCTION)
    })
}

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
    let alpha = noise_alpha();
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
    /// Whether to inject Dirichlet noise into root priors for exploration diversity.
    /// Set to `true` for self-play (exploration required) and `false` for evaluation
    /// (deterministic play preferred). Ignored when `gumbel_top_k.is_some()` since
    /// Gumbel sampling provides its own root noise.
    pub add_root_noise: bool,
    /// If `Some(k)`, use Gumbel-Top-k + sequential halving at the root in place
    /// of PUCT. Internal nodes still use PUCT. `k` is capped to legal-action count.
    /// `None` (default) preserves the original PUCT-only behavior.
    pub gumbel_top_k: Option<usize>,
}

impl Default for MCTSConfig {
    fn default() -> Self {
        Self {
            num_simulations: 800,
            exploration_constant: 1.5,
            add_root_noise: true,
            gumbel_top_k: None,
        }
    }
}

/// Per-child snapshot taken from the MCTS root after search.
///
/// All vectors are parallel to `root.legal_actions` (sorted at the call site).
/// Unexpanded children report `visit=0`, `q=0`.
pub struct RootDiagnostics {
    pub child_visits: Vec<u32>,
    pub priors: Vec<f32>,
    pub q_values: Vec<f32>,
}

/// Transient MCTS tree built fresh for each move.
/// After search, extract the visit distribution, then discard the tree.
pub struct MCTSTree {
    root: MCTSNode,
    config: MCTSConfig,
    /// Running min/max of node Q-values for PUCT normalization. Reset at the
    /// start of each `run_simulations` and updated during `backpropagate`.
    min_max: MinMaxStats,
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
        // Gated by config.add_root_noise: disabled for evaluation games AND when
        // Gumbel sampling is active (Gumbel adds its own root-level noise).
        if config.add_root_noise && config.gumbel_top_k.is_none() {
            let n = root.priors.len();
            if n > 0 {
                let noise = dirichlet_noise(n);
                for (prior, eta) in root.priors.iter_mut().zip(noise.iter()) {
                    *prior = (1.0 - noise_epsilon()) * *prior + noise_epsilon() * eta;
                }
            }
        }

        Self {
            root,
            config,
            min_max: MinMaxStats::new(),
        }
    }

    /// Run all simulations. Dispatches to either PUCT (default) or Gumbel-Top-k +
    /// sequential halving (when `config.gumbel_top_k.is_some()`).
    pub async fn run_simulations(&mut self, evaluator: &dyn Evaluator) {
        // Fresh normalization window each search.
        self.min_max = MinMaxStats::new();
        if self.config.gumbel_top_k.is_some() {
            self.run_simulations_gumbel(evaluator).await;
            return;
        }
        self.run_simulations_puct(evaluator).await;
    }

    /// Original PUCT-based simulation loop. Each simulation: select -> expand -> backpropagate.
    async fn run_simulations_puct(&mut self, evaluator: &dyn Evaluator) {
        // ---------------------------------------------------------------------------
        // Trace setup (zero cost when HYZERO_MCTS_TRACE is unset or "0").
        // ---------------------------------------------------------------------------
        let do_trace = trace_enabled() && {
            // First call: try to claim writer role.
            if !is_writer() && try_claim_writer() {
                WRITER_LOCAL.with(|w| *w.borrow_mut() = true);
                open_trace_file();
            }
            is_writer()
        };

        // Determine the current move number (fetch-and-increment atomically).
        // Only the writer task touches TRACE_MOVE_CTR, so no contention.
        let move_num = if do_trace {
            let m = TRACE_MOVE_CTR.fetch_add(1, Ordering::Relaxed);
            if m >= TRACE_MOVE_LIMIT {
                // Cap reached — stop writing for the rest of the game.
                WRITER_LOCAL.with(|w| *w.borrow_mut() = false);
                usize::MAX // sentinel; the block below is unreachable
            } else {
                m
            }
        } else {
            usize::MAX
        };

        let do_trace = do_trace && move_num < TRACE_MOVE_LIMIT;

        if do_trace {
            let legal = self.root.legal_actions.len();
            let c = self.config.exploration_constant;
            let sims = self.config.num_simulations;
            trace_write(&format!(
                "[mcts_trace] move={move_num} root_setup legal={legal} c={c:.4} sims={sims}"
            ));

            // Log top-5 root priors.
            let mut prior_idx: Vec<(usize, f32)> = self
                .root
                .priors
                .iter()
                .enumerate()
                .map(|(i, &p)| (i, p))
                .collect();
            prior_idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let top5: String = prior_idx
                .iter()
                .take(5)
                .map(|(i, p)| format!("{}:{:.4}", i, p))
                .collect::<Vec<_>>()
                .join(" ");
            trace_write(&format!(
                "[mcts_trace] move={move_num} root_priors_top5 {top5}"
            ));
        }

        for sim_i in 0..self.config.num_simulations {
            // Log per-sim PUCT scores at root before selecting.
            if do_trace {
                let parent_visits = self.root.visit_count;
                let c = self.config.exploration_constant;
                let scores: String = self
                    .root
                    .children
                    .iter()
                    .enumerate()
                    .map(|(i, child_opt)| {
                        let (q, cv) = match child_opt {
                            Some(ch) => {
                                let q = if ch.visit_count > 0 {
                                    ch.total_value / ch.visit_count as f32
                                } else {
                                    0.0
                                };
                                (q, ch.visit_count)
                            }
                            None => (0.0, 0),
                        };
                        let prior = self.root.priors[i];
                        let expl = c * prior * (parent_visits as f32).sqrt() / (1.0 + cv as f32);
                        let score = q + expl;
                        format!(
                            "{}:q={:.4},p={:.4},N={},expl={:.4},puct={:.4}",
                            i, q, prior, cv, expl, score
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
                trace_write(&format!(
                    "[mcts_trace] move={move_num} sim={sim_i} root_scores {scores}"
                ));
            }

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

                let child_idx = if qnorm_enabled() {
                    select_child_normalized(
                        node,
                        self.config.exploration_constant,
                        &self.min_max,
                        fpu_reduction(),
                    )
                } else {
                    select_child(node, self.config.exploration_constant)
                };
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

            if do_trace {
                let path_str: String = path
                    .iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
                    .join(",");
                trace_write(&format!(
                    "[mcts_trace] move={move_num} sim={sim_i} path={path_str} leaf_value={value:.4}"
                ));
            }

            // Backpropagate: walk back up the path, updating visit counts and values
            self.backpropagate(&path, value);
        }

        // Log final visit distribution at the root.
        if do_trace {
            let visit_str: String = self
                .root
                .children
                .iter()
                .enumerate()
                .map(|(i, c)| {
                    let v = c.as_ref().map_or(0, |n| n.visit_count);
                    format!("{}:{}", i, v)
                })
                .collect::<Vec<_>>()
                .join(" ");
            trace_write(&format!(
                "[mcts_trace] move={move_num} final_visit_dist {visit_str}"
            ));
            trace_flush();
        }
    }

    /// Gumbel-Top-K + sequential halving simulation loop.
    ///
    /// At the root: sample one Gumbel(0) value per legal action, take the top-K
    /// candidates by `logit + g`, then run sequential halving — each round
    /// simulates each surviving candidate, then halves the set by
    /// `logit + g + sigma(q)`. The visit distribution at the root reflects
    /// the halving allocation; later candidates that survived get more visits.
    ///
    /// Internal nodes (below root) use standard PUCT.
    async fn run_simulations_gumbel(&mut self, evaluator: &dyn Evaluator) {
        let n_legal = self.root.legal_actions.len();
        if n_legal == 0 {
            return;
        }

        let total_sims = self.config.num_simulations;
        let k_initial = self.config.gumbel_top_k.unwrap_or(16).min(n_legal).max(1);

        // logit(a) = ln(prior) — used as the "policy logit" surrogate. Gumbel-Top-K
        // sampling on (logit + g) is equivalent to drawing from softmax(logit) once.
        let logits: Vec<f32> = self.root.priors.iter().map(|&p| p.max(1e-9).ln()).collect();

        // Sample one Gumbel value per legal action (shared across all sims).
        let gumbel_noise = crate::mcts::gumbel::sample_gumbel(n_legal);

        // Top-K considered set by (g + logit).
        let mut scored: Vec<(usize, f32)> = (0..n_legal)
            .map(|i| (i, gumbel_noise[i] + logits[i]))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut considered: Vec<usize> = scored.iter().take(k_initial).map(|&(i, _)| i).collect();

        // Sequential halving: ceil(log2(K)) rounds. At each round, each surviving
        // candidate gets `total_sims / (num_rounds * |considered|)` sims (≥1).
        // Single-candidate case (K=1): just dump remaining budget into it below.
        let num_rounds = crate::mcts::gumbel::num_rounds(k_initial);
        let mut sims_done: u32 = 0;

        while considered.len() > 1 && sims_done < total_sims {
            let per_cand =
                crate::mcts::gumbel::sims_per_candidate(total_sims, num_rounds, considered.len());
            // Snapshot the current considered list to iterate; halving mutates
            // `considered` only after the round completes.
            let round_set = considered.clone();
            for &cand_idx in &round_set {
                for _ in 0..per_cand {
                    if sims_done >= total_sims {
                        break;
                    }
                    self.simulate_with_root_action(cand_idx, evaluator).await;
                    sims_done += 1;
                }
                if sims_done >= total_sims {
                    break;
                }
            }

            // Halve the considered set by (g + logit + sigma(q)).
            let max_visits = considered
                .iter()
                .map(|&i| {
                    self.root.children[i]
                        .as_ref()
                        .map(|c| c.visit_count)
                        .unwrap_or(0)
                })
                .max()
                .unwrap_or(0);

            let mut round_scored: Vec<(usize, f32)> = considered
                .iter()
                .map(|&i| {
                    let q = self.root.children[i]
                        .as_ref()
                        .map(|c| c.q_value())
                        .unwrap_or(0.0);
                    let s =
                        gumbel_noise[i] + logits[i] + crate::mcts::gumbel::sigma_q(q, max_visits);
                    (i, s)
                })
                .collect();
            round_scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let new_size = (considered.len() / 2).max(1);
            considered = round_scored
                .iter()
                .take(new_size)
                .map(|&(i, _)| i)
                .collect();
        }

        // Spend any remaining budget on the survivor (or the lone considered
        // candidate when K_initial=1).
        while sims_done < total_sims {
            let cand = considered.first().copied().unwrap_or(0);
            self.simulate_with_root_action(cand, evaluator).await;
            sims_done += 1;
        }
    }

    /// Run one MCTS simulation, forcing the first step at the root to take
    /// the given child index. Below the root, standard PUCT selection applies.
    async fn simulate_with_root_action(
        &mut self,
        root_action_idx: usize,
        evaluator: &dyn Evaluator,
    ) {
        let mut path: Vec<usize> = vec![root_action_idx];

        // Step 1: handle the forced root child. Either descend if expanded, or
        // expand-and-backprop right here.
        let root_child_was_some = self.root.children[root_action_idx].is_some();
        if !root_child_was_some {
            // Expand the root child via evaluator. This is the leaf for this sim.
            let action = self.root.legal_actions[root_action_idx];
            let (new_hidden, reward, policy, value) =
                evaluator.expand_leaf(&self.root.hidden_state, action).await;
            let child_actions = top_k_actions(&policy, 64);
            let child = MCTSNode::new(new_hidden, &policy, child_actions, reward);
            self.root.children[root_action_idx] = Some(Box::new(child));
            self.backpropagate(&path, value);
            return;
        }

        // Already-expanded root child: descend with PUCT until we hit an
        // unexpanded child or a terminal node.
        let mut current: *const MCTSNode = self.root.children[root_action_idx]
            .as_ref()
            .unwrap()
            .as_ref() as *const MCTSNode;

        let leaf_value: f32 = loop {
            let node = unsafe { &*current };
            if node.legal_actions.is_empty() {
                // Terminal node — backprop with current value
                break node.q_value();
            }

            let child_idx = if qnorm_enabled() {
                select_child_normalized(
                    node,
                    self.config.exploration_constant,
                    &self.min_max,
                    fpu_reduction(),
                )
            } else {
                select_child(node, self.config.exploration_constant)
            };
            path.push(child_idx);

            match &node.children[child_idx] {
                Some(child) => {
                    current = child.as_ref() as *const MCTSNode;
                }
                None => {
                    // Expand and break out
                    let leaf_action_idx = child_idx;
                    let parent = self.navigate_to_parent_mut(&path);
                    let action = parent.legal_actions[leaf_action_idx];
                    let (new_hidden, reward, policy, value) =
                        evaluator.expand_leaf(&parent.hidden_state, action).await;
                    let child_actions = top_k_actions(&policy, 64);
                    let child = MCTSNode::new(new_hidden, &policy, child_actions, reward);
                    parent.children[leaf_action_idx] = Some(Box::new(child));
                    break value;
                }
            }
        };

        self.backpropagate(&path, leaf_value);
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
    /// Canonical MuZero backup: walk leaf → root accumulating `G = r + γ·(−G_child)`
    /// with γ = 1 for 2-player zero-sum (chess). The reward on each edge contributes
    /// a sign-flipped term because the POV alternates per ply.
    ///
    /// Sign convention (matching PUCT in puct.rs, which uses `child.q_value` directly):
    /// - Each node stores Q from its PARENT's perspective.
    /// - The root has no parent; it stores Q from its own player's perspective.
    /// - `value` is the leaf evaluation from the LEAF's player's perspective.
    /// - `child.reward` is the reward received when transitioning INTO that child
    ///   (from the mover's POV at the parent of that child).
    ///
    /// For path of length D with leaf at depth D, leaf value `v`, and edge
    /// rewards `r_1..r_D` (where `r_k` is the reward on the edge entering the
    /// depth-k node), the return from each depth's POV is computed by the
    /// backward recurrence:
    ///     G_D     = v
    ///     G_{k-1} = r_k − G_k
    /// The stored-Q perspectives are:
    ///     stored_0      = G_0          (root, own POV)
    ///     stored_k ≥ 1  = G_{k-1}      (stored in parent's POV)
    ///
    /// When all edge rewards are zero this degenerates to the prior behavior:
    /// value walks up with a sign flip per ply, so existing zero-reward tests
    /// continue to pass bit-for-bit.
    fn backpropagate(&mut self, path: &[usize], value: f32) {
        let d = path.len();

        // Step 1: collect edge rewards along the path (r_1..r_D).
        let mut rewards: Vec<f32> = Vec::with_capacity(d);
        {
            let mut node: &MCTSNode = &self.root;
            for &idx in path {
                let child = node.children[idx].as_ref().unwrap();
                rewards.push(child.reward);
                node = child;
            }
        }

        // Step 2: compute G_k for k = 0..=D via reverse recurrence (γ = 1).
        // g_values[k] = G_k from depth-k's POV.
        let mut g_values: Vec<f32> = vec![0.0; d + 1];
        g_values[d] = value;
        for k in (0..d).rev() {
            // G_k = r_{k+1} − G_{k+1}; rewards[k] is r_{k+1} (0-indexed).
            g_values[k] = rewards[k] - g_values[k + 1];
        }

        // Step 3: update each node's stats with the correct POV-stored G.
        // Root stores G_0 (own POV); depth-k node (k ≥ 1) stores G_{k-1} (parent's POV).
        // After each update, fold the node's running Q into the MinMaxStats window
        // used for PUCT normalization (selection-only; stored totals are untouched).
        self.root.visit_count += 1;
        self.root.total_value += g_values[0];
        self.min_max.update(self.root.q_value());

        let mut node = &mut self.root;
        for (i, &idx) in path.iter().enumerate() {
            // Node at depth (i + 1) stores G_i (its parent's POV).
            let stored = g_values[i];
            let child = node.children[idx].as_mut().unwrap();
            child.visit_count += 1;
            child.total_value += stored;
            self.min_max.update(child.q_value());
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

    /// Snapshot of root statistics needed to reconstruct PUCT in a replay viewer.
    /// Each returned vector is parallel to `root.legal_actions`: position `i` describes
    /// the child for `legal_actions[i]`. Unexpanded children report visits=0, q=0.
    pub fn extract_root_diagnostics(&self) -> RootDiagnostics {
        let n = self.root.legal_actions.len();
        let mut child_visits = Vec::with_capacity(n);
        let mut q_values = Vec::with_capacity(n);
        for child_opt in &self.root.children {
            let (v, q) = match child_opt {
                Some(c) => (c.visit_count, c.q_value()),
                None => (0, 0.0),
            };
            child_visits.push(v);
            q_values.push(q);
        }
        RootDiagnostics {
            child_visits,
            priors: self.root.priors.clone(),
            q_values,
        }
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
            // Deterministic branch: find max visit count, then break ties uniformly
            // at random. `max_by` would pick the first-encountered max, creating a
            // lowest-`legal_actions`-index bias whenever there are tied-max visit
            // counts (common under uniform priors + value=0). This bias — combined
            // with the color-asymmetric `legal_actions` ordering from
            // `get_legal_moves` (which iterates absolute sq 0..63) — systematically
            // favored one color's move types in self-play.
            let max_visits = visits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let tied: Vec<usize> = visits
                .iter()
                .enumerate()
                .filter(|(_, &v)| (v - max_visits).abs() < f32::EPSILON)
                .map(|(i, _)| i)
                .collect();
            let best_idx = if tied.len() == 1 {
                tied[0]
            } else if tied.is_empty() {
                0
            } else {
                use rand::Rng;
                tied[rand::rng().random_range(0..tied.len())]
            };
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
            add_root_noise: true,
            gumbel_top_k: None,
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
            add_root_noise: true,
            gumbel_top_k: None,
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
            add_root_noise: true,
            gumbel_top_k: None,
        };

        let mut tree = MCTSTree::new(HiddenState::new(64), &policy, 0.5, legal_actions, config);

        tree.run_simulations(&MockEvaluator).await;

        // Temperature 0 should always pick the same action
        let action1 = tree.select_action(0.0);
        let action2 = tree.select_action(0.0);
        assert_eq!(action1, action2);
    }

    /// Gumbel mode: 200 sims with sharp prior should still distribute visits across
    /// the top-K considered set (sequential halving guarantees baseline sims).
    /// Under PUCT, ~all visits go to the top-prior move; under Gumbel, top-K
    /// candidates each get visits in round 1.
    #[tokio::test]
    async fn test_gumbel_distributes_visits() {
        // Sharp prior: action 0 gets 0.9, others share the rest.
        let mut policy = vec![0.0f32; NUM_ACTIONS];
        let n_legal = 16;
        let legal_actions: Vec<ActionIndex> = (0..n_legal as ActionIndex).collect();
        policy[0] = 0.9;
        let rest = 0.1 / (n_legal - 1) as f32;
        for i in 1..n_legal {
            policy[i] = rest;
        }

        let config = MCTSConfig {
            num_simulations: 200,
            exploration_constant: 1.5,
            add_root_noise: false,
            gumbel_top_k: Some(16),
        };

        let mut tree = MCTSTree::new(HiddenState::new(64), &policy, 0.0, legal_actions, config);
        tree.run_simulations(&MockEvaluator).await;

        let visits: Vec<u32> = tree
            .root
            .children
            .iter()
            .map(|c| c.as_ref().map_or(0, |n| n.visit_count))
            .collect();

        // Count children that received any visits at all.
        let visited = visits.iter().filter(|&&v| v > 0).count();
        // With K=16 and 200 sims, ALL 16 candidates should get at least 1 visit
        // in round 1 (round 1 budget alone = 200/4 = 50 sims for 16 candidates).
        assert!(
            visited >= 8,
            "Gumbel should visit many candidates with sharp prior; only {} of {} had visits",
            visited,
            n_legal,
        );

        // Top-prior move should NOT have all the visits — under PUCT it would.
        let top_visits = visits[0];
        let total_visits: u32 = visits.iter().sum();
        let frac = top_visits as f32 / total_visits as f32;
        assert!(
            frac < 0.7,
            "Gumbel root visits over-concentrated on top prior: {}/{} = {:.2}",
            top_visits,
            total_visits,
            frac,
        );
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
            add_root_noise: true,
            gumbel_top_k: None,
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

    /// Statistical MCTS ordering invariance regression.
    ///
    /// Runs 100 independent MCTS searches with UNSORTED legal_actions and 100 with
    /// SORTED legal_actions (both starting from the same set of action IDs at the root).
    /// Computes mean visit distributions mapped by ACTION ID. The mean distributions
    /// should agree within tolerance=0.05 per action, confirming that the visit
    /// distribution is not sensitive to the presentation order of legal_actions.
    ///
    /// This is a statistical test; a single MCTS tree with Dirichlet noise is
    /// non-deterministic so individual runs will differ, but means over 100 runs
    /// should converge. A failure here indicates a structural ordering bias
    /// (e.g., argmax tie-break always picking lowest-index child regardless of action ID).
    #[tokio::test]
    #[ignore = "statistical ordering invariance — 200 MCTS runs; run with --ignored"]
    async fn test_mcts_visit_distribution_ordering_invariance() {
        // Use a small set of action IDs that are NOT already in sorted order.
        // Representing moves from the starting position (e2e4, d2d4, c2c4, g1f3, b1c3).
        let e2e4: ActionIndex = 12 * 64 + 28; // 796
        let d2d4: ActionIndex = 11 * 64 + 27; // 731
        let c2c4: ActionIndex = 10 * 64 + 26; // 666
        let g1f3: ActionIndex = 6 * 64 + 21; // 405
        let b1c3: ActionIndex = 1 * 64 + 18; // 82

        // Unsorted order: descending by action ID (796, 731, 666, 405, 82)
        let unsorted: Vec<ActionIndex> = vec![e2e4, d2d4, c2c4, g1f3, b1c3];
        // Sorted order: ascending (82, 405, 666, 731, 796)
        let mut sorted = unsorted.clone();
        sorted.sort_unstable();

        let n_actions = unsorted.len();
        let num_sims = 200u32;
        let num_trials = 100usize;

        // action-ID sorted reference order (same for both orderings — same set of IDs)
        let sorted_ids: Vec<ActionIndex> = sorted.clone();

        // Run `num_trials` MCTS searches with legal_actions in `legal` order.
        // Returns mean visit distribution indexed by `sorted_ids` position.
        macro_rules! run_trials {
            ($legal:expr) => {{
                let legal: &Vec<ActionIndex> = &$legal;
                let mut acc: Vec<f64> = vec![0.0; n_actions];
                for _ in 0..num_trials {
                    let policy = vec![1.0 / NUM_ACTIONS as f32; NUM_ACTIONS];
                    let config = MCTSConfig {
                        num_simulations: num_sims,
                        exploration_constant: 1.5,
                        add_root_noise: true,
                        gumbel_top_k: None,
                    };
                    let mut tree =
                        MCTSTree::new(HiddenState::new(64), &policy, 0.0, legal.clone(), config);
                    tree.run_simulations(&MockEvaluator).await;
                    let dist = tree.extract_visit_distribution();
                    // Map child-slot → sorted_ids position → accumulator
                    for (slot, &action) in legal.iter().enumerate() {
                        if let Some(pos) = sorted_ids.iter().position(|&a| a == action) {
                            acc[pos] += dist[slot] as f64;
                        }
                    }
                }
                acc.iter()
                    .map(|&s| (s / num_trials as f64) as f32)
                    .collect::<Vec<f32>>()
            }};
        }

        let mean_unsorted = run_trials!(unsorted);
        let mean_sorted = run_trials!(sorted);

        let tolerance = 0.05f32;
        let mut failures: Vec<(ActionIndex, f32, f32, f32)> = Vec::new();
        let sorted_ids: Vec<ActionIndex> = {
            let mut s = unsorted.clone();
            s.sort_unstable();
            s
        };
        for (i, (&action, (&mu_u, &mu_s))) in sorted_ids
            .iter()
            .zip(mean_unsorted.iter().zip(mean_sorted.iter()))
            .enumerate()
        {
            let _ = i;
            let delta = (mu_u - mu_s).abs();
            if delta > tolerance {
                failures.push((action, mu_u, mu_s, delta));
            }
        }

        assert!(
            failures.is_empty(),
            "MCTS visit distribution differs by >={} for {} action(s) between \
             unsorted and sorted legal_actions (tolerance={}).\n\
             Failures (action, mean_unsorted, mean_sorted, |delta|): {:?}\n\
             Full unsorted means: {:?}\n\
             Full sorted means:   {:?}",
            tolerance,
            failures.len(),
            tolerance,
            failures,
            mean_unsorted,
            mean_sorted,
        );
    }

    /// Q-normalization is selection-only: folding each node's running Q into the
    /// MinMaxStats window during backprop must NOT alter the stored `total_value`
    /// or `visit_count`. This guards the sign-convention regressions
    /// (`test_backpropagate_alternates_signs`, `test_backpropagate_includes_mating_reward`)
    /// against normalization leaking into the backup math.
    ///
    /// The reverse recurrence `G_{k-1}=r_k−G_k` (γ=1) is reward-driven and
    /// independent of `min_max`; here we drive a set of paths with known rewards
    /// and assert every stored total equals the recurrence value exactly, while
    /// `min_max` has nonetheless absorbed those Q-values (window non-degenerate).
    #[tokio::test]
    async fn qnorm_does_not_change_stored_total_value() {
        let uniform_policy = vec![1.0 / NUM_ACTIONS as f32; NUM_ACTIONS];
        let nil_config = MCTSConfig {
            num_simulations: 0,
            exploration_constant: 1.5,
            add_root_noise: false,
            gumbel_top_k: None,
        };

        let legal: Vec<ActionIndex> = (0..2).collect();
        let mut tree = MCTSTree::new(
            HiddenState::new(64),
            &uniform_policy,
            0.0,
            legal,
            nil_config,
        );

        // Build a D=2 path with a mating reward on the leaf edge.
        let legal_d1: Vec<ActionIndex> = (0..2).collect();
        let d1 = MCTSNode::new(HiddenState::new(64), &uniform_policy, legal_d1, 0.0);
        tree.root.children[0] = Some(Box::new(d1));
        let legal_d2: Vec<ActionIndex> = (0..2).collect();
        let d2 = MCTSNode::new(HiddenState::new(64), &uniform_policy, legal_d2, 1.0);
        tree.root.children[0].as_mut().unwrap().children[0] = Some(Box::new(d2));

        let root_tv_before = tree.root.total_value;
        tree.backpropagate(&[0, 0], 0.0);

        // Stored totals follow the recurrence (γ=1, leaf value 0, edge reward +1):
        //   G_2=0, G_1=+1, G_0=-1 → root+=-1, depth-1+=-1, depth-2+=+1.
        assert!((tree.root.total_value - root_tv_before - (-1.0)).abs() < 1e-6);
        let stored_d1 = tree.root.children[0].as_ref().unwrap().total_value;
        let stored_d2 = tree.root.children[0].as_ref().unwrap().children[0]
            .as_ref()
            .unwrap()
            .total_value;
        assert!(
            (stored_d1 - (-1.0)).abs() < 1e-6,
            "depth-1 total altered by qnorm"
        );
        assert!(
            (stored_d2 - 1.0).abs() < 1e-6,
            "depth-2 total altered by qnorm"
        );

        // The window DID absorb the Q-values (so normalization would be active),
        // proving the updates ran without touching stored totals.
        assert!(
            !tree.min_max.is_degenerate(),
            "min_max should have a non-degenerate window after backprop"
        );
    }

    #[tokio::test]
    async fn test_backpropagate_alternates_signs() {
        // Test backpropagate directly on hand-built trees for path lengths 1, 2, 3.
        //
        // Sign convention (matches puct.rs which reads child.q_value directly):
        //   - Leaf value is from the LEAF player's perspective.
        //   - Root stores Q from root's own player's POV.
        //   - Non-root nodes store Q from their PARENT's POV.
        //   - In a two-player alternating game, perspective flips per ply.
        //
        // Expected accumulated total_value contributions from a single backprop of +1.0:
        //   D=1 path [a]:         root=-1, depth-1[a]=-1
        //   D=2 path [a, b]:      root=+1, depth-1[a]=+1, depth-2[a,b]=-1
        //   D=3 path [a, b, c]:   root=-1, depth-1[a]=-1, depth-2[a,b]=+1, depth-3[a,b,c]=-1

        let uniform_policy = vec![1.0 / NUM_ACTIONS as f32; NUM_ACTIONS];
        let nil_config = MCTSConfig {
            num_simulations: 0,
            exploration_constant: 1.5,
            add_root_noise: false,
            gumbel_top_k: None,
        };

        // Helper: install a freshly-created child at path[0..path.len()] below root.
        // Assumes all intermediate children already exist.
        fn install_child(
            tree: &mut MCTSTree,
            parent_path: &[usize],
            slot: usize,
            policy: &[f32],
            num_grandchild_slots: usize,
        ) {
            let legal: Vec<ActionIndex> = (0..num_grandchild_slots as ActionIndex).collect();
            let child = MCTSNode::new(HiddenState::new(64), policy, legal, 0.0);

            // Walk down to the parent.
            let mut node: &mut MCTSNode = &mut tree.root;
            for &idx in parent_path {
                node = node.children[idx].as_mut().unwrap();
            }
            node.children[slot] = Some(Box::new(child));
        }

        // ---------- D=1 ----------
        {
            let legal: Vec<ActionIndex> = (0..2).collect();
            let mut tree = MCTSTree::new(
                HiddenState::new(64),
                &uniform_policy,
                0.0, // root_value = 0 so root.total_value starts at 0
                legal,
                nil_config.clone(),
            );
            // root.total_value is 0.0 from MCTSTree::new (root_value=0, visit_count=1).
            // Install a depth-1 child in slot 0 so backpropagate can unwrap it.
            install_child(&mut tree, &[], 0, &uniform_policy, 2);
            let root_tv_before = tree.root.total_value;

            tree.backpropagate(&[0], 1.0);

            assert!(
                (tree.root.total_value - root_tv_before - (-1.0)).abs() < 1e-6,
                "D=1 root delta should be -1.0, got {}",
                tree.root.total_value - root_tv_before,
            );
            let d1 = tree.root.children[0].as_ref().unwrap();
            assert!(
                (d1.total_value - (-1.0)).abs() < 1e-6,
                "D=1 depth-1 should be -1.0, got {}",
                d1.total_value,
            );
        }

        // ---------- D=2 ----------
        {
            let legal: Vec<ActionIndex> = (0..2).collect();
            let mut tree = MCTSTree::new(
                HiddenState::new(64),
                &uniform_policy,
                0.0,
                legal,
                nil_config.clone(),
            );
            install_child(&mut tree, &[], 0, &uniform_policy, 2);
            install_child(&mut tree, &[0], 0, &uniform_policy, 2);
            let root_tv_before = tree.root.total_value;

            tree.backpropagate(&[0, 0], 1.0);

            assert!(
                (tree.root.total_value - root_tv_before - 1.0).abs() < 1e-6,
                "D=2 root delta should be +1.0, got {}",
                tree.root.total_value - root_tv_before,
            );
            let d1 = tree.root.children[0].as_ref().unwrap();
            assert!(
                (d1.total_value - 1.0).abs() < 1e-6,
                "D=2 depth-1 should be +1.0, got {}",
                d1.total_value,
            );
            let d2 = d1.children[0].as_ref().unwrap();
            assert!(
                (d2.total_value - (-1.0)).abs() < 1e-6,
                "D=2 depth-2 should be -1.0, got {}",
                d2.total_value,
            );
        }

        // ---------- D=3 ----------
        {
            let legal: Vec<ActionIndex> = (0..2).collect();
            let mut tree = MCTSTree::new(
                HiddenState::new(64),
                &uniform_policy,
                0.0,
                legal,
                nil_config.clone(),
            );
            install_child(&mut tree, &[], 0, &uniform_policy, 2);
            install_child(&mut tree, &[0], 0, &uniform_policy, 2);
            install_child(&mut tree, &[0, 0], 0, &uniform_policy, 2);
            let root_tv_before = tree.root.total_value;

            tree.backpropagate(&[0, 0, 0], 1.0);

            assert!(
                (tree.root.total_value - root_tv_before - (-1.0)).abs() < 1e-6,
                "D=3 root delta should be -1.0, got {}",
                tree.root.total_value - root_tv_before,
            );
            let d1 = tree.root.children[0].as_ref().unwrap();
            assert!(
                (d1.total_value - (-1.0)).abs() < 1e-6,
                "D=3 depth-1 should be -1.0, got {}",
                d1.total_value,
            );
            let d2 = d1.children[0].as_ref().unwrap();
            assert!(
                (d2.total_value - 1.0).abs() < 1e-6,
                "D=3 depth-2 should be +1.0, got {}",
                d2.total_value,
            );
            let d3 = d2.children[0].as_ref().unwrap();
            assert!(
                (d3.total_value - (-1.0)).abs() < 1e-6,
                "D=3 depth-3 should be -1.0, got {}",
                d3.total_value,
            );
        }
    }

    #[tokio::test]
    async fn test_backpropagate_includes_mating_reward() {
        // Canonical MuZero backup: a mating reward on an edge must propagate
        // into the root's Q value (sign-flipped per ply back to root POV).
        //
        // Setup: D=2 path root → depth-1 → depth-2 (leaf). The transition from
        // depth-1 into depth-2 is a mating move, so that edge carries reward
        // r_2 = +1 from depth-1's POV (depth-1 just mated its opponent). The
        // leaf is the absorbing post-mate state with v_leaf = 0.
        //
        // Canonical recurrence (γ=1):
        //   G_2 = 0            (leaf/absorbing POV)
        //   G_1 = r_2 - G_2 = +1   (depth-1's POV: depth-1 won)
        //   G_0 = r_1 - G_1 = -1   (root's POV: root's opponent won → root loses)
        //
        // Stored values:
        //   root         += G_0 = -1
        //   depth-1      += G_0 = -1   (stored in parent=root's POV)
        //   depth-2 leaf += G_1 = +1   (stored in parent=depth-1's POV)
        //
        // Without the reward term (old backup), all contributions would be 0 and
        // the mating signal would be invisible to MCTS selection.

        let uniform_policy = vec![1.0 / NUM_ACTIONS as f32; NUM_ACTIONS];
        let nil_config = MCTSConfig {
            num_simulations: 0,
            exploration_constant: 1.5,
            add_root_noise: false,
            gumbel_top_k: None,
        };

        let legal: Vec<ActionIndex> = (0..2).collect();
        let mut tree = MCTSTree::new(
            HiddenState::new(64),
            &uniform_policy,
            0.0,
            legal,
            nil_config,
        );

        // Install depth-1 child with zero reward on root → depth-1 edge.
        let legal_d1: Vec<ActionIndex> = (0..2).collect();
        let d1_node = MCTSNode::new(HiddenState::new(64), &uniform_policy, legal_d1, 0.0);
        tree.root.children[0] = Some(Box::new(d1_node));

        // Install depth-2 leaf with reward = +1 on the depth-1 → depth-2 edge
        // (the mating transition).
        let legal_d2: Vec<ActionIndex> = (0..2).collect();
        let d2_node = MCTSNode::new(HiddenState::new(64), &uniform_policy, legal_d2, 1.0);
        tree.root.children[0].as_mut().unwrap().children[0] = Some(Box::new(d2_node));

        let root_tv_before = tree.root.total_value;

        // Backpropagate with leaf value = 0 (absorbing post-mate state).
        tree.backpropagate(&[0, 0], 0.0);

        // Root delta must be -1 (root lost because its opponent delivered mate).
        let root_delta = tree.root.total_value - root_tv_before;
        assert!(
            (root_delta - (-1.0)).abs() < 1e-6,
            "root delta should be -1.0 (opponent mated root); got {}",
            root_delta,
        );

        // Depth-1 stores Q from root's POV → -1 (depth-1 is root's opponent
        // who just won, which is -1 from root's POV).
        let d1 = tree.root.children[0].as_ref().unwrap();
        assert!(
            (d1.total_value - (-1.0)).abs() < 1e-6,
            "depth-1 should be -1.0 (stored in root's POV); got {}",
            d1.total_value,
        );

        // Depth-2 leaf stores Q from depth-1's POV → +1 (depth-1 mated its
        // opponent; from depth-1's POV the return is +1).
        let d2 = d1.children[0].as_ref().unwrap();
        assert!(
            (d2.total_value - 1.0).abs() < 1e-6,
            "depth-2 leaf should be +1.0 (stored in depth-1's POV); got {}",
            d2.total_value,
        );

        // Sanity: q_value at depth-1 reflects the mate. PUCT at root will see
        // this -1 and push search toward the OTHER child (not losing to mate).
        let d1_q = d1.q_value();
        assert!(d1_q < 0.0, "d1.q_value should be negative; got {}", d1_q);
    }
}
