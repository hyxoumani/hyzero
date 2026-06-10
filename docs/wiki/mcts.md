# MCTS (Monte Carlo Tree Search)

A fresh transient tree is built for every move, searched for `num_simulations`,
then the visit distribution is extracted and the tree discarded. Implementation
in `src/mcts/{tree,node,puct,gumbel,evaluator}.rs`.

## Evaluator Interface

`src/mcts/evaluator.rs` defines the abstraction the tree calls into:

```rust
trait Evaluator {
    async fn root_setup(&self, obs: &BoardObservation, legal_mask: &[bool])
        -> (HiddenState, Policy, f32);                       // h() + f()
    async fn expand_leaf(&self, hs: &HiddenState, action: ActionIndex)
        -> (HiddenState, f32, Policy, f32);                  // g() + f()
}
```

Implementations: `ChannelEvaluator` (routes to the Python inference batcher) and
`RandomEvaluator` (uniform policy, zero value — used in tests and as the
ladder's v0 baseline).

## Per-move Flow

1. Encode board → `BoardObservation` (102 planes; see [Board Encoding](board-encoding.md)).
2. `evaluator.root_setup(obs, legal_mask)` → `(hidden, policy, value)`.
3. `MCTSTree::new(...)` seeds the root (1 visit, root_value), then optionally mixes Dirichlet noise.
4. `run_simulations(evaluator)` runs `num_simulations` simulations.
5. `select_action(temperature)` picks the move from visit counts.
6. Apply the move, record a `StepRecord`, discard the tree.

## A Single PUCT Simulation (`run_simulations_puct`)

1. **SELECT**: walk down with `select_child` (PUCT) until reaching an unexpanded child or a terminal (no legal actions).
2. **EXPAND**: `expand_leaf(hidden, action)` → next hidden state, reward, child policy, value.
3. **EVALUATE**: the leaf value initializes the new child.
4. **BACKUP**: `backpropagate(path, value)` propagates the value to the root.

**PUCT score** (`src/mcts/puct.rs::puct_score`):
```
score(a) = Q(s,a) + c · P(s,a) · sqrt(N_parent) / (1 + N(a))
```
`Q = total_value / visit_count` (0 if unvisited), `c` = `exploration_constant`
(default 1.5). `select_child` collects all children within `TIE_EPSILON = 1e-6`
of the best score and **breaks ties uniformly at random** (see "Selection
Mechanics").

**Backup is reward-aware** (`backpropagate`): for a path of depth D with leaf
value `v` and edge rewards `r_1..r_D`, it computes `G_D = v`, `G_{k-1} = r_k − G_k`
(γ = 1) and stores each node's return in its parent's POV. With zero rewards this
degenerates to the classic two-player negation (`value` flips sign per ply), so
zero-reward tests pass bit-for-bit.

## Root Noise (Dirichlet)

When `add_root_noise` is true and Gumbel is off, root priors are mixed:
`P(a) = (1 − ε)·P(a) + ε·η_a`, with `η ~ Dir(α)` sampled via Marsaglia-Tsang
Gamma. Defaults `ε = 0.25` (`HYZERO_DIRICHLET_EPS`; `scripts/run_baseline.sh`
exports 0.10), `α = 0.3` (`HYZERO_DIRICHLET_ALPHA`, the AlphaZero chess value).
`add_root_noise` is true
for self-play, false for evaluation. Dirichlet sampling is slow in debug builds —
use `--release`.

## Gumbel-Top-k + Sequential Halving

When `MCTSConfig.gumbel_top_k = Some(k)`, root selection switches from PUCT-with-
Dirichlet to Gumbel-Top-k with sequential halving (`run_simulations_gumbel`,
helper in `src/mcts/gumbel.rs`). `k` is capped to the legal-action count. Gumbel
sampling provides its own root noise, so Dirichlet is auto-disabled in this mode.
Internal nodes still use PUCT.

## Action Selection (`select_action`)

`MCTSConfig` default: `num_simulations = 800`, `exploration_constant = 1.5`,
`add_root_noise = true`, `gumbel_top_k = None`.

- **temperature ≤ EPSILON** (greedy): find the max visit count, collect all
  tied-max indices, pick one **uniformly at random**. A plain first-max argmax
  would bias toward `legal_actions[0]`.
- **temperature > 0**: sample proportional to `visit_count^(1/temperature)`.

## Selection Mechanics — Color-Symmetry Caveats

Two interacting issues at the action-selection boundary previously produced a
systemic color bias (~83% Black dominance) despite symmetric rules:

1. **Argmax tie-break bias.** Under uniform priors and value≈0 early in training,
   visit ties are common. Picking the first-encountered max biases toward index 0.
   *Fix:* random tie-break in both `select_action` (root) and `select_child`
   (internal PUCT), implemented today.
2. **Color-asymmetric `legal_actions` ordering.** `get_legal_moves()` iterates
   absolute squares 0..63, so White's index-0 action is a knight move while
   Black's is a pawn move. Combined with bias #1 this systematically favored one
   color's move types. *Fix:* callers must `legal_actions.sort_unstable()` after
   POV-flipping so both colors present identical sorted lists. Verified by
   `test_legal_actions_ordering_is_color_symmetric_after_sort`.

**Lesson**: current-player-perspective encoding makes POV invariance fragile —
every consumer of `legal_actions` must be POV-aware (symmetric indices, random
tie-breaks, aligned visit distributions).

## Replay Capture (diagnostic)

`extract_root_diagnostics()` snapshots per-child `child_visits`, `priors`,
`q_values` (parallel to `root.legal_actions`) for the opt-in `.replay` capture —
see [Replay Subsystem](replay-subsystem.md). Distinct from the training replay
buffer.

## Gotchas

1. **Transient tree**: discarded after each move; no caching between moves.
2. **Backup negation**: value flips sign per ply (zero-reward case) — the general
   recurrence `G_{k-1} = r_k − G_k` subsumes it.
3. **Legal-mask NaN**: masked illegal logits become `-inf`; `nan_to_num` keeps
   `0·(−inf)` from producing NaN.
4. **Dirichlet cost**: very slow in debug — always `--release` for e2e.

## Related

- [Self-Play Coordinator](selfplay-coordinator.md) — the game loop that drives search
- [Neural Networks](neural-networks.md) — the evaluator's h/g/f networks
- [Board Encoding](board-encoding.md) — action flipping, legal-action ordering
- `src/mcts/tree.rs` — `MCTSTree`, `run_simulations_*`, `backpropagate`, `select_action`
- `src/mcts/puct.rs` — `puct_score`, `select_child`
- `src/mcts/gumbel.rs` — Gumbel-Top-k helper
