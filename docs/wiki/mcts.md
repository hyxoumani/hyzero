# MCTS (Monte Carlo Tree Search)

A fresh transient tree is built for every move, searched for `num_simulations`, then
the visit distribution is extracted and the tree discarded. Below the root the tree is
**terminal-blind** — children are latent hidden states seeded from `top_k(64)` priors,
so real-board terminals are invisible except at depth 1, where the driver injects ground
truth. Implementation in `src/mcts/{tree,node,puct,gumbel,evaluator}.rs`; grounding in `src/selfplay/game_task.rs`.

## Evaluator Interface

`src/mcts/evaluator.rs` defines the abstraction the tree calls into:

```rust
trait Evaluator {
    async fn root_setup(&self, obs, legal_mask) -> (HiddenState, Policy, f32); // h()+f()
    async fn expand_leaf(&self, hs, action)   -> (HiddenState, f32, Policy, f32); // g()+f()
}
```

Implementations: `ChannelEvaluator` (routes to the Python inference batcher) and
`RandomEvaluator` (uniform policy, zero value — tests + ladder v0 baseline).

## Per-move Flow

1. Encode board → `BoardObservation` (see [Board Encoding](board-encoding.md)).
2. `root_setup` → `(hidden, policy, value)`; `MCTSTree::new` seeds the root.
3. Driver computes `root_child_terminals(...)` → `set_root_terminals(...)`.
4. `run_simulations` runs `num_simulations`; `select_action(temperature)` picks.
5. Apply the move, record a `StepRecord`, discard the tree.

## PUCT Simulation

SELECT (walk down via `select_child`) → EXPAND (`expand_leaf`) → EVALUATE (leaf
value) → BACKUP. **Score:** `Q + c·P·sqrt(N_parent)/(1+N)`, `c=1.5`. With
`HYZERO_MCTS_QNORM` on (default) selection uses `select_child_normalized`: MinMax-
normalized Q plus FPU reduction (`DEFAULT_FPU_REDUCTION=0.25`) for unvisited children.
**Backup is reward-aware:** `G_{k-1} = r_k − G_k` (γ=1); zero rewards degenerate to classic per-ply negation (**exactly one sign flip per ply**).

## Root-child Terminal Grounding (2026-07-08)

`root_child_terminals` applies each depth-1 legal move to the **real** cloned board:
mate → `-1.0`, rule-draw → `0.0` (both leaf/resulting-STM POV), Ongoing/Err →
`None`. `set_root_terminals` installs the vector (parallel to `legal_actions`). On
expansion a grounded child becomes a terminal node whose TRUE value is backed up
**verbatim on every visit** — never replaced by the network estimate, never
expanded below. Fixes defender exploitation and root-mate detection. Black actions
are `flip_action`-mapped to absolute coords first.

## Forced-line Quiescence Extension (2026-07-12)

Env-gated (`HYZERO_FORCED_EXTENSION=1`, off by default; `HYZERO_FORCED_EXT_DEPTH`
default 8). When on, an Ongoing depth-1 child walks a **single-legal-move** forced
chain on a cloned board (`forced_line_value`): follows the lone reply until a
terminal, a real choice (>1 legal move → `None`), or the depth cap (→ `None`,
network fallback). Sound without minimax because a one-move position has a determined
continuation. Mate parity: `(-1)^(n+1)` at the start (leaf) POV; draws `0.0` at any
parity. Gate resolved once per `root_child_terminals` call, not per child; off-path
is bit-identical to immediate-terminal-only grounding.

## MLH Search Bonus

`MlhBonus` (lc0 moves-left bonus) parsed from env once via
`MlhBonus::from_env_cached()` (`OnceLock`, hot-path safe). Default OFF
(`HYZERO_MLH_SEARCH_BONUS=0.0` → `is_off`); as a conversion lever it was validly
**exhausted** — see [[conversion-levers]].

## Root Noise + Gumbel

Dirichlet (`add_root_noise` && no Gumbel): `P=(1−ε)P+ε·η`, `η~Dir(α)`, ε=0.25
(`HYZERO_DIRICHLET_EPS`; baseline exports 0.10), α=0.3 (`HYZERO_DIRICHLET_ALPHA`);
on for self-play, off for eval. When `gumbel_top_k=Some(k)`, root uses Gumbel-Top-k
+ sequential halving (`gumbel.rs`) supplying its own root noise so Dirichlet auto-
disables; internal nodes stay PUCT. All env knobs `OnceLock`-cached.

## Action Selection

Default `num_simulations=800`. **temp ≤ EPSILON (greedy):** collect all tied-max
visit indices, pick uniformly at random (plain argmax biases `legal_actions[0]`).
**temp > 0:** sample ∝ `visit_count^(1/temp)`. Callers must `sort_unstable()` the
POV-flipped `legal_actions` so both colors present identical sorted lists (fixes ~83%
color bias; POV-encoding makes invariance fragile — every consumer must be POV-aware).

## Gotchas

1. **Terminal-blind below root** by construction: latent `top_k(64)` children are never
   empty and true terminals invisible — the architectural constraint behind the three-campaign conversion conclusion. See [[conversion-levers]].
2. **Search scaling can INVERT** with a miscalibrated value head: 400 sims scored
   0 where 100 scored 4% — deeper search amplifies confident-wrong values.
3. **Leaf-POV parity is load-bearing:** grounding values stored leaf-POV, backprop applies
   exactly ONE sign flip per ply. New grounding sources must match — parity bugs are silent.
4. **Transient tree:** discarded each move; no cross-move caching.
5. **Legal-mask NaN:** masked logits → `-inf`; `nan_to_num` avoids `0·(−inf)`.

## Related

- [[conversion-levers]] — why grounding/MLH did not solve conversion
- [[selfplay-coordinator]] — the game loop + driver that grounds the root
- [Neural Networks](neural-networks.md) — the evaluator's h/g/f networks
- `src/mcts/tree.rs`, `src/mcts/puct.rs`, `src/selfplay/game_task.rs`
