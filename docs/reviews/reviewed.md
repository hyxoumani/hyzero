# Bug Review Log

Automated bug-focused reviews run on a schedule. Each entry records the review
range and the findings so subsequent runs skip already-reviewed commits.

## 2026-08-09 — reviewed through `bde4f9b`

Baseline pass. Scope was scoped to the two most substantive recent code
changes; older commits are treated as reviewed by acknowledgement, not
verification, and will not be re-scanned on future runs.

**Reviewed:**

- ELO-promotion feature — commits `7b53e5d..bde4f9b`
- MCTS Gumbel-Top-K + sequential halving — commit `5f30ea8`

### ELO-promotion feature — CLEAN

No bugs. Cross-checks passed on: Elo formula (`E = 1/(1+10^((Ropp-Ra)/400))`,
`R' = R + K*(S-E)`), bootstrap vs. pool promotion gate direction, symmetric
score mapping across colors, pool-exclusion keying vs. archive naming, env-var
names matching startup notices, Mutex not held across `.await`, empty-pool /
divide-by-zero guards, awk `candidate_elo` extractor fallback in
`scripts/run_baseline.sh`.

### MCTS Gumbel-Top-K — 2 medium, 4 low, 1 info

- **MED** `src/mcts/gumbel.rs:42` — `sigma_q` uses raw `q` values in `[-1, 1]`
  without the paper's min-max normalization to `[0, 1]`. With `c_visit=50`,
  `c_scale=1.0`, |sigma| reaches ~53 vs. Gumbel std ~1.28, so sigma dominates
  `gumbel + logit` and halving degenerates to q-argmax rather than the
  intended Gumbel-perturbed selection.
- **MED** `src/mcts/tree.rs:586` — halving uses `child.q_value()` from the
  parent's POV as the "completed Q". Unvisited or zero-total-value children
  silently score `q = 0` via `unwrap_or(0.0)`. The paper requires an explicit
  V-completion estimate for unvisited actions. Matters whenever
  `total_sims < num_rounds * k_initial`, i.e. some round-1 candidates never
  get a visit before halving.
- **LOW** `src/mcts/tree.rs:596` — `new_size = considered.len() / 2` uses
  floor division. For non-power-of-2 `K` (3→1, 5→2, 6→3→1, 7→3→1) the
  schedule collapses a level so the actual halving-round count is less than
  `num_rounds`, wasting the middle-round sim budgets on the survivor.
  Harmless at default `K=16`; bites if `HYZERO_GUMBEL_TOP_K` is odd or a
  non-power-of-2.
- **LOW** `src/mcts/tree.rs:544` — inner `for &cand_idx in &round_set` breaks
  early on `sims_done >= total_sims` but still enters the halving block, so
  an aborted round halves candidates using stale/zero `q` for the unvisited
  tail. Benign for `K=16` / 200 sims; wrong when `total_sims` is small.
- **LOW** `src/mcts/gumbel.rs:24` / `src/mcts/tree.rs:523` — `sample_gumbel`
  uses `rand::rng()` (thread-local, unseeded), so search is non-reproducible.
  No specific-ordering test, so no CI flake, but repeatable eval / self-play
  is not possible without a seed hook.
- **LOW** `src/mcts/tree.rs:991-1005` — the new `test_gumbel_distributes_visits`
  asserts only `visited >= 8` and `top_visits / total < 0.7`. With `K=16` and
  200 sims, round-1 alone visits all 16 and the survivor gets ~52/200 ≈ 0.26,
  so both bounds are trivially met. The test verifies "doesn't panic +
  round-1 ran", not sequential-halving convergence or the paper's argmax
  property.
- **LOW** `src/mcts/tree.rs:511` — `k_initial = gumbel_top_k.unwrap_or(16)
.min(n_legal).max(1)` also silently clamps a caller-supplied
  `MCTSConfig.gumbel_top_k = Some(0)` to 1 with no validation or warning.
  Future footgun.
- **INFO** `src/mcts/gumbel.rs:88` — `improved_policy` is dead-code with
  `#[allow(dead_code)]`. Commit message confirms this is intentional
  (training target is visit counts, not the paper's improved policy), so
  training targets diverge from Gumbel-MuZero theory even when the search
  path is enabled.
