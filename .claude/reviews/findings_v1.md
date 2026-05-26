# Review Findings — v1 (commit ee132c4)

Bugs and concerns from the TB-supervision squash review, ordered by severity.

---

## High severity

### 1. Black-underpromotion action plane misaligned with observation

**File**: `src/data/encoding.rs:292-331` (and its Python port
`python/hyzero/data/board_encoder.py:187-218`).

For underpromotion actions, `encode_action_spatial_for_color(action, white=false)`
places the from/to squares at ranks **1→0** (raw board), while the **observation
is in POV coords**. For Black to move, the obs encoder rank-mirrors all squares
(`flip_sq` in `encode_board`), so a Black pawn at raw rank 1 (about to promote)
appears at **obs rank 6**, and its promotion target (raw rank 0) appears at **obs
rank 7**. The action plane points at the wrong end of the board — by ~6 ranks.

For comparison, **base** actions are POV-aligned because `flip_action()` rewrites
the from/to squares in the index itself (via `flip_base_action`). But
`flip_action` is the identity on underpromos (line 357-363), and the encoder
then uses raw promo ranks for Black, breaking POV alignment.

Why the existing tests don't catch this: `test_flip_action_planes_matches_flip_action_invariant`
verifies a _self-consistency_ property (rank-mirroring the planes equals
re-encoding with `!color`), not that the plane aligns with the observation
orientation. The invariant _requires_ one color to be at the "wrong" rank end.

Suggested fix: encode underpromo planes at obs rank 6→7 for **both** colors
(matches POV obs orientation, identical for white/black, and `flip_action_planes`
on that plane would give rank 1→0 — which would no longer equal the
`!color`-re-encoding, so the existing invariant test needs to relax).

**Note**: base-action e7-e5 example is POV-consistent because `flip_action`
rewrites the from/to in the index. Only underpromos are broken.

Affects: every training step that includes a Black-side underpromotion. Self-play
underpromos are rare, so the corruption is small in practice — but TB supervision
specifically over-samples KPK/KRKP/etc. positions where underpromos can occur,
making this more impactful for TB rows.

---

### 2. `trainer.py:652-657` — unbounded `/tmp/hyzero_diag_probe.txt` writes per training step

```python
with open("/tmp/hyzero_diag_probe.txt", "a") as _pf:
    _pf.write(f"[diag_reached] step={self.model_version}\n")
    _pf.flush()
```

This runs on **every** `train_batch` call, appends to `/tmp`, and re-opens the
file each time. At ~10–100 steps/sec over a long run, this grows unboundedly,
incurs syscall overhead, and pollutes the host's `/tmp`. The leading comment
calls it a "probe" — clearly leftover debug code that should be removed before
merging into a baseline run.

---

### 3. `trainer.py:607` — gradient-scaling hook captures the wrong tensor for consistency loss

```python
hidden = self.g(hidden, action_plane)
hidden_states.append(hidden)
hidden.register_hook(lambda grad: grad * 0.5)
```

The 0.5× gradient scale is registered on each dynamics output. But the same
tensor is later reused in the consistency loss path (`hidden_states[k_idx]`
feeds `self.h.predict(self.h.project(...))`). Gradients flowing back through
the consistency loss into `g`-output also get halved — likely _not_ what the
MuZero Appendix G prescription intends (consistency loss is supposed to give
`g` a direct gradient signal independent of `f`). The result is that the
EfficientZero consistency-loss gradient into `g` is silently weakened by 2×.

Either:

- detach `hidden` before passing to the consistency branch, or
- compute the consistency loss path with a separate tensor that doesn't carry
  the 0.5× hook.

---

## Medium severity

### 4. `trainer.py:633-637` — inconsistent reward-loss denominator between k=1 and k≥2 under TB

```python
if k >= 2:
    total_reward_loss = total_reward_loss + (per_sample_rwd * non_tb).sum() / non_tb_count
else:
    # k == 1: TB step-1 reward carries the real mating-action signal.
    total_reward_loss = total_reward_loss + per_sample_rwd.mean()
```

At `k=1` the mean is over **all B** samples; at `k≥2` it's the mean over
**non-TB only** samples. If TB rows have a strong `target_rewards[:,1]=+1`
signal mixed with replay's mostly-zero step-1 rewards, the k=1 term scales
differently from k≥2 contributions. Not a numerical disaster, but it does
mean step-1 reward error is averaged on a different denominator than step-2…K.
Consider always normalizing per-step by the same denominator (e.g., always
include all rows at every step — the trajectory format already sets
`is_tablebase=False` so the masking branch should be a no-op for trajectory TB).

### 5. `trainer.py:91` — `log_probs.nan_to_num` with `neginf=0.0` can silently mask real `-inf`

```python
log_probs = log_probs.nan_to_num(nan=0.0, neginf=0.0)
```

Only safe when the corresponding target probability is 0 at the masked
position (in which case `0 * 0 = 0` is the correct contribution). The
docstring explains this for the legal-mask case, but the same call is used
in `_policy_loss_per_sample` where there is **no** legal mask — and a `-inf`
log-prob at a position with non-zero target would silently contribute 0
instead of pushing the loss to infinity. For trained-model logits this
shouldn't produce `-inf`, but a NaN check earlier or `softmax_temperature`
guard would be safer. Low risk in practice; flagging for awareness.

### 6. `tablebase.py:189-191` — `random.sample` raises if `n > pool size` on the non-trajectory path

```python
if n >= len(pool):
    return random.choices(pool, k=n)  # with replacement
return random.sample(pool, n)         # without replacement
```

The boundary `n == len(pool)` goes to `random.choices` (with replacement), so
the user gets duplicated samples when asking for exactly the full set. This is
probably fine, but the docstring says "with replacement if n > len" — off by
one. Either change the condition to `n > len(pool)` or update the docstring.

### 7. `mcts/tree.rs:170-225` — `dirichlet_noise` Marsaglia–Tsang has a dead-code branch

```rust
let x: f32 = rng.random::<f32>() * 6.0 - 3.0; // rough normal approx
// Use Box-Muller for a proper normal sample
let u1: f32 = rng.random::<f32>();
let u2: f32 = rng.random::<f32>();
let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
...
if v <= 0.0 {
    let _ = x; // suppress unused warning
    continue;
}
```

`x` is computed and immediately thrown away with `let _ = x`. The intent
seems to have been to use `x` (cheap normal approx) before switching to
Box-Muller. Dead code, not a correctness bug — but worth removing for
clarity.

Also: `u1` can be 0 from `rng.random::<f32>()`, in which case `u1.ln() = -inf`
and `z` becomes `NaN`. A retry loop or `u1 = u1.max(f32::MIN_POSITIVE)` would
make this robust.

### 8. `build_tb_batch:266` — silent index truncation when `act >= num_actions`

```python
for act in sample.optimal_actions:
    if 0 <= act < num_actions:
        target_policies[i, 0, act] = policy_weight
```

If a malformed cache contains `act >= NUM_ACTIONS` (e.g., from a stale build
script), the entry is silently dropped, and the resulting `target_policies[i,0]`
will not sum to 1.0. The K-step loss MSE on values wouldn't notice, but the
cross-entropy policy loss now has a target that doesn't normalize. A
post-loop assert that `target_policies[i,0].sum() ∈ {0, 1}` would catch this.

---

## Low / style

### 9. `trainer.py:534` — `tb_indices` returned but never used

```python
batch, tb_indices = self._maybe_mix_tb_samples(batch)
```

The `set` is built but never consumed (the `is_tb_mask` bool array is what's
actually used). Drop it from the tuple.

### 10. `trainer.py:816` — redundant inner `if k_steps > 0`

The outer guard already established `k_steps > 0`. Dead code.

### 11. `selfplay/game_task.rs:32` — process-wide single trace writer truncates on first call

`logs/mcts_summary.log` is truncated on open. If two processes run concurrently
(rare for this codebase but possible — e.g., evaluator + trainer both linking
this binary) they race on the truncate. Not a bug given current usage; flag for
future-proofing.

### 12. `build_starting_positions.py:140-143` — sys.modules["__main__"] cleanup branch

Restoration logic only handles the case where `_prev is not None`. In normal
script execution there is always a prev, so this is correct in practice — but
the `else: del _sys.modules["__main__"]` branch present in
`tablebase.py:144` is absent here. Minor inconsistency.

### 13. `tablebase.py:992-998` — branching default for `is_tablebase`

```python
merged["is_tablebase"] = np.zeros(b, dtype=bool)
tb_flag = tb_dict.get("is_tablebase")
if tb_flag is not None:
    merged["is_tablebase"][b - n_tb:] = tb_flag
else:
    merged["is_tablebase"][b - n_tb:] = True
```

Both builders (`build_tb_batch` and `build_tb_batch_trajectories`) always set
`is_tablebase`, so the `else` is unreachable. Defensive but dead code.

---

## Things I checked and found OK

- **MCTS canonical backup** (`tree.rs:505-542`): the new `G_{k-1} = r_k − G_k`
  recurrence is correct and the new test `test_backpropagate_includes_mating_reward`
  pins the right behavior. Zero-reward paths still produce alternating signs
  identical to the old backup, so existing tests pass bit-for-bit.

- **Tie-break in `select_action`** (`tree.rs:584-608`): collects all tied-max
  indices and picks uniformly. Matches the rule in `mcts-pov-symmetry.md`.

- **Legal-action POV flip + sort** (`game_task.rs:272-290, 401-419`): both
  `play_game` and `play_game_dual` apply `flip_action` then `sort_unstable()` —
  matches the convention required by `mcts-pov-symmetry.md`.

- **Terminal reward POV** (`game_task.rs:537-542`): converts white-absolute
  `game_outcome` to last-step POV via `last_side_sign`. Pinned by
  `test_terminal_reward_pov_conversion`.

- **TB trajectory builder mate-step accounting** (`build_tablebase_trajectory_cache.py:336-344`):
  fires the reward at the absorbing step `k+1` (after the mating push), keeps
  `fens[k+1] = None`, leaves later steps zero. Matches the contract documented
  in `tablebase.py` `TBTrajectory`.
