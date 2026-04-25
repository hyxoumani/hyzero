# Branch Review Log: claude/modest-rubin-3kY0F

Tracks what has been code-reviewed on this branch so future passes can resume
from where we left off.

## Review Cursor

- **Last reviewed HEAD**: `ee132c4` (2026-04-25)
- **Base**: `origin/main` (3041902)
- **Reviewer**: Claude (Opus 4.7)
- **Pass**: 1

## Methodology

Per-commit code review focused on bugs (correctness, off-by-ones, panics, race
conditions, sign errors). Docs/log-only commits skipped. Findings recorded
with severity (high / medium / low / nit) and `file:line` references.

To resume on a future pass: read this file, then
`git log --oneline ee132c4..HEAD` to see what's new.

## Commits reviewed (pass 1)

Code commits (33 ahead of `origin/main`); docs/log-only commits skipped.

| sha               | summary                                                          | verdict                                |
| ----------------- | ---------------------------------------------------------------- | -------------------------------------- |
| ee132c4           | TB supervision infra + canonical backup + diverse starts         | clean (1 false-positive flagged below) |
| 7243aec           | mcts+selfplay: fix color asymmetry in self-play move selection   | clean                                  |
| 773cb90           | selfplay+training: POV isolation infra + terminal-reward POV fix | clean                                  |
| b012944           | training: fix value/reward target sign under color augmentation  | clean                                  |
| 2edb194           | mcts: make Dirichlet noise epsilon/alpha env-controllable        | clean (1 doc gap)                      |
| 6aff3d4           | encoding: initial-position color-symmetry regression test        | clean                                  |
| 5e201cc           | training: fix off-by-one in dynamics action indexing             | clean                                  |
| 7fca9ea           | training: EfficientZero self-supervised consistency loss         | clean                                  |
| 0882a7b → 08911f1 | draw penalty added then replaced w/ prioritized sampling         | clean (no dead code)                   |
| bdc8301           | training: stage entropy bonus on policy loss (off by default)    | clean                                  |
| 64300ce           | training: gentler default for decisive-sample fraction           | clean                                  |
| 72d0c8e           | training: diagnostic instrumentation for value head              | clean                                  |

## Findings

### M-1 (medium, doc gap, NOT a code bug) — Dirichlet env vars undocumented

**Where**: `CLAUDE.md` env-vars list vs `src/mcts/tree.rs:146,158`.

`HYZERO_DIRICHLET_EPSILON` (default 0.25) and `HYZERO_DIRICHLET_ALPHA` (default 0.3
for chess) were introduced in commit `2edb194` but are not listed in the
`Env vars` paragraph of `CLAUDE.md`. Implementation itself is sound: cached
`OnceLock` reader, bounds-checked (epsilon ∈ [0,1], alpha > 0), applied at the
right place in `MCTSTree::add_noise`.

**Suggested fix**: Append to the env-vars list in `CLAUDE.md`:

> `HYZERO_DIRICHLET_EPSILON` (default 0.25, fraction of root prior replaced
> by Dirichlet noise), `HYZERO_DIRICHLET_ALPHA` (default 0.3, Dirichlet
> concentration tuned for chess branching factor)

### Investigated and dismissed

**FP-1 (false positive)** — Reviewer-flagged "TB reward loss masks real mating
signal at k≥2" in `python/hyzero/training/trainer.py:633-637`.

Verified not a bug. The masking branch
`if k >= 2: ... (per_sample_rwd * non_tb).sum() / non_tb_count` only zeroes
out **TB-flagged rows**. The two TB batch builders set the flag deliberately:

- `build_tb_batch` (snapshot format) — `is_tablebase = np.ones(n)`
  (`python/hyzero/data/tablebase.py:239`). Snapshot rows have mate-at-step-1
  by construction; padding at k≥2 is intentional and masking is correct.
- `build_tb_batch_trajectories` (trajectory format) — `is_tablebase =
np.zeros(n)` (`python/hyzero/data/tablebase.py:345`). Trajectories supply
  real K+1-step targets; setting the flag to False intentionally bypasses the
  masking so the mating reward at any step k\* ∈ 1..K is supervised.

So mate at k≥2 is _not_ masked — it just goes through the
`(per_sample_rwd * non_tb).sum() / non_tb_count` path with `non_tb = 1.0` for
those rows, which reduces to the correct mean.

Note for future readers: the comment on line 630 ("step 1 has a real
mating-action target for TB rows") refers specifically to **snapshot-format**
TB rows. The comment is technically accurate but easy to misread as a
universal claim about all TB data — a one-line clarification would prevent
re-flagging on the next review.

### Code paths verified clean

- **Off-by-one fix (5e201cc)**: `actions[bi, k] = steps[k].action` for
  `k ∈ 0..K` is correct end-to-end (encoder → trainer → MCTS); regression
  test in place.
- **Consistency loss (7fca9ea)**: stop-gradient correctly applied to target
  side via `self.h.project(target_latent).detach()`; SimSiam projector/
  predictor split correct; weight wired through `HYZERO_CONSISTENCY_LOSS_WEIGHT`;
  no-op when weight=0.0.
- **Draw-penalty revert (0882a7b → 08911f1)**: full revert; `draw_penalty()`
  removed; `is_draw` field retained only for replay-pool classification in
  `replay_buffer.rs::sample_batch`; unit test
  `test_value_target_applies_draw_penalty` deleted; no dead code.
- **Entropy bonus (bdc8301)**: sign convention correct (`−β·H(π) = β·Σ π·log π`
  added to loss); NaN-safe — masked_fill + nan_to_num so illegal moves
  contribute `0·0=0`, no `-inf` gradient leakage.
- **Decisive-sample fraction (64300ce)**: clamped to `[0.0, 1.0]` in
  `replay_buffer.rs:73`; zero-decisive-samples falls back to uniform sampling
  at lines 101-105.
- **Value-head diagnostics (72d0c8e)**: all probes wrapped in `torch.no_grad()`;
  no gradients affected when disabled.
- **POV/color triple (b012944, 773cb90, 7243aec)**: `flip_sign` applied
  symmetrically to `root_value_target` and `outcome_in_step_perspective`;
  `game_outcome` correctly converted to step-POV using
  `last.white_to_move`; `select_action` and `legal_actions` sorting
  applied identically across both `play_game` and `play_game_dual`;
  conditional-β logic (`!sample.is_draw → β=1.0`) is correct given the
  semantics that `is_draw == true` means "non-checkmate terminal".

## Next pass

When new commits land past `ee132c4`, run:

```
git log --oneline ee132c4..HEAD
```

…then review only the new code commits and append findings under a new
"Pass 2" section.
