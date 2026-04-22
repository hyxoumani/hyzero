# Mistakes Log

Record of agent failures with root cause analysis and error classification.

## 2026-04-13: Zobrist Implementation — Incomplete Cleanup

**Date**: 2026-04-13
**Agent**: Implementation agent (Task 30.1)
**Domain**: Chess engine (zobrist hashing)
**Error Type**: Quality — incomplete cleanup of old code

**What happened**: Zobrist table implemented correctly with incrementally-maintained hash, but old `position_hash()` function left in codebase and no tests added initially.

**Root Cause**: First implementation didn't follow the "test-driven" pattern. Function was added but old code wasn't removed, causing confusion about which hash to use. Lack of explicit test coverage for zobrist incremental updates meant the bug wasn't caught at commit.

**Fix**: Follow-up agent removed `position_hash()`, added 4 tests for zobrist hash consistency (initial, after move, after capture, after castling/EP). Tests now verify that `zobrist_hash` field matches recalculated hash.

**Escalation Tier**: Gotcha → added to wiki (chess-engine.md, item 1: Zobrist maintains incrementally, replaces old position_hash)

---

## 2026-04-13: Auto-format Hook — CWD-Relative Paths

**Date**: 2026-04-13
**Agent**: Implementation agent (Task 30.3)
**Domain**: Development infrastructure (formatting hooks)
**Error Type**: Context — acted on incomplete environment state

**What happened**: `auto-format.sh` hook ran after edit, but failed silently when cwd was `/home/devs/workspace/hyzero/python/` (agent had `cd python` to run pytest). Hook script used relative paths (e.g., `./target/release/rustfmt`) that don't exist from `python/` directory.

**Root Cause**: Hook script didn't check its own directory context or use absolute paths. When an agent `cd`'s into a subdirectory (reasonable for running tests), the hook becomes misconfigured.

**Fix**: Update `auto-format.sh` to use absolute paths or explicitly cd to repo root before running formatters. Ensure hooks always work from any cwd.

**Escalation Tier**: Rule — add to `.claude/rules/hooks.md` or CLAUDE.md: "Hooks must use absolute paths or explicitly cd to repo root before executing."

---

## 2026-04-13: Perft Agent — Missing Castling Constraint

**Date**: 2026-04-13
**Agent**: Validation agent (Task 30.5 perft)
**Domain**: Chess engine (move generation)
**Error Type**: Context — incomplete understanding of move generation API

**What happened**: Perft agent wrote test that assumed all legal king moves (including castling) would be in `get_move_mask(king_sq, color)`. Test failed because castling moves are generated separately via `get_castling_moves()`, not included in the basic move mask. This caused discrepancy in node counts.

**Root Cause**: Move generation API asymmetry: king's move mask only includes 1-square moves (precomputed table). Castling is special-cased. The interface doesn't make this obvious. Selfplay code (game_task.rs) already handles this correctly by calling both functions, but the constraint wasn't documented.

**Fix**: (1) Perft test updated to call both `get_move_mask()` and `get_castling_moves()` for king. (2) Wiki updated with gotcha #6: "Castling not in king move mask" and explicit note that move generation code must call both functions.

**Escalation Tier**: Gotcha → encoded in chess-engine.md and code comments. Added to perft.rs as inline doc comment on how to correctly enumerate king moves.

---

## 2026-04-13: Perft Terminal Counting — Non-Standard Convention

**Date**: 2026-04-13
**Agent**: Validation agent (Task 30.5 perft)
**Domain**: Chess engine (perft counting)
**Error Type**: Quality — convention violation

**What happened**: `perft(startpos, d=5)` returned 4,865,617 instead of 4,865,609 — exactly 8 extra nodes. Test comparison against python-chess reference showed mismatch. Root cause found: perft implementation was counting checkmate positions as +1 leaf instead of recursing to depth 0.

**Root Cause**: In `src/game/perft.rs`, the perft function had a terminal-position check: `if new_board.result() == GameResult::Ongoing { recurse } else { nodes += 1 }`. This implements a non-standard perft convention. Standard perft counts reachable positions at depth D, not terminal nodes. The +1 was counting Fool's Mate checkmates (8 of them reachable at ply 4 with depth=5 remaining).

**Fix**: Removed the terminal check and always recurse. The `depth == 1` shortcut already returns `legal_moves.len()`, which is 0 for checkmate. Verified with diagnostic `slow_perft_no_terminal` variant.

**Escalation Tier**: Gotcha → documented in chess-engine.md test coverage section ("terminal counting fixed"). Code comment added to perft.rs explaining standard convention.

---

## 2026-04-13: Researcher Session Timeout — Missing Memory Persist

**Date**: 2026-04-13
**Agent**: Researcher agent (perft d5 bug investigation)
**Domain**: Development infrastructure (agent workflow)
**Error Type**: Context — incomplete session close procedure

**What happened**: Researcher agent investigating perft d=5 overcount timed out at 2.7M ms (45 min) without writing findings to `/home/devs/workspace/hyzero/.claude/agent-memory/researcher/perft_d5_bug.md`. The analysis was correct but lost because the file wasn't written before timeout.

**Root Cause**: Agent got caught in a deep investigation loop (hunt_overcount, hunt_wrong_terminal, hunt_duplicates, hunt_missed_terminal) without periodically flushing findings. Timeout occurred before the summary could be written.

**Fix**: (1) Researcher completed the perft_d5_bug.md file in a follow-up session. (2) Added note to agent-memory/README.md: "Persist findings every 10 min of investigation to avoid timeout loss."

**Escalation Tier**: Rule (candidate) — could add to CLAUDE.md under agent workflow: "Write findings to agent-memory every 10-15 minutes during long investigations to avoid timeout loss."

---

## 2026-04-13: calculate_pins() Queen Omission

**Date**: 2026-04-13
**Agent**: Researcher agent (perft position 6)
**Domain**: Chess engine (pin detection)
**Error Type**: Quality — incomplete logic

**What happened**: `calculate_pins()` in `src/game/board.rs` was missing Queen from the `enemy_sliders` bitboard. This caused false negatives in `calculate_checkmate()` and `calculate_stalemate()` when the pinning piece was a queen. Regular move validation was unaffected.

**Root Cause**: Pin calculation built `enemy_sliders` from Rook and Bishop only:
```rust
player2.pieces_bb[Rook] | player2.pieces_bb[Bishop]  // Missing Queen
```
Copy-paste error from earlier code that may have predated Queen sliding piece support.

**Fix**: Added Queen to `enemy_sliders` in both Color branches. Tests already pass because the pin-detection bug only affects the internal bitmask; `validate_move()` uses clone+apply+check instead.

**Escalation Tier**: Gotcha → added to chess-engine.md gotcha #8 as a reminder to include all sliders in pin detection.

---

## 2026-04-13: Stalemate Castling Escape — Missing Check

**Date**: 2026-04-13
**Agent**: Validation agent (Task 31 engine validation)
**Domain**: Chess engine (game termination — stalemate)
**Error Type**: Quality — incomplete logic

**What happened**: `calculate_stalemate()` checked if the king had legal moves by iterating 1-square king moves and checking pins. Castling was never checked as an escape option. This is extremely rare but legal: in some positions, castling is the only legal move and prevents stalemate.

**Root Cause**: Stalemate logic assumed castling was not a valid escape. The function checked `get_move_mask(king_sq, color)` which returns only 1-square moves; castling generated separately via `get_castling_moves()`. Code path didn't call the second function.

**Fix**: After the king 1-square loop, added explicit calls to `validate_move()` for both kingside and queenside castling options. If either is legal, return `false` (not stalemate). Added 2 tests covering this edge case.

**Escalation Tier**: Gotcha → added to chess-engine.md gotcha #10 (stalemate must check castling escape).

---

## 2026-04-13: Stalemate Parameter Ordering — Bits Swapped for Black

**Date**: 2026-04-13
**Agent**: Validation agent (Task 31 engine validation)
**Domain**: Chess engine (game termination — stalemate)
**Error Type**: Quality — parameter swapping

**What happened**: `calculate_stalemate()` called `get_move_mask(sq, color)` by passing `(friendly_bits, opponent_bits)` as the occupancy parameter. However, `get_move_mask()` expects `(white_pieces, black_pieces)` — canonical color order, not relative to the moving player. For Black-to-move, the bits were swapped, causing incorrect move masks and missed legal escapes.

**Root Cause**: Parameter naming confusion. The function tried to optimize by passing relative (friendly/opponent) bits, but the magic bitboard lookup tables precompute moves for absolute color positions (white/black). The mismatch meant Black-to-move got swapped occupancy and returned wrong moves.

**Fix**: Derive canonical `white_bits` and `black_bits` from `color` at the function entry, then pass these to `get_move_mask()` instead of friendly/opponent. Added 7 tests covering Black stalemate scenarios.

**Escalation Tier**: Gotcha → added to chess-engine.md gotcha #9 (stalemate must pass canonical white/black bits to get_move_mask).

---

## 2026-04-13: Threefold Repetition Off-by-One — Initial Position Not Counted

**Date**: 2026-04-13
**Agent**: Validation agent (Task 31 engine validation)
**Domain**: Chess engine (draw rules — threefold repetition)
**Error Type**: Quality — initialization bug

**What happened**: `threefold_repetition()` always returned `false` in positions where the same position appeared 3 times total, requiring 4 repetitions to trigger the draw. Root cause: `position_history` map started empty. The initial board position was never inserted, so the first occurrence wasn't counted. A position seen after move 5 would count as occurrence #1 (not #2), requiring move 15 and move 25 to reach 3 total.

**Root Cause**: Both `init_game_board()` and `board_from_fen()` created an empty `position_history`, then called `update_board()` for the first time. The update increments the hash count, but never initializes it as occurrence #1. Standard threefold repetition counts the initial position as the first occurrence.

**Fix**: After board construction, explicitly insert `board.position_history.insert(board.zobrist_hash, 1)` to register the initial position as occurrence #1. Now the 2nd and 3rd occurrences properly trigger the draw rule at counts 2 and 3. Added 2 tests.

**Escalation Tier**: Gotcha → documented in wiki but considered a critical draw-rule fix. Not escalated to Rule because it only affects initialization (single fix point).

---

## 2026-04-13: PyO3 Test Mock Data — Production Code Mismatch

**Date**: 2026-04-13
**Agent**: Implementation agent (Task 32.3 — checkpoint resume)
**Domain**: Rust-Python Integration (PyO3)
**Error Type**: Quality — test setup masking production bug

**What happened**: `test_resume_checkpoint_restores_model_version` passed, but production `load_checkpoint()` was reading `model_version` from the wrong source. Test constructed a mock Python return dict and manually inserted `"model_version"` into it. Production code attempted to read from this dict, but the Python trainer object never populated it. The bug would have caused a panic at runtime with real checkpoints.

**Root Cause**: Test setup did not match production Python behavior. Production `load_checkpoint()` in Python trainer saves model state but does not return model_version in a dict. Instead, it's available as a trainer object attribute (`trainer.model_version`). The test mocked a return dict, and production code was written to read from it. The mock made the test pass, but the code would fail at runtime.

```rust
// WRONG (what was written):
let version_dict: Py<PyDict> = ...;  // mock dict with "model_version" key
let version: u64 = version_dict.getattr(py, "model_version")?.extract(py)?;

// RIGHT (what was fixed):
let version: u64 = self.trainer.getattr(py, "model_version")?.extract(py)?;
```

**Fix**: Changed `load_checkpoint()` to read `model_version` directly from the trainer object attribute via `trainer.getattr("model_version")` instead of expecting it in a return dict. Updated test to match.

**Escalation Tier**: Rule — add to `.claude/rules/testing.md` or create new `.claude/rules/pyo3.md`: "When testing PyO3 return values, verify the test setup matches what production Python code actually returns. Don't manually insert values that production code never provides. Mock data should mirror real Python behavior or use a fixture that does."

---

## 2026-04-14: Log-Softmax NaN with Legal-Move Masking

**Date**: 2026-04-14
**Agent**: Implementation agent (Batch 1 Representation Overhaul)
**Domain**: Python trainer (loss computation)
**Error Type**: Quality — incomplete masking handling

**What happened**: Legal-move masking was applied by setting illegal logits to `-inf`, then calling `F.log_softmax()` on the masked logits. This produced `0 × log(0)` NaN values in the final log-probabilities for illegal moves, which contaminated the policy loss during backprop.

**Root Cause**: `log_softmax(-inf)` returns NaN when the exponential underflows. The masked-fill operation sets illegal actions to `-inf`, but `log_softmax` doesn't handle this edge case gracefully. Standard masking workflows apply masking *after* softmax, not before.

**Fix**: Apply masking before softmax, then call `nan_to_num(neginf=0.0)` to replace NaN with 0. This makes the loss contribution from illegal moves identically zero, as intended. Code pattern:
```python
logits = self.policy_head(x)
logits.masked_fill_(~legal_mask, float('-inf'))
log_probs = F.log_softmax(logits, dim=-1)
log_probs = log_probs.nan_to_num(neginf=0.0)
```

**Escalation Tier**: Gotcha — documented here for manual avoidance.

---

## 2026-04-14: Bash set -euo pipefail with Empty Grep Tail

**Date**: 2026-04-14
**Agent**: Implementation agent (Baseline evaluation)
**Domain**: Development infrastructure (shell scripts)
**Error Type**: Quality — incomplete error handling

**What happened**: Shell script with `set -euo pipefail` running `grep pattern file | tail -n 1` silently exited the entire script when grep returned no matches. The pipe breaks and tail reads EOF, exiting with code 0, but the compound command had already failed, causing the script to abort.

**Root Cause**: When `grep` finds no matches, it returns exit code 1. With `pipefail`, the compound pipeline fails. However, if tail were to run, it would exit 0 on EOF. The script interprets this as a catastrophic error and exits due to `set -e`.

**Fix**: Guard grep with `|| true` or use `grep ... | tail -n 1 || echo "default"` to provide a fallback. Alternatively, save grep output to a variable first and check before piping. For robustness:
```bash
result=$(grep pattern file || echo "")
if [ -z "$result" ]; then
  # handle no match
fi
```

**Escalation Tier**: Gotcha — documented here for manual avoidance in shell scripts.

---

## 2026-04-14: MCTS Tree Reuse — Q-Value Warm-Start Regression

**Date**: 2026-04-14
**Agent**: Implementation agent (Autoresearch Batch 2)
**Domain**: MCTS search optimization
**Error Type**: Experimental regression (plan deviation with unmeasured actual implementation)

**What happened**: Commit `118f824` attempted to implement MCTS tree reuse between moves to speed up search. Initial plan: carry forward the entire subtree from the previous move (AlphaZero-style). Implementation discovered MuZero latent children don't store board state, so legal moves aren't computable. Pivoted to a weaker version: warm-start the root Q-value with accumulated value from prior move's expansion. Result: devastating regression across all three metrics (score 1.8121 vs 5.7646 baseline, -3.95 delta). All games were draws, longer than baseline (227 vs 182 moves), policy loss regressed (4.47 vs 3.96).

**Root Cause**: When plan is blocked by architectural constraints, the modified experiment is no longer testing the original hypothesis. Q-value warm-start with early-training network noise injected bias: weak expansions from the first 100 games produced noisy latent states and poor value estimates. These warm-started Q-values misled PUCT selection away from actually good moves. The training network was undermined by stale, noisy warm-start signal, leading to drawish, ineffective play.

**Fix**: Reverted commit `118f824` with `d0b0681`. No tree reuse in current codebase.

**Lesson**: Measure expected delta of the *actual* implementation before running, not the intended design. If the implementation deviates fundamentally from the plan, run a smaller feasibility test (5 min run) to validate the pivot before committing to a full baseline.

**Escalation Tier**: Gotcha — documented in mcts-selfplay.md section "Tree Reuse: Why It's Hard in MuZero" with detailed architectural explanation and future options.

---

## 2026-04-14: Recency-Weighted Replay Buffer — Catastrophic Forgetting

**Date**: 2026-04-14
**Agent**: Autoresearch (commit 003eaf9)
**Domain**: Training data distribution (replay buffer)
**Error Type**: Insight (not a bug) — illuminated subtle tradeoff between batch quality and value-head diversity

**What happened**: Commit `003eaf9` added exponential decay weighting to replay buffer sampling to prioritize recent on-policy games. Two runs tested:
- **decay=0.1**: policy loss improved dramatically (3.02 vs 3.96 baseline, -0.94), but decisive_ratio collapsed from 0.50 at v20 to 0.10 by v57. Score 4.7791 (-0.99 vs baseline 5.7646).
- **decay=0.05**: policy loss even better (2.93), 78% more throughput, but same collapse pattern (v20 decisive=0.20 → v61 decisive=0.10). Score 5.2719 (-0.49).

Classic catastrophic forgetting: exponential decay progressively narrows the effective buffer distribution. By v57-61, only the last ~10 model versions have non-negligible weight, so the value head trains on a narrow recent-only distribution and learns unreliable value estimates. Policy loss continued decreasing (network avoided costly mistakes) but value head lost signal diversity, leading to drawish, repetitive play.

**Root Cause**: Early random games provide high-variance outcome distribution (wins, losses, draws). Recent on-policy games converge toward a narrow distribution (likely draws and repetitions). Recency weighting trades value-head diversity for policy-batch quality. This is especially damaging when the policy is already decent — then the narrow distribution trains the value head to expect nearly-zero outcomes, making it fail when the policy diverges.

**Fix**: Reverted via `ec2f6c0` to equal-weight sampling (commit `d0b0681` baseline).

**Lesson**: When policy loss and decisive_ratio move in opposite directions, you've traded one for the other. Log both first-eval and last-eval values to spot forgetting. The *true* metric should measure the first eval cycle too (e.g., v20 was good, but v60 is bad — the model forgot).

**Escalation Tier**: Gotcha — documented in mcts-selfplay.md section "Replay Buffer Distribution Dynamics" with signature symptom (eval cycles diverge) and future options (separate samplers, diversity floor, outcome prioritization).

---

## 2026-04-14: Self-Play Evaluation Symmetry Collapse

**Date**: 2026-04-14
**Agent**: Autoresearch (runtime HYZERO_SIMS=100, no code change)
**Domain**: Evaluation metrics (self-play vs fixed opponent)
**Error Type**: Metric flaw — optimization misalignment with model quality

**What happened**: Three consecutive autoresearch experiments (recency decay=0.1, decay=0.05, SIMS=100 runtime override) all exhibited identical failure mode: policy_loss improved, but decisive_ratio dropped to 0 (all eval games drew). Pattern is now statistically significant:
- recency decay=0.1: policy loss 3.02 vs 3.96 baseline ✓ but v20 decisive 0.50 → v57 decisive 0.10 ✗
- recency decay=0.05: policy loss 2.93 ✓ but v20 decisive 0.20 → v61 decisive 0.10 ✗
- SIMS=100 (runtime, no code change): policy loss 3.09 vs 3.96 baseline ✓ but eval at v20 all draws (0 decisive) ✗

**Root Cause**: This is **fundamental to symmetric self-play evaluation**. As the MuZero policy and value head improve, both players in self-play converge to identical strong play → identical game trees → all games draw. The stronger the model, the more likely self-play-vs-self produces draws. Conversely, weaker or high-variance policies produce more decisive games. The `training_score` metric is currently optimized by either (a) intentional model weakness/variance, or (b) asymmetric evaluation. Measuring self-play decisive_ratio rewards the wrong thing past a certain point.

**Fix**: This is not a bug to fix, but a metric to redesign. The `training_score` formula should not use self-play-decisive-ratio as a signal. Better alternatives: (1) Win rate vs a FIXED reference opponent (e.g., `RandomEvaluator` from `src/selfplay/evaluation.rs`), (2) Puzzle-solving rate (separate benchmark), (3) Blitz tournament vs historical versions.

**Lesson**: Self-play win rates measure relative strength between versions. They do not measure absolute strength. When both players are identical, all games draw regardless of skill. For eval, use an asymmetric opponent (fixed or random).

**Escalation Tier**: Metric design — escalates to project roadmap. Proposed: track `training_score` component changes separately (policy_loss, avg_length) and defer win-rate-based scoring to Phase 4 (Tactical Metric) with dedicated `RandomEvaluator` benchmark.

---

## 2026-04-14: Baseline Measurement Noise — ±1 Point Variance

**Date**: 2026-04-14
**Agent**: Baseline rerun / measurement validation
**Domain**: Development infrastructure (metric reliability)
**Error Type**: Quality — hidden measurement noise, masked by single-run reporting

**What happened**: Four consecutive runs of equivalent or near-equivalent code produced widely scattered scores: 5.7646, 3.6911, 3.6947, 4.1277 (mean ~4.3, range ~2.0). The original "Batch 1 win" at 5.7646 appeared decisive, but was within noise of the initial baseline 4.7798. Subsequent reruns at commits 46c3d0d-rerun1 and 46c3d0d-rerun2 dropped to ~3.7. Discovery during third rerun revealed the script bug (see below) — but also exposed that even with the bug fixed, variance remains ±1.0 point per run.

**Root Causes**:
1. **Measurement bug** (fixed in d98289b): Script used `grep | tail -1` to extract the LAST eval cycle's decisive_ratio. But catastrophic forgetting means the LAST cycle is often collapsed (0.00) even when earlier cycles had 0.30-0.50. Script should have picked MAX decisive_ratio across all eval cycles. Fixed: now uses `grep -oP 'decisive_ratio: \K[\d.]+' | sort -nr | head -1` or equivalent.
2. **Inherent eval noise**: Each self-play eval runs only 10 games, so decisive_ratio has binomial variance (±0.15-0.20 per cycle). Training step count varies ±50% between runs due to OS scheduler/inference throughput jitter. Over a 30-min run with 5-7 eval cycles, these compound.

**Implication**: Single-run experiments with effect size <1.5 points are indistinguishable from noise. Batch 1's +0.98 improvement (5.7646 vs 4.7846 baseline) is marginal. True effect, if any, requires multi-run averaging or longer baseline runs.

**Fix**: (1) Script fixed to use MAX cycle. (2) New baseline established at 4.1277 (commit d98289b, 2026-04-14) — the median-ish value of recent 4 observations. (3) Future experiments: rerun marginal changes (±0.5 points) at least 2x before claiming. (4) Roadmap Phase 4: implement multi-run averaging (each experiment runs 3x, median reported).

**Escalation Tier**: Gotcha → documented in testing.md under "Baseline Measurement Reliability" section with rerun guidance. Rule candidate: add to CLAUDE.md metric section that changes <1.5 points require validation reruns.

---

## 2026-04-14: Dirichlet Noise Alpha — Wrong Game Constant

**Date**: 2026-04-14
**Agent**: Autoresearch (commit d407281)
**Domain**: MCTS root exploration (Dirichlet noise)
**Error Type**: Wrong hyperparameter — long-standing constant copy from different game domain

**What happened**: `NOISE_ALPHA` in `src/mcts/tree.rs:9` was hardcoded to 0.03 (AlphaZero **Go** setting) when chess requires 0.3 per the AlphaZero paper. This was a systemic bug affecting every experiment from inception. Over-concentrated Dirichlet noise at the root meant all exploration mass concentrated on 1-2 random moves out of ~35 legal moves → narrow state space coverage → training data biased toward a few game patterns → network learned slowly → decisive_ratio stuck at ~0.20.

**Evidence of impact**: 15-min baseline comparison:
- **Before fix** (α=0.03): score 4.13, policy_loss 4.32, decisive_ratio 0.20, 45 games in 347 steps
- **After fix** (α=0.3): score 6.78, policy_loss 3.40, decisive_ratio 0.30, 97 games in 768 steps
- **Delta**: +2.65 score (well above ±1.0 noise floor), −0.92 policy loss, +0.10 decisive_ratio, +2.16x throughput

**Root Cause**: Copy-paste from AlphaZero codebase without domain validation. The paper explicitly states: "α = {0.3, 0.15, 0.03} for chess, shogi and Go respectively." The value 0.03 was correct for Go; chess needed 0.3.

**Fix**: Commit `d407281` — single-line change: `const NOISE_ALPHA: f64 = 0.03;` → `0.3`. Also sets correct `NOISE_EPSILON = 0.25` (fraction of noise mixed into prior; already correct).

**Escalation Tier**: Rule — add to `.claude/rules/mcts.md` to prevent future domain-crossing errors.

---

## 2026-04-15: Value Head Dead — Self-Referential Bootstrap

**Date**: 2026-04-15
**Agent**: Architectural analysis (no code change)
**Domain**: Training dynamics (MuZero value head)
**Error Type**: Design issue (suboptimal target formula)

**What happened**: Deep investigation of training logs showing `value=0.0000` throughout all experiments. Analysis confirmed the problem is architectural, not a tuning parameter. Value head never receives gradient signal despite correct implementation of training loop.

**Root Cause**: Value target formula is self-referential. Code (`src/py/training.rs:98`) sets:
```
target_values[bi * kp1 + k] = step.root_value;
```
where `root_value` is the MCTS root node's value estimate, which is initialized from (and backed up through) the value head's own output. When the value head is untrained (f(s) ≈ 0), all root_value ≈ 0 → all targets ≈ 0 → loss ≈ 0 → no gradient. The bootstrap loop never closes.

This is **not canonical MuZero**. Schrittwieser et al. (2020) Appendix F specifies value target = game outcome (n-step bootstrap with n=∞, γ=1) for board games. Our approach (bootstrapping from untrained network) is a failed optimization attempt.

**Verified by code**:
- `game_outcome` IS available in `GameTrajectory` (src/data/replay_buffer.rs:93, types.rs:96)
- But `game_outcome` is NEVER passed to training batch assembly (src/py/training.rs line 98)
- Only the final step's `reward` field contains the outcome (src/selfplay/game_task.rs:150)

**Lesson**: Design patterns that work in theory don't always work in practice. The iterative bootstrap hypothesis sounded good, but the self-referential loop kills the gradient signal. Canonical approaches (outcome targets) exist for a reason.

**Escalation Tier**: Gotcha → documented in neural-networks.md (new section "Canonical MuZero Value Target for Board Games") and mcts-selfplay.md ("Why the Value Head is Dead").

---

## 2026-04-15: Reward Head Dead — Class Imbalance (Terminal Reward Only)

**Date**: 2026-04-15
**Agent**: Architectural analysis (no code change)
**Domain**: Training dynamics (MuZero reward head)
**Error Type**: Design issue (sparse reward targets)

**What happened**: Training logs show `reward=0.0006` (collapsed to zero) despite correct reward head implementation. Investigation revealed the problem is fundamental to the data pipeline: reward targets are 99% zeros.

**Root Cause**: Only terminal steps carry outcome signal.
- `src/selfplay/game_task.rs:107-114` initializes every step with `reward: 0.0`
- `src/selfplay/game_task.rs:149-151` sets only the **last step** to `reward = game_outcome`
- For a 100-move game, random K+1-step sampling (src/data/replay_buffer.rs:87-89) has ~1% chance of including the terminal step
- In typical batches, ~99% of `target_rewards` entries are 0.0

An MSE loss on 99% zeros + 1% outcome signal is optimized by predicting 0 everywhere.

**Why This Breaks MuZero**: MuZero requires the reward head to signal terminal states in latent space (no real board exists to check `.result()`). Dead reward head means MCTS backup can't detect "game over" — the tree may continue expanding nonsensically past actual terminals.

**Lesson**: Reward-only-at-terminal is correct physics but creates severe class imbalance during training. Supervised learning needs balanced targets. Options: (a) resample to oversample terminals, (b) use outcome as a per-step auxiliary target, (c) separate sparse-reward and dense-loss formulations.

**Escalation Tier**: Gotcha → documented in neural-networks.md ("Key Gotchas #10") and mcts-selfplay.md ("Why the Reward Head is Dead").

---

## 2026-04-15: Game Outcome Perspective — Absolute White vs Side-to-Move Relative (RESOLVED)

**Date**: 2026-04-15
**Agent**: Architectural analysis (verified during material-signal fix)
**Domain**: Data encoding (MuZero observation consistency)
**Error Type**: Context — incomplete understanding, not a bug

**What happened**: Initial analysis noted outcome targets are White-absolute, but the network sees absolute piece positions + side-to-move indicator. Raised concern that outcome → value target conversion wasn't done correctly.

**Investigation**: VERIFIED RESOLVED in commit 1846b78. The conversion IS implemented at `src/py/training.rs:136`:
```rust
let ply_flip: f32 = if k % 2 == 0 { 1.0 } else { -1.0 };
let outcome_in_step_perspective = sample.game_outcome * root_side_sign * ply_flip;
```

This applies both `root_side_sign` (extracted from root observation) and `ply_flip` to convert White-absolute outcome to step-relative perspective. Correctly handles both the side-to-move and ply parity.

**Root Cause**: Initial analysis predated material-signal-fix implementation. The conversion logic was already in the codebase but wasn't thoroughly traced during the earlier investigation.

**Escalation Tier**: CLOSED. Not escalated to Rule because this is not a recurring error — it's a single misunderstanding now verified correct.

---

## 2026-04-15: Metric Inflation from Training-Version Tag vs Promotion Count

**Date**: 2026-04-15
**Agent**: Autoresearch session
**Domain**: Development infrastructure (metric definition and measurement)
**Error Type**: Quality — ambiguous variable name masking a logic error

**What happened**: A 30-minute training run reported `training_score = 28.33` (final JSON output), but manual inspection of the log revealed only 2 promotions (`grep -c "\[eval\] promoted"` returned 2). Expected score ~8.3 based on 2 promotions; actual reported 28.33 is 3.4× inflated.

Root cause: The bash script extracting the metric multiplied by `max_champion_version = 12` instead of the promotion count. The naming confusion: `champion_version` is a tag from checkpoint filenames (`best_v012.pt`), not a counter of promotion events. Over 30 minutes, the model trained through ~12 version checkpoints while only promoting twice — a ~6× rate difference between training steps and eval cycles.

**Fix**: Commit `2a273d4` — replaced `max_champion_version` multiplier with explicit event count `PROMOTIONS=$(grep -c "\[eval\] promoted" "$run_log")`. Formula now correctly uses the number of discrete promotions, not the version tag of the latest checkpoint.

```bash
# BEFORE (wrong):
champion_version=$(grep -oP 'max_champion_version:\K\d+' "$run_log" | tail -1)
score=$(echo "8.55 - $policy_loss + $champion_version * 2.0 ..." | bc)

# AFTER (correct):
PROMOTIONS=$(grep -c "\[eval\] promoted" "$run_log")
score=$(echo "8.55 - $policy_loss + $PROMOTIONS * 2.0 ..." | bc)
```

**Lesson**: Any derived metric whose multiplier can change at a different rate than the event it's supposed to count needs precise definition. "Version" is ambiguous — it can mean build number, model checkpoint index, or event serial. Use "count" language when you mean discrete events. Test the metric extraction against ground truth before high-stakes reporting. The baseline was correct before (commit d407281 measured honestly), but this bug would have invalidated future experiments if left unfixed.

**Escalation Tier**: Gotcha → documented in testing.md section on "Baseline Measurement Reliability" with note on metric definition precision.

---

## 2026-04-15: Value Loss Weight Overshoot — Destabilizes Whole Pipeline

**Date**: 2026-04-15
**Agent**: Autoresearch session
**Domain**: Training hyperparameter tuning (loss weights)
**Error Type**: Experimental instability — closed-loop feedback in multi-head training

**What happened**: Test of `HYZERO_VALUE_LOSS_WEIGHT=5.0` (amplify value gradient 5x) at β=0.3 (outcome blend). Hypothesis: value loss was 60x smaller than policy loss, so boosting would accelerate value head training. Result: catastrophic regression from 11.63 to 4.84 score (−6.79), 0 promotions. Notably: policy loss achieved new best (2.70 vs baseline 3.40), but challenger **lost to Random** at eval cycles 3–4 (win_rate=0.375 against trivial opponent). Training converged fast (11 eval cycles, ~102-move games).

**Root Cause**: Value head overshoot created a feedback loop. With 5x gradient, early-training value estimates oscillate wildly because the network hasn't stabilized. MCTS uses value estimates to prune the search tree via PUCT selection. Poor value estimates → poor move selection during self-play → poor training data generated. The policy head then trains on garbage data (learning how to avoid costly moves in bad positions that shouldn't have existed). The policy loss *looks good locally* (network is learning which moves to avoid) but the play quality collapses because the data generator (MCTS under poor value guidance) was corrupt from the start.

**The insight**: MuZero training is a **closed-loop multi-head system**. Each network head's quality directly affects the training data distribution the other heads see. In a two-player game via MCTS, the value estimate controls which positions are explored and which are pruned. Amplifying one head's gradient without matching the others' convergence rate creates a stability feedback loop: one head corrupts the shared state/action trajectory before the others can adapt.

**Comparison table**:
| Config | Policy Loss | Promotions | Score | Eval Result |
|--------|------------|-----------|-------|-------------|
| Baseline (β=0.3) | 3.40 | 3+ | 11.63 | Wins drawn |
| VALUE_WEIGHT=5.0 | 2.70 ✓ | 0 ✗ | 4.84 | Loses to Random |

**Fix**: Keep all loss weights at 1.0. To increase value learning signal, tune the outcome blend β instead of the loss weight. Example: if you want value head to see more outcome signal, try β=0.5 (50% outcome, 50% Q-estimate) instead of increasing weight. This scales the target signal without amplifying gradient instability.

**Escalation Tier**: Gotcha — documented in neural-networks.md and learning rule encoded as: "Loss weights (HYZERO_{POLICY,VALUE,REWARD}_LOSS_WEIGHT) default to 1.0 and should stay near that."

---

## 2026-04-15: Fast-Training Paradox — Lower Loss Doesn't Mean Better Model

**Date**: 2026-04-15
**Agent**: Autoresearch session (11-experiment β sweep)
**Domain**: Training dynamics (closed-loop self-play system)
**Error Type**: Insight — non-obvious metric misalignment in multi-head learning

**What happened**: An 11-experiment autoresearch sweep tested variations in loss weights, game counts, simulations, learning schedules, and the β outcome-blend parameter. Experiments with the *best* training metrics (policy loss 2.4–2.7) all **regressed catastrophically** in promotions and play quality. Meanwhile, the configuration with *worse* training loss (β=0.3, loss 3.40) achieved 4 promotions and peak score 11.63. Pattern confirmed across 6 independent experiments:

| Config | policy_loss | promotions | score | note |
|--------|---|---|---|---|
| value_weight=5.0 | 2.70 (best) | 0 | 4.84 | Lost to Random |
| games_per_side=6 | 2.41 (best) | 0 | 5.48 | Lost to Random |
| β=0.4 | 2.63 | 1 | 6.80 | Weak play |
| β=0.3 | 3.40 (worst) | 4 | **11.63 (winner)** | Strong play |
| β=0.2 | 3.26 | 2 | 8.33 | OK play |

**Root Cause**: MCTS self-play is a **closed-loop system**. The model generates training data (via MCTS value-guided search), trains on that data, and the updated model generates the next games. If the training speed increases (lower loss) without a corresponding improvement in MCTS quality, the effect is that the model trains on lower-quality training data. Here's why:

1. Early in training, MCTS has poor value estimates (network untrained).
2. If the policy trains too fast (faster loss reduction), the network converges to a locally-good policy before MCTS builds reliable value estimates.
3. The policy learns to avoid costly moves, but it learned this from games where MCTS was making bad move choices due to unreliable values.
4. Policy loss *looks good* because the network faithfully memorizes the MCTS visit-distribution targets. But those targets reflect whatever MCTS produced (which was poor).
5. Result: excellent training loss, but when the model plays against a fixed opponent (eval), it loses because it learned bad play patterns.

This is not a bug in code — it's a **metric misalignment**. The metric "policy loss" measures *local* learning (how well targets are fit) but not *global* play quality (whether the model actually improves). These are decoupled in this pipeline.

**Comparison**: In supervised learning (fixed data), lower loss = better model. In RL with self-play, lower loss can mean "we're memorizing garbage more faithfully."

**Key insight**: The β=0.3 winner had:
- **Longer games** (151.6 moves vs ~106 for regressions) — more exploration time
- **Higher policy loss** (3.40 vs 2.4–2.7) — slower convergence
- **More promotions** (4) — actually better model
- Why? Slower convergence → MCTS had time to refine value estimates → better training data → more wins.

**Fix**: Always validate by promotions (real wins) and evaluation play. Training loss is a secondary signal. If loss decreases while promotions drop, you've hit the closed-loop paradox.

**Escalation Tier**: Gotcha → documented in mcts-selfplay.md section "Closed-Loop Training Paradox" with evidence table and intuition. Rule candidate: "For any experiment, require promotions ≥ baseline AND score ≥ baseline. A drop in policy loss alone is not a win."

---

## 2026-04-17: Encoding Asymmetry — Network Learned One-Sided Evaluation

**Date**: 2026-04-17
**Agent**: Batch 1 Autoresearch session (April 17 improvements)
**Domain**: Board representation (encoding)
**Error Type**: Context — fundamental perspective mismatch between encoding and training

**What happened**: Original encoding always placed White in planes 0–5 and Black in 6–11, regardless of whose turn it was. This violated AlphaZero convention (current-player perspective). Result: 85–90% Black-win bias in evaluation. Network learned to evaluate positions asymmetrically — White's pieces were always in the same plane locations, so the network learned "planes 0–5 = my side's chances, 6–11 = opponent's strength." When Black-to-move, this mapping was inverted relative to who was moving, causing systematic evaluation bias.

**Root Cause**: The encoding predated the value-head bootstrap crisis investigation. The original design assumed "White absolute" would be simpler, but didn't account for how the network learns feature maps. A piece in plane 0 (White pawn) is semantically different depending on whose turn it is: my weakness vs opponent's weakness.

**Fix** (commit bb39db6):
- Observation encoding now uses current-player perspective: planes 0–5 always = my pieces, 6–11 = opponent's
- Board is rank-mirrored for Black-to-move so the current player always occupies bottom ranks
- Actions encoded in current-player space, flipped to absolute board space at MCTS boundary via `flip_action()`
- `action_to_notation()` bug fix: no longer appends 'q' to all back-rank moves, only pawn promotions

**Residual issue**: Black-win bias persists even with fixed encoding. Empirically ~50% in fresh runs, but asymmetric when adjudication is enabled. Root cause likely the adjudication passivity trap (see next entry), not the encoding.

**Escalation Tier**: Gotcha → documented in [Board Encoding](board-encoding.md) and [MCTS & Self-Play](mcts-selfplay.md). Code pattern: always flip actions at MCTS boundary when using current-player observations.

---

## 2026-04-17: Adjudication Passivity Trap — Degenerate Play from Inverted Incentives

**Date**: 2026-04-17
**Agent**: Batch 1 Autoresearch session (eval_games.pgn inspection)
**Domain**: Self-play training dynamics (adjudication mechanism)
**Error Type**: Quality — training signal that rewards the wrong behavior

**What happened**: Model converged to degenerate play: e.g., Na3 (knight move) followed by rook shuttle between a1 and b1 for 100+ moves. This pattern persisted across 2026-04-17 session despite encoding fix and all other experimental variations. PGN logs (commit d8aa3c1) revealed the exact pattern: the model finds a "safe" move (one that doesn't lose material), then learns nothing from adjudication (no threshold breach) and material-at-cap (near 0 outcome), so the policy gets stuck.

**Root Cause**: Adjudication mechanism introduced in commit 1846b78 aims to bootstrap the value head by early-declaring winners based on material. However, it only has a **negative signal** ("you will lose because you're down material") and **no positive incentive** ("you should move because passive play loses"). In the early-training regime where value estimates are poor, MCTS explores via Dirichlet noise. Once a passive move is found (e.g., Na3), the policy learns "this is safe" without learning "this is bad." The model then shuttles moves to preserve material without ever learning that passive play loses via checkmate.

**AlphaZero precedent**: AlphaZero never used adjudication. All games play to completion (checkmate, stalemate, or game-length cap). This natural feedback — "if you play passively, you will eventually get checkmated" — prevents the trap. Adjudication removed this signal and replaced it with a narrower one that can be gamed.

**Manifestation across all configurations**:
- Encoding fix (bb39db6): Persists
- Model size increase (0f35653): Persists
- Color augmentation (0a07591): Persists
- All hyperparameter sweeps (2026-04-15 autoresearch): Only manifests in high-adjudication regimes (low material threshold)

**Proposed fix (not yet committed)**:
- Remove adjudication entirely
- Keep material-at-cap as weak bootstrap for 300-move games (outcome = tanh(Δmaterial / 5))
- Keep material-for-draws as weak signal for equal-material positions
- Hypothesis: Games playing to completion (checkmate, stalemate, or cap) will form patterns that punish passivity; value head learns "passive play = loss" through real game outcomes

**Escalation Tier**: Gotcha → documented in [MCTS & Self-Play](mcts-selfplay.md) section "Passivity Trap" with manifestation patterns and AlphaZero precedent. Escalation to Rule once fix is validated.

---

## 2026-04-21: Reward Head Aggregate-Math vs OOD Probe Discrepancy

**Date**: 2026-04-21
**Agent**: Orchestrator session (draws converging, games shuffling)
**Domain**: Neural network diagnostics (reward head training validation)
**Error Type**: Context — aggregate training metrics masked distributional-overfitting failure

**What happened**: Training logs showed reward head predictions alive during batch construction (aggregate `[reward_stats]` lines in baseline logs revealed mathematically-derived predictions ≈±0.99 at late training). However, a probe of the same checkpoint (best_v1489.pt, model_version=15050) on 90 positions from eval_games.pgn (9 terminal + 81 near-terminal) showed reward head outputs in [−0.008, +0.004] for ALL transitions, including actual mating moves. Both `.train()` and `.eval()` modes showed identical behavior; BN running-stats divergence only ~12%, ruling out eval-mode artifact.

**Root Cause**: The reward head overfit to the self-play distribution it was trained on. Aggregate batch stats looked alive because batches come from in-distribution self-play terminals (positions this specific model generated). Probes on real checkmates from other model versions (v5 vs v0, v32 vs v600) showed the head was actually unable to generalize to out-of-distribution terminals.

This is the first concrete evidence of **distributional collapse**: the network converged to a narrow functional range that fits training batches but fails on held-out positions.

**Diagnostic technique** (reusable): Closed-form derivation of batch-aggregate predictions. Given batch statistics (target_std, target_mean, pred_mean, pred_var, MSE), derive the signed magnitude of predictions: `m = S · (1 + B·(pred_var + pred_mean² − MSE)) / 2` where S = sign(target_mean). Test on early, middle, and late slices. Compare with probe on out-of-distribution positions.

**Escalation Tier**: Gotcha — encoded in this entry and in neural-networks.md under "Distributional Overfitting" with diagnostic procedure.

---

## 2026-04-21: Value Head Dead on Canonical Positions Since Step 650 (Unnoticed)

**Date**: 2026-04-21
**Agent**: Orchestrator session (log-history review)
**Domain**: Neural network training validation
**Error Type**: Context — metric was visible in logs but never examined

**What happened**: The trainer's own canonical-position probe (lines 686–699 in `python/hyzero/training/trainer.py`: `[start_value]`, `[kqk_value]`, `[kvk_queenless_value]`) has been logging value-head collapse since the first checkpoint after bootstrap (step 650 onward). Example trajectory for kqk_value (KQ vs K, White to move, trivially won by White):

| Step | kqk_value |
|------|-----------|
| 600 | −0.529 |
| 650 | −0.020 |
| 5,550 | +0.147 |
| 10,550 | −0.003 |
| 15,550 | −0.012 |
| 25,550 | −0.008 |

The value head learned 0 cannot distinguish a trivial mate from a balanced starting position. More critically: this probe runs in `train()` mode (trainer never calls `.eval()` around canonical probes), so this is NOT an eval-mode artifact. The network genuinely failed to learn.

**Root Cause**: Logs were being generated and printed correctly, but nobody was watching them. The probe is a critical early-warning signal. If a value head shows [start_value], [kqk_value], [kvk_queenless_value] staying in [−0.1, +0.1] for >1000 steps without improvement, training is broken.

**Prevention rule**: Any training summary script must report canonical-position probes. Alert if `|kqk_value| < 0.3` and step > 1000 (or adaptive threshold).

**Escalation Tier**: Rule candidate — "Include canonical-probe values in metric summaries; raise alert if value head shows no discrimination on trivial positions for extended periods."

---

## 2026-04-21: BN Eval-Mode Divergence Hypothesis Refuted

**Date**: 2026-04-21
**Agent**: Orchestrator session (diagnostic chain)
**Domain**: Infrastructure (batch normalization behavior)
**Error Type**: Context — incorrect hypothesis from incomplete diagnosis

**What happened**: Earlier suspicion was that the inference server's `.eval()` mode (see `python/hyzero/inference/server.py:137-139`) was causing BN to use stale running-stats while MCTS ran on dead heads, but the training process saw them alive (BN using batch stats). Ran a diagnostic: probed the same checkpoint in both `.train()` and `.eval()` modes on 90 positions. Results were near-identical. Running-stats gap vs batch-stats was < 12%.

**Resolution**: BN eval-mode divergence is NOT the culprit. The heads are genuinely collapsed even in train mode. Inference server is not being sabotaged by batch norm. The real issue is distributional collapse (see entry above).

**Escalation Tier**: Gotcha — documented as a refutation to avoid future false suspicions on BN.

---

## 2026-04-21: Syzygy Tablebase Supervision — Major Outcome (Distributional Collapse Broken)

**Date**: 2026-04-21
**Agent**: Orchestrator + Researcher + Implementer (full session)
**Domain**: Training data pipeline (external supervision + value-head recovery)
**Error Type**: Outcome (hypothesis validation, not a mistake)

**What happened**: Decision to inject Syzygy tablebase supervision (3-4-5-man, WDL+DTZ labels) at 45% fraction with masked padded-step loss and biased value-head reinit (+0.3). Five experimental runs with iterative refinement led to a major breakthrough: first promotion achieved, distributional collapse broken, score improved from 6.05 → 8.16 in a single 2-hour run.

**Iteration Arc**:

1. **Run 1** (unbiased reinit, unbalanced cache, TB_FRAC=0.1): Stochastic outcome. Post-reinit kqk_value = +0.88 (random positive seed), sustained +0.85. First-move diversity recovered to ~8 different openings. Killed at step 21800 by safety gate (suspected excessive TB signal).

2. **Run 2** (balanced cache, unbiased reinit): Negative outcome. Post-reinit kqk_value = −0.51 (random negative seed). Feedback loop: negative reinit → negative self-play targets → further collapse. Killed at step ~15700.

3. **Run 3** (biased reinit +0.3, TB_FRAC=0.2, balanced cache): Partial success. kqk_value peaked +0.54, but killed by gate #2 at step ~18000 with peak reached around +0.28. Showed mechanism works but TB signal still diluting.

4. **Run 4** (infrastructure bug): Loss-masking commit wasn't on main branch. Reverted to exact baseline failure mode. Confirmed infrastructure must be merged to main before launching.

5. **Run 4b** (bug fixed, TB_FRAC=0.3): Clean improvement. kqk_value peak +0.54, no gate fired, but training stalled (killed for other reasons around step 18000).

6. **Run 5** (TB_FRAC=0.45 + masking + biased reinit, PID 1206967): **BREAKTHROUGH**
   - kqk_value: peak +0.85 (sustained for ~3000 steps)
   - **First promotion**: v15283 beat v15051 (win_rate=0.562) in eval ladder
   - **2 actual checkmates**: Appeared in self-play at steps 16650 and 21800 (reward head finally signaling)
   - **Score: 8.1572** (vs pre-TB baseline 6.0468; vs β=0.3 absolute baseline 14.51)
   - **White first-move distribution**: 77% Na3 → spread to ~8 different openings (qualitative improvement from narrow shuffle patterns)
   - **43% decisive self-play rate** on recent games (vs ~1% pre-TB)

**Root Cause of Success**: Mixed TB at 45% fraction with:
- **Masked padded-step loss**: K=5-step TB pseudo-trajectories have targets at step 0 only (±1 for WDL). Masking loss at padded steps 1–4 (target=0) prevented dilution of the ±1 signal.
- **Biased reinit +0.3**: After kaiming_normal_ reset, set value_head output-layer bias to +0.3, deterministically biasing initial responses toward positive (toward actual mate outcomes from TB). Eliminates 50% stochasticity.
- **Balanced TB cache**: Equilibrate +1/-1 samples to prevent drift toward either attractor.
- **High TB fraction (0.45)**: 45% of batch samples from TB means at every gradient step, approximately 45% of positions receive ground-truth supervision. At this proportion, even sparse checkmate arrivals in self-play (1% of steps) receive reinforcement.

**Remaining Failure Mode**: kqk_value oscillates — peaks at +0.85 → drops to −0.34 → back up. Root cause identified: replay buffer dilution. As self-play games accumulate, the effective TB signal proportion shrinks (TB circular buffer stays fixed, replay buffer grows, hence TB fraction = size(TB_buffer) / (size(TB_buffer) + size(selfplay_buffer)) → 0 as training time increases).

**Fix for next session**: Dedicated circular buffer for TB (fixed pool, refreshed from Syzygy cache periodically) instead of mixing at batch-assembly time. This maintains constant 45% proportion throughout the run.

**Evidence from log** (`logs/baseline_20260421_181216.log`, PID 1206967):
```
[start_value]   [kqk_value]   [kvk_queenless_value]   [cm_count]
0.123           +0.85         +0.72                    2 (at step ~20k)
```

No safety-gate kills. Training completed full 2-hour window. Score 8.1572 extracted from final JSON.

**Escalation Tier**: Gotcha → documented in neural-networks.md (new subsection under "Value-Head Failure Modes") and in mcts-selfplay.md (new section "Recovery: External Supervision"). Rules for future TB iterations:
- L1: Reinit of collapsed head is stochastic; screen initial outputs or bias-correct at init (now: bias +0.3)
- L2: Padded-step loss dilutes sparse TB signal; mask loss at padded steps
- L4: Infrastructure commits must be on main before launching experiments
- L5: Maintain constant TB proportion via dedicated circular buffer, not batch-time mixing

**Next Steps**: (1) Implement dedicated TB circular buffer to sustain kqk_value throughout run. (2) Launch longer run (4h) from v15283 champion to explore first promotion momentum. (3) Add opening-book supervision to extend the diverse-first-move pattern beyond the 43% decisive ceiling.

---

## Escalation Tiers

Mistakes escalate from manual avoidance to automation:
1. **Gotcha** (wiki page section) — documented, read-once, agent uses judgment
2. **Rule** (CLAUDE.md / `.claude/rules/`) — loaded into every session context
3. **Hook** (pre-commit/pre-edit) — blocked automatically by tooling

Error types: **context** (wrong/stale info), **breakage** (reintroduced bug), **security** (secrets/injection), **quality** (incomplete logic/validation).

## 2026-04-19: Color Asymmetry from legal_actions Ordering + Argmax Tie-Break

**Date**: 2026-04-19
**Agent**: Orchestrator + Researcher (April 13 session)
**Domain**: MCTS and board encoding (action selection)
**Error Type**: Semantic divergence — code was deterministic but data representation asymmetric

**What happened**: All self-play training runs showed persistent 71–100% Black dominance (7/7 replicate fresh-start runs), with random-evaluator games at ~70% Black wins. Despite identical network weights and symmetric game rules, the color bias was systemic. Investigation traced the bug through a diagnostic chain: Dirichlet sampler (verified uniform), MCTS FP precision (negligible), move generation (symmetric), and finally to the action-selection code path where two separate bugs combined to produce the asymmetry.

**Bug 1 — Argmax Tie-Break to First-Max** (`src/mcts/tree.rs`):
When temperature ≤ ε (deterministic selection), `select_action()` called `max_by` to find the move with the highest MCTS visit count. Ties are common early in training (uniform priors, value ≈ 0, low visit counts). The `max_by` iterator picks the **first encountered** maximum (lowest-index action), deterministically biasing selection. At the starting position with random evaluator, this was the only decision point, and Bug 1 alone didn't explain the color asymmetry — both colors would tie-break identically. But combined with Bug 2 (see below), the two bugs interacted to create the dominant bias.

Fix: Collect all indices with visit count = max, then pick uniformly at random.

**Bug 2 — Color-Asymmetric legal_actions Ordering** (`src/selfplay/game_task.rs`):
`get_legal_moves()` iterates absolute squares (0..64) and collects pieces in that order. White's pieces are at squares 0–15 (knights at 1, 6 come before pawns at 8–15), so White's legal_actions has moves in the order: [Knight moves, pawn moves, castling, ...]. Black's pieces are at squares 48–63 (pawns at 48–55 come before knights at 57, 62), so Black's legal_actions has moves in the order: [pawn moves, knight moves, castling, ...].

After POV-flipping the action coordinates via `flip_action()`, the VALUES are correct (the moves themselves are symmetric), but the POSITIONS in the list remain in absolute-iteration order. So at the starting position, `legal_actions[0]` for White is a knight move, but `legal_actions[0]` for Black is a pawn move.

Combined with Bug 1 (tie-break to index 0), this caused:
- White: select `legal_actions[0]` = Nc3 (knight)
- Black: select `legal_actions[0]` = a6 (pawn)

These aren't POV mirrors of each other. Combined with any MCTS concentration (e.g., Dirichlet noise alpha=0.03, the old bug), this strongly biased one color toward knight development and the other toward kingside edge-pawn pushes. Over hundreds of games, these asymmetric move choices accumulated into self-reinforcing policy patterns, producing the 83% color domination.

Fix: After POV-flipping, sort `legal_actions.sort_unstable()` in both `play_game()` and `play_game_dual()`. Now both colors present identical sorted lists at equivalent positions.

---

## 2026-04-20: Underpromotion Action Spatial Encoding — Color-Blind Under Augmentation

**Date**: 2026-04-20
**Agent**: Autoresearch session (training diagnostics)
**Domain**: Board representation (action encoding under color augmentation)
**Error Type**: Quality — asymmetric behavior under transformation

**What happened**: When `encode_action_spatial(action, white_to_move)` was called with action ≥ 4096 (underpromotion), the function returned rank-specific spatial planes indexed from White's perspective (promotion from rank 6→7). Under color augmentation, flipping the board and calling `encode_action_spatial(flipped_action, white_to_move=False)` returned planes indexed from Black's perspective (promotion from rank 1→0). This violated the representation invariant: `encode(flip(a), flipped_color) ≠ flip_planes(encode(a, original_color))` for all 576 underpromotion actions.

**Root Cause**: Underpromotion indices are color-agnostic at the action ID level (action IDs 4096–4671 are position-specific), but the spatial encoding maps rank indices (6→7, 1→0) using White-centric logic. The function didn't account for color when converting underpromotion action IDs to spatial planes. Non-underpromotion moves (0–4095) were unaffected because they don't involve rank-specific logic.

**Fix** (commit cc58506): Added `encode_action_spatial_for_color(action, white_to_move)` which takes the color context into account. Under color flip, both the action ID and the color parameter flip, now satisfying: `encode_for_color(flip(a), False) == flip_planes(encode_for_color(a, True))`. Added regression test `test_encode_action_spatial_under_color_flip` covering all 576 underpromo cases.

**Escalation Tier**: Gotcha → encoded in [Board Encoding](board-encoding.md) under "Representation Consistency Invariants" with the invariant class and test reference. Rule candidate: "Any representation transform must satisfy flip invariants; add regression test at encoding refactor time."

---

## 2026-04-20: Value-Head Target Attenuation Under Outcome Blending with Sparse Outcome

**Date**: 2026-04-20
**Agent**: Autoresearch session (canonical-position probing)
**Domain**: Neural network training (value target magnitude)
**Error Type**: Context — overlooked consequence of formula when outcome is sparse

**What happened**: With `HYZERO_DISABLE_MATERIAL_SHAPING=0` disabled (no material proxy) and `HYZERO_VALUE_OUTCOME_BETA=0.3`, the value target formula `(1-β) × root_value + β × outcome` produces max magnitude ±0.3 on decisive games (since outcome is ±1 but root_value is untrained ≈0 early in training). Probing canonical positions on best.pt and model_v000069.pt showed all value outputs in [−0.009, +0.035] — a collapsed ±0.035 range across mate-in-1, KQ-vs-K, K-vs-KQ, and starting position. The value head correctly learned its training-data distribution but had zero discriminative power.

**Root Cause**: Value target magnitude is directly set by β when root_value ≈ 0. With β=0.3 and no shaping, the target signal has a ceiling of ±0.3, not ±1. The value head is trained correctly to fit these targets but has no room to express decisiveness across the [−1, 1] output range.

**Fix**: Use conditional β (decisive games use β=1.0, drawn games use β=0.3) to allow full ±1 signal on decisive outcomes, or enable weak material shaping to give drawn games non-zero targets, increasing signal diversity. Commit 18ce8d9 added the conditional-β mechanism. See `HYZERO_CONDITIONAL_BETA` env flag in [MCTS & Self-Play](mcts-selfplay.md).

**Escalation Tier**: Gotcha → encoded in [Neural Networks](neural-networks.md) under "Value-Head Failure Modes" (Mode 2: Shaping OFF + β>0) with diagnosis (per-checkmate-arrival probe) and prevention (conditional β). Rule candidate: "When changing value target formula, verify target magnitudes on canonical positions and confirm they use the full [−1, 1] output range."

---

## 2026-04-20: Sparse-Signal Collapse Under Conditional β Without Gradient Recovery

**Date**: 2026-04-20
**Agent**: Autoresearch session (training log inspection)
**Domain**: Neural network training (multi-head feedback under sparse signal)
**Error Type**: Design insight — mechanism works but signal rate insufficient

**What happened**: Run #4 enabled both conditional β (decisive→β=1.0) and value-head reinitialization, expecting the combination to escape the collapsed-attractor problem. Empirical trace showed value-head outputs jumped +0.35 at checkmate-arrival events, confirming the mechanism responds to decisive signal. However, within 500 training steps, outputs decayed back to ~0. Root cause: only ~5 checkmates appear per 14,452 training steps (~0.035% of samples). The ±1 signal is overwhelmed by 99.5% of drawn-game samples with target=0.

**Root Cause**: Sparse-outcome regime. Conditional β ensures the signal is loud when present (±1), but presence is too rare. The value head correctly learns "checkmate looks like 1.0" but then trains on 500 subsequent drawn-game samples with target=0, and the network quickly converges to the attractor solution 0 (MSE-optimal under 99.5% zeros). The mechanism (conditional β + reinit) works; the application doesn't have enough signal to sustain.

**Fix** (not yet validated): Add weak material shaping (e.g., `HYZERO_MATERIAL_SHAPING_SCALE=20`) so drawn games receive `0.7 × root_value + 0.3 × tanh(Δ/20)`, providing non-zero targets in [−0.3, +0.3] range. This keeps drawn-game training signal non-zero while staying below the 0.5 shuffle-exploit threshold. Decisive games still get ±1 under conditional β. Combined, the value head sees both frequent weak signal (drawn) and rare strong signal (decisive), enabling gradient recovery between decisive arrivals.

**Escalation Tier**: Gotcha → encoded in [Neural Networks](neural-networks.md) under "Value-Head Failure Modes" (Mode 3: Conditional β + Reinit → Sparse-Signal Decay). Future rule (pending validation of the weak-shaping fix): "Under sparse-outcome regimes, ensure non-zero targets on the majority class to avoid sparse-signal collapse."

**Root Cause**: The board encoding was converted to current-player perspective (commit bb39db6), which requires all consumers of `legal_actions` to be POV-aware. However, the action selection code wasn't updated to sort, and the tie-breaking code wasn't updated to be random-tie-aware. An abstraction leak: POV-invariance of the observation planes wasn't extended to the action-list representation.

**Validation**:
- Random-evaluator at 40 sims (N=50 games): Pre-fix 6% W / 70% B / 24% D → Post-fix 40% W / 44% B / 16% D (within noise)
- Regression test `test_legal_actions_ordering_is_color_symmetric_after_sort` added
- All 130 existing tests pass
- Fresh-start training run (post-fix) showed balanced eval results (vs 71–100% B pre-fix)

**Escalation Tier**: Rule — encoded in `.claude/rules/mcts-pov-symmetry.md` to prevent future regressions. Key rule: "When adding code that consumes legal_actions or MCTS visit distributions, verify it works identically for both colors on mirror-equivalent positions. Sorting action lists is required for POV symmetry. Tie-breaking in deterministic selection MUST be random over ties."

---

## 2026-04-21: TB + REINIT Stochastic — Negative Reinit Seed Produces Negative Attractor

**Date**: 2026-04-21
**Agent**: Implementer (4-hour validation run)
**Domain**: Training dynamics (TB supervision + value-head reinit)
**Error Type**: Design insight — stochastic reinit is sensitive to initial seed direction

**What happened**: Two runs with identical config (TB_FRAC, cache, checkpoint):
- Run 1 (short, 155748): Post-reinit kqk_value = +0.16 → climbed to +0.88 → settled +0.43. **Success**.
- Run 2 (long, 163446): Post-reinit kqk_value = -0.08 → briefly reached +0.04 → spiraled to -0.51 at step 15700. **Killed**.

**Root Cause**: `_reinit_value_head()` uses `kaiming_normal_` initialization (random). With ~50% probability, the random weights produce a negative initial response on KQK positions. When this happens:

1. Post-reinit: value head predicts -0.08 for KQK (should be +1)
2. Self-play: root_value ≈ -0.08 → value target = 0.7 × (-0.08) + 0 = -0.056
3. Value head trains on slightly-negative self-play targets
4. MCTS produces worse estimates → new games have more negative Q-values
5. Feedback loop: value head → self-play targets → value head → ... → saturates at -1

With a positive starting seed (+0.16), the reverse happens — TB +1 signal reinforces
the positive direction, and start_value stabilizes near the TB cache mean.

**Balanced Cache Interaction**: Balancing +1/-1 TB samples (to fix start_value drift)
removed the slight positive bias that helped Run 1 sustain positive kqk values. With
perfectly balanced TB, the average TB signal is 0 — so TB pushes toward 0, not +1.
Combined with slightly-negative self-play targets, the balanced cache slightly worsens
the negative attractor problem.

**Fix Options**:
1. **Screen reinit**: After reinit, check kqk_value. If < 0, reinit again. Repeat up to 5x.
2. **Biased output init**: Set `value_head[-1].bias = +0.1` after kaiming_normal_ reinit. Guarantees kqk starts positive.
3. **Accept stochasticity**: Run multiple short 5-min runs, pick the one where kqk went positive, then switch to long run.

**Escalation Tier**: Gotcha → documented in agent-memory/orchestrator/tablebase_4h_validation.md with full trajectory data.

---

