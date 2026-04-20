# MCTS & Self-Play Infrastructure

## Closed-Loop Training Paradox: Faster Training ≠ Better Model

**CRITICAL**: MCTS self-play is a closed loop. If training speed increases (lower loss) without MCTS quality improvements, the model trains on garbage targets. Policy loss improvements become a false positive: the network memorizes poor visit-distribution targets. Signature: policy loss decreases while promotions drop to zero.

**Evidence**: 11-experiment β sweep (2026-04-15) — all configs with lower loss (2.4–2.7 vs 3.4 baseline) regressed in promotions and score. β=0.3 winner had *higher* loss (3.40) but *longer games* (151.6 moves), more exploration, better value estimates, and *actual wins*.

**Key Decision**: Validate by promotions (real wins), not training loss. See `docs/wiki/mistakes.md` (2026-04-15 entry) for full table and analysis.

## Selection Mechanics (Tie-breaking & POV Symmetry)

See [MCTS Action Selection Mechanics](selection-mechanics.md) for detailed analysis of two critical bugs (commit 41f6681) that produced 83% Black dominance. Key fix: sort `legal_actions.sort_unstable()` after POV-flipping, and random-break ties in argmax.


## MCTS Tree Search (Per Move)

**Per-move flow**:
1. Encode board → `BoardObservation` (103 planes: current position + 7 historical positions × 12 piece planes + 7 auxiliary channels)
2. Call `evaluator.root_setup(observation)` → (hidden_state, policy, value)
3. Run N simulations (50 for CPU dev, 200+ for GPU production)
4. Extract visit count distribution, select action by visits (temperature scheduling)
5. Apply move, record StepRecord, **discard entire tree** (transient)

## Single MCTS Simulation

1. **SELECT**: Walk down tree using PUCT = Q(s,a) + P(s,a) * sqrt(N(s)) / (1 + N(s,a))
2. **EXPAND**: Call g(s, a) → (s_new, reward) in latent space
3. **EVALUATE**: Call f(s_new) → (policy, value) to initialize children
4. **BACKUP**: Propagate value to root with negation per ply, increment visit counts (two-player zero-sum)

## Inference Batching

Game threads send `InferenceRequest` (RootSetup or ExpandLeaf) to batcher. Batcher collects up to `batch_size` requests or times out (T_timeout), makes single PyO3 call to Python, distributes results via oneshot channels. Reduces GIL acquisitions from ~800/move to ~1/move.

## Replay Buffer

Ring buffer (`VecDeque<GameTrajectory>`) with K-step sampling: pick random trajectory (weighted by length), pick random index t where t+K ≤ len, return steps t..=t+K. Each trajectory tagged with model_version. Serialized with bincode for checkpoints.

## Self-Play Coordinator

Spawns N persistent long-lived tokio game loop tasks (no semaphore gating). Each task loops indefinitely: read current `model_version` from watch channel → play one game → send GameTrajectory → repeat. Each game: root setup → MCTS → move selection → step record, repeating until terminal, adjudication, or 300-move limit. GameTrajectory tagged with model_version and outcome. Awaits all JoinHandles on shutdown.

**Design**: No semaphore — tasks are persistent, not spawned per game. This reduces overhead and scales cleanly to many concurrent games. Watch channel updates when trainer publishes new weights, ensuring game tasks always use current model.

### Game Termination Paths

Three mechanisms end a game (checked in order in `play_game()` loop):

1. **Terminal state** (`GameResult::Checkmate`, `GameResult::Stalemate`): Write true game outcome (±1 or 0).

2. **Adjudication** (NEW in commit 1846b78): If `|Δmaterial| ≥ HYZERO_ADJ_THRESHOLD` (default 6) sustained for `HYZERO_ADJ_PLIES` (default 10) consecutive plies, declare winner by material dominance. Write `outcome = sign(Δmaterial)`. Counter resets if material diff drops below threshold (e.g., capture narrows gap). Env vars allow threshold tuning for smoke tests without rebuild.

3. **Material-at-cap** (NEW in commit 1846b78): Game hits 300-move limit without terminal or adjudication. Instead of synthetic `outcome = 0`, write `outcome = tanh(Δmaterial / 5.0)`, where Δmaterial = white_material − black_material (piece values: P=1, N=3, B=3, R=5, Q=9). Preserves White-absolute sign; trainer applies ply-flip at batch assembly time (`src/py/training.rs:136`). This breaks the zero-target bootstrap loop that killed the value head.

**Effect on average game length**: With default adjudication at 6 points (roughly ±2 pawns from equal), random play adjudicates at ~40 moves. Stronger play lasts longer (more balanced positions). As value head learns material, adjudication rate naturally decreases and games converge to true terminal outcomes.

### Root Noise for Exploration

Dirichlet noise added to root policy before move selection. Implemented via Marsaglia-Tsang Gamma sampling (MuZero paper).
- **NOISE_ALPHA = 0.3** (AlphaZero chess value — spreads exploration across many moves)
- **NOISE_EPSILON = 0.25** (fraction of noise mixed into prior)

**Game-specific constants**: AlphaZero paper specifies α={0.3, 0.15, 0.03} for {chess, shogi, Go}. Using the wrong value (e.g., 0.03 for chess) over-concentrates noise on 1-2 random moves, starving exploration of the state space (see 2026-04-14 mistakes log: Dirichlet alpha bug).

**WARNING**: Slow in debug mode; use `--release` builds for end-to-end testing.

## Checkpoint Management

`PyTrainingThread` manages rolling window of saved checkpoints with zero-padded filenames (`model_v000050.pt`). Every `checkpoint_interval_steps` training steps (default 50), saves weights to disk. Maintains list of saved files in `VecDeque<PathBuf>`, prunes oldest when window size exceeds `checkpoint_keep_last` (default 5).

### Resume from Checkpoint

`from_default_config(resume_checkpoint: Option<&str>)` loads a prior checkpoint. PyO3 call loads weights into trainer, reads `model_version` from trainer object attribute, returns both. Weights pushed to InferenceServer, `model_version` published via watch channel. All inference and game tasks resume from the loaded model state.

## Evaluation Task

Separate async task watches `model_version` via watch channel. When version advances by `eval_interval_steps` (default 200), spawns `eval_games` (default 10) self-play games using the current model playing against itself (ChannelEvaluator, not random). After all games complete, logs statistics:
- `white_wins`, `black_wins`, `draws`
- `white_win_rate` (white_wins / total)
- `decisive_ratio` ((white_wins + black_wins) / total)
- `avg_length` (mean game length)

**PGN Logging** (commit d8aa3c1): Eval games append PGN-formatted moves to `logs/eval_games.pgn` with headers (event, white, black, result) and numbered move pairs. Critical debugging tool — allows inspection of what the model is actually playing during eval cycles. Revealed 2026-04-17 session's passivity trap (rook shuffle patterns).

Provides continuous signal of model quality during training. Runs in parallel with main self-play loop.

## Action Encoding

Action space: 4672 (AlphaZero 8×8×73 encoding: 64 origin squares × 64 destination squares + 64 underpromotion slots). Encoding: `action = from_sq * 64 + to_sq` for regular moves; underpromotion offsets at 4096-4671. Network learns to suppress illegal moves via legal-move masking (before softmax, with `nan_to_num` handling).

## Replay Buffer Distribution Dynamics

**Equal-weight sampling intuition**: Early random games (weak model) contribute the same gradient updates as recent on-policy games (strong model). Intuitively wasteful — why train on old noise?

**Recency weighting tradeoff**: Exponential decay (weight ∝ exp(-decay × version_age)) prioritizes recent batches. However, this **starves the value head of outcome diversity**. Early random games have wide outcome distributions (draws, decisive wins, losses); recent games from a converging policy narrow toward repetitive patterns. The value head needs signal from diverse positions and outcomes to learn stable value estimates.

**Catastrophic forgetting signature**: Commit `003eaf9` ran with `HYZERO_REPLAY_DECAY=0.1` and observed:
- v20 eval: decisive_ratio = 0.50 (good)
- v57 eval: decisive_ratio = 0.10 (collapsed)
- Policy loss kept improving (3.96→3.02) but tactical play regressed

This divergence signals value-head collapse: policy learned to avoid loss but value estimates became unreliable, leading to drawish play. Second run with decay=0.05 showed same pattern (v20 decisive=0.20, v61 decisive=0.10).

**Future options**: (a) Separate value/policy samplers with different decay schedules (policy favors recent; value preserves old diversity); (b) Add diversity floor (minimum 10-20% uniform sampling to force value head to see wide outcomes); (c) Prioritize by outcome (high-variance games preferred over recent repetitive ones).

## Dead Value & Reward Heads — Historical Context

**Value head bootstrap failure** (pre-1846b78): Self-referential bootstrap. Target is MCTS root_value (untrained), initialized from f-network output and backed up from leaf. Loop: `f(s) ≈ 0` → `root_value ≈ 0` → target ≈ 0` → loss ≈ 0 → `f` stays 0. Canonical MuZero uses outcome targets; our MCTS Q-estimate approach lost this signal. With Q ≈ 0, MCTS search reduced to prior sampling — policy self-imitating with no improvement. Manifested as 99% games hitting 300-move cap with `outcome = 0`, perpetuating the cycle.

**Fixed (commit 1846b78)**: Material-at-cap + adjudication inject outcome-like signals (material proxy) to break the loop. See "Game Termination Paths" above.

**Reward head dead** (training logs: `reward=0.0006`): Class imbalance — only terminal steps have non-zero targets. For 100-move games, terminal appears in ~1% of K+1-step slices. MSE-optimal solution is 0 everywhere. MuZero needs reward head for latent-space terminal detection; dead reward breaks backup signal and MCTS may expand past terminal states. Env var `HYZERO_REWARD_OUTCOME_GAMMA` allows soft outcome-blend (similar to β for value head) if future experiments need reward head rescue.

See neural-networks.md sections "Canonical MuZero Value Target" and "MCTS as Policy Improvement" for architecture discussion.

## Passivity Trap — Adjudication Creates Degenerate Training Signal (2026-04-17)

**CRITICAL FINDING**: Adjudication mechanism (commit 1846b78) introduces a fundamental training signal inversion. The mechanism says "if you lose material, you lose" but never says "if you don't move, you lose." Result: the model learns to avoid losing material by *not moving*, converging to degenerate play (Na3 + rook shuffle a1↔b1) that stalemates itself.

**Manifestation** (observed in eval_games.pgn from 2026-04-17 session):
- Games get stuck in patterns: e.g., Na3, then rook shuttles between a1 and b1 for 100+ moves
- Eval terminates by 300-move cap, not by checkmate or adjudication
- Score improves on "not losing" (material-for-draws at cap) but actual play quality is unsalvageable

**Root cause**: Adjudication only considers material threshold, not move frequency/repetition. Early in training with poor value estimates, MCTS explores via Dirichlet noise. Once a safe material-preserving move is found (e.g., Na3), the policy pins to it because:
1. Value estimate says "don't move, that's risky"
2. Adjudication never fires (only 3 pawns on board, well below threshold)
3. Material proxy at cap gives ~0.0 outcome (no gradient)
4. Network learns "this move is safe" without learning "it's also useless"

Passive play persists across ALL configs tested (encoding fix, model size, hyperparameters).

**AlphaZero precedent**: AlphaZero never used adjudication. Games played to completion (checkmate, stalemate, or game-length cap). This forces the value head to learn that passive play eventually loses via checkmate. Adjudication short-circuits this signal — you can be passive forever as long as you don't lose material.

**Proposed fix (not yet committed)**:
- Remove adjudication entirely
- Keep material-at-cap for games reaching 300 moves
- Keep material-for-draws as weak bootstrap signal
- Accept slower early training (material signal initially weak)
- Hypothesis: Games playing to completion will form checkmate patterns that punish passivity, breaking the trap

## Dual-Model Ladder Stall — Symmetry Collapse

After a promotion to v1 (new challenger = champion snapshot), both models have identical architecture + weights at t=0. They drift together due to simultaneous training on the same replay buffer. Eval often returns **win_rate ≈ 0.50** for many cycles, delaying the next promotion. This is **expected behavior** (symmetric play → draws) but slows improvement.

**Mitigations observed** (2026-04-15):
- **β=0.1** (soft outcome blend): 1 promotion in 30 min
- **β=0.2**: 2 promotions in 30 min

**Note**: This remains an unsolved problem for longer runs (>1 hour). Future work needed on breaking symmetry inside a single training run (e.g., adversarial ensemble, asymmetric eval). See `docs/plans/next-steps/resume.md` for proposed solutions.

## Key Gotchas

1. **Value negation**: Negated per ply during backprop (two-player zero-sum). See `MCTSTree::backpropagate()`: `child.total_value += -value`.
2. **Dirichlet noise overhead**: Marsaglia-Tsang Gamma sampling for Dir(0.03) is very slow in debug mode. For e2e testing, always use `--release` builds or run the binary directly (not via `cargo run`).
3. **Game length with Dirichlet**: Games run ~200 moves now (vs ~60 before); correct behavior for better play, but impacts iteration speed.
4. **Transient tree**: Discarded after each move. Fresh tree per move, no caching.
5. **Batch timeout tuning**: 10ms timeout is empirical. Few concurrent games → small batches → lower GPU utilization.
6. **Stale model data**: Old trajectories in buffer. Loss initially high (bootstrapping).
7. **Action space mismatch**: 4672 logits vs ~40 legal moves. Legal-move masking applied before softmax + nan_to_num.
8. **Visit distribution sparsity**: replay buffer stores StepRecord with visit counts for each move. Array is sparse (length = num_visits). PyTrainingThread zero-pads to 4672 before passing to trainer.
9. **Stdout buffering in scripts**: `cargo run` buffers output. For log capture in shell scripts, run the binary directly: `target/release/selfplay` instead of `cargo run --bin selfplay`.
10. **action_to_move signature**: `action_to_move(action, board, color)` requires board state and active color to correctly reconstruct castling and en passant moves. The selfplay game_task.rs already handles this correctly; only direct callers of action_to_move need updating.
11. **Legal-move masking NaN**: `log_softmax(-inf)` produces NaN in log-probs. Use `nan_to_num(neginf=0.0)` after softmax to replace illegal-move NaNs with 0.
12. **Game outcome perspective**: `game_outcome` is absolute White-perspective (+1 for White win, -1 for Black win), but observation planes encode absolute piece positions. When using outcome as a value target, account for whose turn it is (plane 101).

## Related

- [Board Encoding](board-encoding.md) — current-player perspective, action flipping at MCTS boundary
- [Neural Networks](neural-networks.md) — value/reward training, loss weighting
- `docs/wiki/mistakes.md` — adjudication passivity trap, encoding asymmetry fixes

## Related Files

- `src/mcts/{mod,node,tree,puct,evaluator}.rs` — tree operations, PUCT selection
- `src/selfplay/{mod,game_task,coordinator,inference,training}.rs` — orchestration
- `src/data/{types,encoding,replay_buffer}.rs` — board observation, action space, buffer
- `docs/TASKS_MCTS_SELFPLAY.md` — detailed task specs (Tasks 17-23)
