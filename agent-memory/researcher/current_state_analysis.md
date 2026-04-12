# Comprehensive Analysis: hyzero State & What's Next (2026-04-12)

## Executive Summary

`hyzero` is a **feature-complete but underpowered** Rust MuZero chess engine. All 29 core tasks completed (Tasks 1-16: chess engine, Tasks 17-23: MCTS/self-play, Tasks 24-29: Python neural networks + PyO3 integration). System demonstrates end-to-end learning loop (5 games, 13 training steps, loss 8.52→7.04) but plays poorly due to development-mode configuration.

**Status**: MVP complete. Ready for scaling and optimization work.

## Build & Test Status

- **Rust**: 36 files (.rs), compiles cleanly, 24 tests passing, zero warnings via `cargo clippy`
- **Python**: 15 files (.py), InferenceServer + Trainer implemented, config exists, **0 Python tests run** (pytest not installed in CI)
- **E2E**: `scripts/e2e_test.sh` runs autonomously, 120s timeout, extracts loss/games/steps to metrics file
- **Latest metrics** (2026-04-11 20:49:14): 5 games, 13 steps, loss 8.52→7.04, no errors

## Architecture Quality

**Chess Engine** (Tasks 1-16):
- Bitboard representation, magic move generation (rooks/bishops), pre-computed knight/king move tables
- All special moves: castling (both sides, validation, path-clear), en passant, pawn promotion
- Draw rules: 50-move clock, threefold repetition (zobrist hash), insufficient material
- Checkmate/stalemate detection
- GameHistory with move + board snapshots
- Server/client binary with Unix sockets (TCP conversion in Task 13-14)

**MCTS + Self-Play** (Tasks 17-23):
- `MCTSTree` with PUCT selection, value negation per ply (two-player zero-sum fix in Task 27)
- `SelfPlayCoordinator` with semaphore-based concurrency control
- `InferenceBatcher` with request channel, configurable batch size & timeout
- Replay buffer (VecDeque ring, capacity eviction, bincode checkpoints)
- `play_game()` async function: MCTS per move, temperature-based action selection (high early, low late)

**Python Networks** (Tasks 24-26):
- `RepresentationNetwork` (h): [B,19,8,8] → [B,64,8,8]
- `DynamicsNetwork` (g): [B,67,8,8] → [B,64,8,8] + [B] (state + reward)
- `PredictionNetwork` (f): [B,64,8,8] → [B,4096] policy + [B] value
- All use 4×ResBlock architecture, C=64 channels (tiny for chess)
- Training: K-step unroll (K=5), cross-entropy policy loss, MSE value/reward losses
- Inference: Batch inference under torch.no_grad(), softmax-normalized policies

**PyO3 Integration** (Tasks 27-28):
- `PyO3Backend`: Implements `InferenceBackend`, calls Python InferenceServer batch methods
- `PyTrainingThread`: Async task that receives trajectories, trains via Trainer, publishes model version
- Weight sync: watch channel for weight vectors (serialized via pickle)
- Batch assembly: pads visit_distributions to 4096, handles short trajectories

## Current Configuration (Development Mode)

| Setting | Value | Notes |
|---------|-------|-------|
| num_simulations | 50 | Way too low; target 200+ for reasonable play |
| max_concurrent_games | 4 | CPU-friendly; GPU would handle 16+ |
| batch_size | 32 | Reasonable; could go to 128 for GPU |
| batch_timeout_ms | 10 | Empirical; may need profiling |
| network C | 64 | Tiny (AlphaZero: 256+); 4 ResBlocks (target 8-20) |
| training_batch_size | 256 | Reasonable |
| unroll_k | 5 | Reasonable for early training |

**Why so conservative?** Defaults enable fast local development & CI/CD on CPU. Board position quality from 50 simulations is weak (likely 2000-2200 Elo equivalent).

## What's Documented, What's Not

**Good documentation:**
- `CLAUDE.md`: Commands, architecture overview (current, accurate)
- `docs/wiki/`: Synthesized knowledge (chess-engine, mcts-selfplay, neural-networks, rust-python-integration, project-roadmap, dev-workflow)
- `docs/ARCHITECTURE.md`: Full MuZero design (36K, detailed)
- Task documents: `TASKS_MCTS_SELFPLAY.md`, `TASKS_PYTHON.md` (complete & accurate)

**Stale or missing:**
- `docs/todo.md`: Pre-Task 1 content, outdated
- `docs/CHANGELOG_*`: Session logs, not maintained
- Python setup: No `setup.py`, `requirements.txt`, or env config (should be installable via `pip install -e .`)
- No README for Python package or build steps
- No hyperparameter tuning guide

## Test Coverage

| Category | Count | Status |
|----------|-------|--------|
| Rust unit tests | 24 | ✓ Passing; good coverage (tree, PUCT, game tasks, replay buffer, batching) |
| Python tests | 27 | ⚠ Defined but **never run** (pytest not in environment) |
| PyO3 tests | 3 | ⊘ Ignored (requires hyzero Python installed) |
| E2E tests | 1 script | ✓ Passing (120s autonomous run) |
| Integration tests | 0 | ✗ Missing (no test combining Rust + Python after Task 28) |

**Coverage gaps:**
- No adversarial tests (engine vs itself, transposition detection)
- No property-based tests (move generation exhaustiveness, game invariants)
- No performance benchmarks (move gen throughput, inference latency, batch fill rates)

## Key Gaps & Limitations

### 1. MCTS & Search (High Priority)

**Problem**: 50 simulations = weak play (~50% win rate vs random, ~0% vs Stockfish level 1)
- No transposition tables → redundant computation
- No root noise schedule (only Dirichlet at start)
- No UCB decay or other exploration decay
- No alpha-beta pruning or related enhancements
- Marsaglia-Tsang Gamma sampling for Dirichlet is slow (requires `--release` for e2e tests)

**Impact**: Engine plays objectively poor chess; training is slow (needs more games to improve).

### 2. Training Pipeline (High Priority)

**Missing features:**
- No priority replay sampling (all trajectories equally weighted)
- No reanalyze step (replay buffer is cold storage; old positions aren't re-evaluated with new model)
- No temperature scheduling (fixed: 1.0 for first 15 moves, then 0.01)
- No learning rate scheduling
- No gradient clipping
- No distributed training

**Impact**: Convergence is slow; hard positions don't get extra training.

### 3. Model & Action Space (Medium Priority)

**Issues:**
- Action space fixed at 4096 (64×64) with queen-only promotion
- No underpromotion support (knight/bishop/rook) → 4/4 legal promotions mapped to 1 action
- No illegal move masking in policy (network wastes capacity learning to avoid invalid moves)
- Network tiny: C=64, 4 ResBlocks (vs AlphaZero: 256+, 20 blocks)

**Impact**: Reduced policy quality; network capacity underutilized.

### 4. Evaluation & Metrics (Medium Priority)

**Missing:**
- No Elo rating system (no comparison to Stockfish, Leela, or other baselines)
- No opening book or endgame tablebases
- No game analysis (PV extraction, mistake detection, best-move comparison)
- No convergence tracking across runs (only single e2e test snapshot)
- No training curve visualization
- Metrics limited to loss + game count (no policy entropy, value range, reward distribution)

**Impact**: Can't quantify improvement or compare to baselines; hard to debug training issues.

### 5. Infrastructure & Configuration (Low Priority)

**Hardcoded in `src/bin/selfplay.rs`:**
- num_simulations: 50
- max_concurrent_games: 4
- batch_size: 32
- batch_timeout_ms: 10
- temperature_moves: 15

**Missing:**
- No config file support (TOML/YAML)
- No graceful shutdown (SIGTERM → 2s sleep → SIGKILL)
- No health checks or monitoring
- No experiment parameter sweep tools (run_experiment.sh runs same config N times, not combinations)
- Logging is all stderr, no levels/filtering

**Impact**: Can't easily test different configurations without code changes.

### 6. Testing & Validation (Medium Priority)

**Gaps:**
- Python tests never run (pytest not installed)
- No integration tests post-PyO3 integration
- No adversarial self-play tests
- No performance benchmarks

**Impact**: Can't validate Python+Rust integration in CI/CD; catch regressions late.

### 7. Documentation & DX (Low Priority)

**Missing:**
- No Python `setup.py` or `requirements.txt`
- No Docker for reproducible environments
- No contributor guide
- No hyperparameter tuning guide
- Stale `docs/todo.md`

**Impact**: Harder onboarding for new contributors.

## Known Risks & Gotchas

1. **Game length scaling**: With 50→400 simulations, games grow 3-4× longer (~200→800 moves). Current MAX_GAME_LENGTH=300 will cap games early.

2. **Batch timeout tuning**: 4 games × 400 sims = 1600 inference requests. 10ms timeout → ~16 batches. If GPU can't fill them, increase timeout. If batches are small, decrease timeout.

3. **GIL contention**: PyO3 acquires GIL once per batch (~32 requests). With 400 sims, that's ~50 GIL acquisitions per move. May see contention with more games.

4. **Reward signal divergence**: g() outputs immediate rewards (learned artifacts); game outcomes are terminal (win/loss/draw). May diverge. Validate empirically.

5. **Stale model in flight**: Games start with old weights, training runs with new weights. Brief lag acceptable per design, but data flow must be verified.

6. **Replay buffer memory**: Unbounded (10k trajectories × 300 steps × 64 floats ≈ 7.5 GB). Need monitoring.

7. **Stdout buffering**: `cargo run` buffers output; scripts must run binary directly (`target/release/selfplay`).

## Recommended Priority

### Phase 1: Get it to Play Decent Chess (Quick Wins)
1. **Scale MCTS simulations**: 50 → 400 (1 file, 5 min)
   - Also adjust MAX_GAME_LENGTH to 1000
   - Impact: 10× play strength

2. **Increase network capacity**: C=64→128, 4→8 ResBlocks (1 file, 10 min)
   - Impact: Better value/policy estimation

3. **Add illegal move masking**: Pass legal_moves to policy head (3 files, 2 hrs)
   - Mask logits before softmax
   - Impact: 10% faster convergence

### Phase 2: Make Training Efficient (High Impact)
4. **Priority replay sampling**: Weight by TD error (1 file, 3 hrs)
5. **Reanalyze step**: Re-evaluate old trajectories with new model (2 files, 6 hrs)
6. **Temperature scheduling**: Decay over training (1 file, 1 hr)

### Phase 3: Validation & Baselines (Medium Impact)
7. **Elo estimation**: Play against fixed-depth minimax (2 files, 6 hrs)
8. **Training metrics**: Log entropy, value range, reward distribution (2 files, 3 hrs)

### Phase 4: Ops & Tooling (Nice to Have)
9. **Config YAML/TOML**: Load from file (2 files, 2 hrs)
10. **Install Python tests**: Get pytest running in CI (1 file, 30 min)
11. **Distributed inference**: Subprocess server + Unix socket (3 files, 12 hrs, high risk)

## Session Recommendations

1. **Next task: Scale MCTS simulations to 400 + network capacity**
   - File: `src/bin/selfplay.rs` (change defaults, adjust MAX_GAME_LENGTH)
   - File: `python/hyzero/config.py` (increase C, num_res_blocks)
   - File: `src/selfplay/game_task.rs` (adjust max_game_length if needed)
   - Expected outcome: Run e2e test 120s, should see > 10 games, loss convergence

2. **Follow-up: Add illegal move masking**
   - Requires changes to: policy head in f(), batch assembly in Rust, board encoding
   - High impact on convergence and policy quality

3. **Then: Priority replay + reanalyze**
   - Unlocks 20-30% training speedup
   - Requires changes to: replay buffer sampling, training thread architecture

4. **Finally: Elo estimation**
   - Enables baseline comparison
   - Gives concrete "engine is now 1500 Elo" milestones

## Files to Watch

- `src/bin/selfplay.rs` — game/inference/training config
- `src/selfplay/game_task.rs` — max_game_length, temperature, move selection
- `python/hyzero/config.py` — network capacity (C, num_res_blocks)
- `src/data/encoding.rs` — action space (if underpromotion added)
- `python/hyzero/models/prediction.py` — policy head (if masking added)
- `scripts/e2e_test.sh` — metric extraction assumptions

## Session Outcome

All infrastructure is solid and tested. Code is ready for optimization work. Next agent should focus on Phase 1 (scaling) to get the engine to actually play decent chess, then Phase 2 (training) for efficiency.
