# Project Roadmap

Current state and next steps for hyzero.

## What's Done

- **Tasks 1-16**: Chess engine — bitboards, magic move gen, all special moves, validation
- **Tasks 17-23**: MCTS tree search, self-play pipeline, inference batching, replay buffer
- **Tasks 24-26**: Python MuZero models (h/g/f networks) with training loop and inference server
- **Task 27**: MCTS value negation fix (two-player zero-sum backpropagation)
- **Task 28**: PyO3 integration — full Rust↔Python bridge
- **Task 29**: E2E validation — 5 games, 13 train steps, loss 8.52→7.04
- **Task 30**: Engine foundation — Zobrist hashing, FEN parser, perft validation, pin fixes
- **Task 31**: Engine validation — perft CLI, cross-validation vs python-chess (53M nodes), 3 stalemate/draw bugs fixed
- **Task 32**: Training infrastructure — coordinator simplification (no semaphore), checkpoint rolling window, resume-from-checkpoint, evaluation task, config consolidation with env vars

**Current state**: 89 Rust tests (82 pass, 7 ignored), zero clippy warnings. Pipeline validated with 30-min baseline run (112 games, 444 gradient steps, loss 8.55→3.23).

## Baseline

- **Score**: 6.78 (formula: `(8.55 - policy_loss) + (decisive_ratio * 10) - (avg_length / 100)`)
- **Run command**: `bash scripts/run_baseline.sh 1800`
- **Stored at**: `logs/baseline_score.json`
- **Commit**: d407281 (2026-04-14 — Dirichlet alpha fix: 0.03 → 0.3)

## What's Next

Detailed roadmap with file lists and rationale: [`docs/plans/next-steps/roadmap.md`](../plans/next-steps/roadmap.md)

| Batch | Focus | Key changes | Status |
|-------|-------|-------------|--------|
| — | Bug fix | Dirichlet noise alpha: 0.03 (Go) → 0.3 (chess) per AlphaZero paper | DONE (+2.65, 6.78) |
| 1 | Representation overhaul | History planes (19→103), underpromotion (4096→4672), legal move masking | DONE (superseded by alpha fix) |
| 2 | Search improvements | Tree reuse (DISCARDED: Q-value warm-start regressed), MCTS policy masking, temperature decay | — |
| 3 | Training hardening | LR scheduling, loss rebalance, ~~priority replay~~ (DISCARDED: recency decay caused catastrophic forgetting), train ratio | — |
| 4 | Tactical metric | Puzzle suite, strength evaluator, composite score update | — |
| 5 | UCI protocol | Playable engine, time control, Stockfish benchmarking | — |

## Known Risks

- **Dirichlet noise CPU overhead**: Must use `--release` for e2e/baseline tests
- **Game length ~200 moves**: Correct for exploration but impacts iteration speed
- **Batch timeout tuning**: 10ms empirical, may need adjustment at higher concurrency
- **GIL contention**: One acquisition per batch; monitor if eval + self-play conflict
- **Eval cycle cost**: 10 eval games at 50 sims takes ~3-4 min, may skip threshold crossings
- **Loss weight amplification destabilizes**: value_loss_weight=5.0 test regressed score 11.63→4.84 (2026-04-15). Do not retry above 2.0; prefer tuning outcome blend β instead

## Experimental Results

| Exp | Config | Baseline | Result | Delta | Note |
|-----|--------|----------|--------|-------|------|
| alpha-fix | Dirichlet α: 0.03→0.3 (chess, not Go) | 4.13 | 6.78 | +2.65 | Foundational fix, permanent |
| batch1-rep | History 19→103, underpromotion, masking | 6.78 | (superseded) | — | Nullified by alpha fix timing |
| value-weight=5.0 | Amplify value loss 5x at β=0.3 | 11.63 | 4.84 | −6.79 | Closed-loop instability; do not retry |

## Metric Evolution

**Current metric** (`training_score` formula) uses self-play decisive ratio as a signal. **Problem**: As model improves, self-play-vs-self converges to draws (identical play by both sides). Three autoresearch runs show policy_loss improving while decisive_ratio drops to 0 — metric optimizes toward model weakness.

**Measurement noise** (2026-04-14): Baseline exhibits ±1 point variance from eval running only 10 games (binomial variance ±0.15-0.20 per cycle) and training step count varying ±50% between runs. Single-run claims <1.5 points are within noise; marginal changes require multi-run validation.

**Future work** (Phase 4 priority):
- **Phase 4 infra** (multi-run averaging): Each experiment runs 3x, median reported to reduce noise
- Decouple metric components: track policy_loss and avg_game_length separately during development
- Replace self-play decisive_ratio with win rate vs **fixed reference opponent** (`RandomEvaluator` from `src/selfplay/evaluation.rs`)
- Phase 4 will add puzzle-solving suite and composite strength score

This change prevents metric-gaming, reduces noise, and directly measures progress toward a strong, consistent engine.

## Related
- [Neural Networks](neural-networks.md) — model architecture
- [MCTS & Self-Play](mcts-selfplay.md) — pipeline architecture
- [Rust-Python Integration](rust-python-integration.md) — FFI boundary
- [Development Roadmap](../plans/next-steps/roadmap.md) — detailed next steps
