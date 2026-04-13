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

- **Score**: 4.78 (formula: `(8.55 - policy_loss) + (decisive_ratio * 10) - (avg_length / 100)`)
- **Run command**: `bash scripts/run_baseline.sh 1800`
- **Stored at**: `logs/baseline_score.json`
- **Commit**: c1e5cdc (2026-04-13)

## What's Next

Detailed roadmap with file lists and rationale: [`docs/plans/next-steps/roadmap.md`](../plans/next-steps/roadmap.md)

| Batch | Focus | Key changes |
|-------|-------|-------------|
| 1 | Representation overhaul | History planes (19→103), underpromotion (4096→4672), legal move masking |
| 2 | Search improvements | Tree reuse, MCTS policy masking, temperature decay |
| 3 | Training hardening | LR scheduling, loss rebalance, priority replay, train ratio |
| 4 | Tactical metric | Puzzle suite, strength evaluator, composite score update |
| 5 | UCI protocol | Playable engine, time control, Stockfish benchmarking |

## Known Risks

- **Dirichlet noise CPU overhead**: Must use `--release` for e2e/baseline tests
- **Game length ~200 moves**: Correct for exploration but impacts iteration speed
- **Batch timeout tuning**: 10ms empirical, may need adjustment at higher concurrency
- **GIL contention**: One acquisition per batch; monitor if eval + self-play conflict
- **Eval cycle cost**: 10 eval games at 50 sims takes ~3-4 min, may skip threshold crossings

## Related
- [Neural Networks](neural-networks.md) — model architecture
- [MCTS & Self-Play](mcts-selfplay.md) — pipeline architecture
- [Rust-Python Integration](rust-python-integration.md) — FFI boundary
- [Development Roadmap](../plans/next-steps/roadmap.md) — detailed next steps
