# Project Wiki

hyzero is a chess engine that emulates MuZero: a Rust core (bitboard move
generation, MCTS, self-play coordinator, evaluation ladder) bridged to a PyTorch
neural-network layer (h/g/f models, trainer, inference server) over PyO3. This
wiki is the knowledge base, synthesized from the code and verified against the
current source tree.

## Engine Core
- [Chess Engine](chess-engine.md) — bitboards, magic move generation, perft, special moves, draw rules
- [Board Encoding](board-encoding.md) — 102-plane observation tensor, AlphaZero perspective, action flipping

## Learning Pipeline
- [MCTS](mcts.md) — PUCT tree search, Gumbel-Top-k, selection mechanics, color-symmetry caveats
- [Self-Play Coordinator](selfplay-coordinator.md) — persistent game loops, game_task, inference batching, training pipeline
- [Neural Networks](neural-networks.md) — MuZero h/g/f models, tensor shapes, K-step training loop, SimSiam consistency, pretraining
- [Replay Subsystem](replay-subsystem.md) — in-memory training buffer + opt-in per-ply MCTS replay capture and viewer

## Evaluation & Promotion
- [Elo Ladder Evaluation](elo-ladder-eval.md) — per-opponent champion-challenger ladder, Elo math, promotion gates, `ladder_match` log line
- [Champion Pool & Promotion](champion-pool-promotion.md) — champion archive pool, `best.pt` / `best_v{NNN}.pt`, archive pruning, version recovery
- [Baseline Scoring](baseline-scoring.md) — `run_baseline.sh`, composite score formula, `baseline_score.json` schema

## Integration & Sessions
- [Rust-Python Integration](rust-python-integration.md) — PyO3 bridge, FFI boundary, sidecar inference/training servers
- [Session & Net Play](session.md) — game session stub, Unix-socket server/client binaries

## Development
- [Testing Procedures](testing.md) — test commands, perft cross-validation, conventions
- [Dev Workflow & Framework](dev-workflow.md) — thin-orchestrator workflow, hooks, skills, analyst/developer roles
