# Project Roadmap

Current state and next steps for hyzero.

## What's Done

- **Tasks 1-16**: Chess engine — bitboards, magic move gen, all special moves, validation
- **Tasks 17-23**: MCTS tree search, self-play pipeline, inference batching, replay buffer
- **Tasks 24-26**: Python MuZero models (h/g/f networks) with training loop and inference server
- **Task 27**: MCTS value negation — fixed backpropagation to negate values per ply (two-player zero-sum)
- **Task 28**: PyO3 integration — full Rust↔Python bridge (PyO3Backend, PyTrainingThread, batch assembly, weight sync)
- **Task 29**: End-to-end validation — full loop tested (5 games, 13 training steps, loss 8.52→7.04); Dirichlet noise implemented; max game length added; multi-step training enabled
- **Zobrist hashing**: Replaced collision-prone position hash with proper Zobrist tables (781 splitmix64-seeded values). Maintained incrementally across all board ops.
- **FEN parser**: New `board_from_fen()` function for creating arbitrary board positions from FEN notation.
- **action_to_move fix**: Signature changed to `(action, board, color)` for correct castling/en passant reconstruction with board context.
- **Engine unit tests**: 14 tests covering move generation, special moves, check/mate/stalemate, draw rules (4 Zobrist + 14 game logic).
- **Perft validation**: Perft driver + 10 tests against known-correct counts (startpos, Kiwipete, positions 3/5).
- **Clippy**: Zero warnings, stalemate bitboard-index bug found and fixed
- **Test Suite**: 63 Rust + 27 Python tests (5 new perft tests in ignore list); 3 integration tests with PyO3
- **Framework**: v0.2.0 — wiki, agent memory, tool-augmented review gates, orchestrator persistence
- **Infrastructure**: e2e_test.sh (autonomous validation) and run_experiment.sh (experiment runner) added

## What's Next

### Unscoped: Training Optimization
- Scale up num_simulations (currently 50 for CPU dev; 200+ for production)
- Extend game count (currently 5 per run; target 50+)
- Add convergence criteria (when loss plateaus, start new run)
- GPU inference latency and batch throughput profiling

### Unscoped: Board Representation
- Dynamic action space: optimize encoder to suppress illegal moves (currently 4096 fixed)
- Underpromotion support: add planes for knight/bishop/rook promotions
- Alternative encoding: explore 1D action flattening or hierarchical representation

### Unscoped: MCTS Enhancements
- Root noise for exploration
- Transposition tables for game states
- Depth-based temperature scaling in move selection

## Known Risks

- **Dirichlet noise CPU overhead**: Marsaglia-Tsang Gamma sampling for Dir(0.03) is slow in debug mode; must use `--release` for e2e tests
- **Game length scaling**: Value negation fix + Dirichlet noise cause games to run 3-4× longer (now ~200 moves); correct for better play but impacts iteration speed
- **Batch timeout tuning**: Few concurrent games → small batches → lower GPU utilization; 10ms timeout is empirical, may need adjustment for higher concurrency
- **GIL contention**: One GIL acquisition per batch (32 requests) — monitor with profiling
- **Array/bitboard sync**: `board_arr` and `pieces_bb` must stay in sync — `update_board()` is complex
- **Visit distribution padding**: StepRecord visit_distribution may be < 4096 — must zero-pad for batch consistency
- **Stdout buffering**: `cargo run` buffers output; run binary directly (`target/release/selfplay`) for proper log capture in scripts

## Related
- [Neural Networks](neural-networks.md) — model architecture details
- [MCTS & Self-Play](mcts-selfplay.md) — pipeline architecture
- [Rust-Python Integration](rust-python-integration.md) — FFI boundary
