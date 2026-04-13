# Project Roadmap

Current state and next steps for hyzero.

## What's Done

- **Tasks 1-16**: Chess engine — bitboards, magic move gen, all special moves, validation
- **Tasks 17-23**: MCTS tree search, self-play pipeline, inference batching, replay buffer
- **Tasks 24-26**: Python MuZero models (h/g/f networks) with training loop and inference server
- **Task 27**: MCTS value negation — fixed backpropagation to negate values per ply (two-player zero-sum)
- **Task 28**: PyO3 integration — full Rust↔Python bridge (PyO3Backend, PyTrainingThread, batch assembly, weight sync)
- **Task 29**: End-to-end validation — full loop tested (5 games, 13 training steps, loss 8.52→7.04); Dirichlet noise implemented; max game length added; multi-step training enabled
- **Task 30 (Engine Foundation)**:
  - **Zobrist hashing**: Replaced collision-prone `wrapping_mul` hash with proper Zobrist tables (781 splitmix64-seeded values). Maintained incrementally via XOR in `update_board()`.
  - **FEN parser**: New `board_from_fen()` function for arbitrary board positions from FEN notation.
  - **action_to_move fix**: Signature changed to `(action, board, color)` for correct castling/en passant reconstruction.
  - **Unit tests**: 24 tests (4 zobrist + 14 game logic + 6 encoding); 4 zobrist consistency tests; pin detection tests; move validation tests.
  - **Perft validation**: Driver + 13 tests (10 pass, 3 ignored for speed) against known-correct counts (startpos d1-d5, Kiwipete d1-d3, position 3, position 5). Terminal counting bug fixed (removed incorrect +1 per checkmate).
  - **calculate_pins() queen fix**: Missing queen from enemy_sliders caused false negatives in checkmate/stalemate detection. Fixed.
  - **Test counts**: 63 Rust (6 ignored), 27 Python, 3 integration with PyO3. All pass.
- **Clippy**: Zero warnings, stalemate bitboard-index bug found and fixed
- **Framework**: v0.2.0 — wiki, agent memory, tool-augmented review gates, orchestrator persistence
- **Infrastructure**: e2e_test.sh (autonomous validation) and run_experiment.sh (experiment runner) added
- **Task 31 (Engine Validation)**:
  - **Perft CLI**: `src/bin/perft.rs` with `--divide`, `--moves`, `--status` modes for batch validation and debugging.
  - **Cross-validation script**: `scripts/cross_validate.py` — python-chess comparisons (perft, legal moves, termination status, fuzz testing with 5500+ random positions).
  - **Stalemate castling bug**: `calculate_stalemate()` never checked castling as escape move. Fixed by calling `validate_move()` for both castle options after king 1-square loop. Added 2 tests.
  - **Stalemate parameter ordering bug**: `calculate_stalemate()` passed `(friendly_bits, opponent_bits)` to `get_move_mask()` which expects `(white_pieces, black_pieces)`. For Black-to-move, bits were swapped. Fixed by deriving canonical white/black from color at function entry. Added 7 tests.
  - **Threefold repetition off-by-one**: `position_history` started empty; initial position never counted as first occurrence, requiring 4 repetitions instead of 3. Fixed by inserting `board.position_history.insert(board.zobrist_hash, 1)` after board construction. Added 2 tests.
  - **Test counts**: 80 Rust (6 ignored), perft 28/28 (6 positions × multiple depths, ~53M nodes), termination 13/13 vs python-chess, fuzz 5500+ random positions.
  - **Infrastructure**: Perft CLI for local debugging; cross_validate.py for continuous validation against reference engine.

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
- **Perft reference verification**: Cross-validated against python-chess for perft (28 positions tested), moves (10 edge cases), and termination status (13 draws + checkmate cases). All match. Future heavy perft work should cross-check with Stockfish as independent verification.

## Related
- [Neural Networks](neural-networks.md) — model architecture details
- [MCTS & Self-Play](mcts-selfplay.md) — pipeline architecture
- [Rust-Python Integration](rust-python-integration.md) — FFI boundary
