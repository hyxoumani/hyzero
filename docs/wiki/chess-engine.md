# Chess Engine (Bitboard + Magic Bitboards)

## Board Representation

**Bitboards**: `u64` where bit position = square (rank-major: `sq = rank*8 + file`, A1=0, H8=63). Per-player: `pieces_bb[6]` (one per PieceType). Mailbox fallback: `own_board[64]` for validation.

**Zobrist hashing**: 781 pseudo-random 64-bit values (splitmix64-seeded via `splitmix64()` in `src/lib.rs`). One per piece/square combo (PieceType::Queen=4, etc.). Maintained incrementally in `update_board()` via XOR operations: `hash ^= ZobristTable[piece_type][color][sq]`. Collision probability < 1 in 10^9. Replaced old `wrapping_mul`-based hash. See gotcha #1.

**Magic bitboards** (O(1) move gen): Pre-computed tables indexed by occupancy. Rook/bishop entries contain `mask`, `magic_num`, `sig_bits`, `magic_table`. Pre-calculated at startup (~1s).

## Move Generation

**Entry point**: `GameBoard::get_move_mask(square, color)` — dispatches by piece type.

**Sliding pieces** (Rook, Bishop, Queen): Use magic bitboard tables indexed by occupancy.

**Knight/King**: Pre-computed lookup tables. **King limitation**: `get_move_mask` returns only 1-square moves. Castling generated separately via `get_castling_moves()`.

**Pawn moves**: Diagonals (capture), forward push, double-push from rank 2/7, promotion defaults to Queen.

**Special move flags** in `Move`: `castle_option`, `en_passant`, `promotion_piece_type`. Details in [Special Moves & Draw Rules](special-moves-draws.md).

## Move Validation

**Pseudo-legal check**: From/to piece validation + piece can reach to square.

**Legal check**: Clone board, apply move, recalculate pins, check if own king in check.

**Pin detection**: For each enemy slider (rook/bishop/queen), check precomputed ray masks. Friendly piece blocking ray = pinned. Used by `calculate_checkmate()` and `calculate_stalemate()` to check if pinned pieces have escape moves.

## Board Initialization from FEN

`board_from_fen()` creates arbitrary positions from FEN. **Rank mapping**: FEN rank 8 = board rank 7, FEN rank 1 = board rank 0. Square = `board_rank * 8 + file`. Supports full FEN syntax (placement, color, castling, EP, clocks).

## Test Coverage

**Unit tests (80 total, 6 ignored)**: Original 24 (4 Zobrist, 14 game logic, 6 action_to_move) plus 56 new from validation work:
- Perft 28 tests (6 positions × multiple depths, ~53M nodes total); 10 pass, 3 ignored for speed per position
- Moves 10 tests (edge cases vs python-chess: castling, en passant, promotion, pins, checks)
- Termination 13 tests (stalemate/checkmate vs python-chess; includes new castling-escape, parameter-order, repetition tests)
- Fuzz 5500+ random positions (moves + termination consistency)

**Bugs fixed**: Perft terminal-counting convention (removed non-standard +1), stalemate castling escape (added validate_move calls for both castle options), stalemate parameter ordering (canonical white/black from color), threefold repetition off-by-one (insert initial position into position_history).

## Key Gotchas

1. **Zobrist maintains incrementally**: After each move, XOR the affected squares' zobrist values into the running hash. Never recalculate from scratch (slow). Old `position_hash()` method using `wrapping_mul` was removed.
2. **Square indexing**: Convert bitboard values to indices: `trailing_zeros() as usize`. Magic table offsets baked in.
3. **Occupancy masks**: Exclude target square edges. Mistakes = hash collisions.
4. **Array + bitboard sync**: `update_board()` modifies both. Forget one = subtle bugs (pin detection, check). Always update in pairs.
5. **King square lookup**: `get_king_sq()` returns bitboard (e.g., `1 << 60`), NOT index. Use `.trailing_zeros()` before array access.
6. **Castling not in move mask**: `get_move_mask` returns 1-square moves only. Castling via `get_castling_moves()` separately. Move generation code must call both functions (e.g., in perft, game_task, validation).
7. **En passant flag cosmetic**: `Move.en_passant` is unused. `update_board()` checks `en_passant_target` directly.
8. **Pins require all sliders**: `calculate_pins()` checks rook/bishop/queen for pinning pieces. Missing Queen used to cause false negatives in checkmate/stalemate detection. Now fixed.
9. **Stalemate parameter ordering**: `calculate_stalemate()` must pass canonical `(white_pieces, black_pieces)` to `get_move_mask()`, not friendly/opponent bits. For Black-to-move, flipped bits → wrong move masks → missed escapes. Fixed by deriving canonical color at function entry.
10. **Stalemate castling escape**: After checking king 1-square moves, must also validate castling moves as escape options. Previous code checked only king moves and pins, missing that castling can escape stalemate (rare but legal in exact position context). Fixed by calling `validate_move()` for both castle options.

## Related

- [Special Moves & Draw Rules](special-moves-draws.md) — castling, en passant, promotion, game termination
- [MCTS & Self-Play](mcts-selfplay.md) — how engine is used in game loops
- [Rust-Python Integration](rust-python-integration.md) — action encoding and decoding

## Related Files

- `src/lib.rs` — ZobristTable (781 values, splitmix64), PrecomputedItems
- `src/game/board.rs` — move generation, validation, pins (queen fix), check/mate/draw (stalemate fixes), zobrist incremental, 24 tests
- `src/game/fen.rs` — FEN parser, 5 tests
- `src/game/perft.rs` — perft driver (terminal counting fixed), 28 tests (10 pass, 3 ignored slow per position)
- `src/game/playerobj.rs` — per-player bitboards
- `src/pieces/{rook,bishop}.rs` — magic table generation
- `src/data/encoding.rs` — action_to_move (signature changed to accept board + color)
- `src/bin/perft.rs` — CLI tool: `cargo run --release --bin perft -- [--divide|--moves|--status] <FEN> [depth]`
- `scripts/cross_validate.py` — python-chess validator (perft, moves, termination, fuzz)
