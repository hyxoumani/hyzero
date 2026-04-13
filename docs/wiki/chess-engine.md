# Chess Engine (Bitboard + Magic Bitboards)

## Board Representation

**Bitboards**: `u64` where bit position = square (rank-major: `sq = rank*8 + file`, A1=0, H8=63). Per-player: `pieces_bb[6]` (one per PieceType). Mailbox fallback: `own_board[64]` for validation.

**Zobrist hashing**: 781 pseudo-random 64-bit values (splitmix64-seeded), one per piece/square combo. Maintained incrementally across all operations. Collision probability < 1 in 10^9. Replaces old `wrapping_mul` hash.

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

**Pin detection**: For each enemy slider, check precomputed ray masks. Friendly piece blocking ray = pinned.

## Board Initialization from FEN

`board_from_fen()` creates arbitrary positions from FEN. **Rank mapping**: FEN rank 8 = board rank 7, FEN rank 1 = board rank 0. Square = `board_rank * 8 + file`. Supports full FEN syntax (placement, color, castling, EP, clocks).

## Test Coverage

**Unit tests (14)**: Move generation (all pieces), special moves, game status (check/mate/stalemate), draw rules.

**Perft validation (10)**: Kiwipete, positions 3/5, known-correct depths (startpos: 20 moves, 400 at d=2).

## Key Gotchas

1. **Square indexing**: Convert bitboard values to indices: `trailing_zeros() as usize`. Magic table offsets baked in.
2. **Occupancy masks**: Exclude target square edges. Mistakes = hash collisions.
3. **Array + bitboard sync**: `update_board()` modifies both. Forget one = subtle bugs (pin detection, check). Always update in pairs.
4. **King square lookup**: `get_king_sq()` returns bitboard (e.g., `1 << 60`), NOT index. Use `.trailing_zeros()` before array access.
5. **Castling not in move mask**: `get_move_mask` returns 1-square moves only. Castling via `get_castling_moves()` separately.
6. **En passant flag cosmetic**: `Move.en_passant` is unused. `update_board()` checks `en_passant_target` directly.

## Related

- [Special Moves & Draw Rules](special-moves-draws.md) — castling, en passant, promotion, game termination
- [MCTS & Self-Play](mcts-selfplay.md) — how engine is used in game loops
- [Rust-Python Integration](rust-python-integration.md) — action encoding and decoding

## Related Files

- `src/lib.rs` — ZobristTable, splitmix64, PrecomputedItems
- `src/game/board.rs` — move generation, validation, pins, check/mate, zobrist updates, 18 tests
- `src/game/fen.rs` — FEN parser with 5 tests
- `src/game/perft.rs` — perft driver with 10 tests
- `src/game/playerobj.rs` — per-player bitboards
- `src/pieces/{rook,bishop}.rs` — magic table generation
