# Chess Engine (Bitboard + Magic Bitboards)

## Board Representation

**Bitboards**: `u64` where bit position = square (rank-major: `sq = rank*8 + file`, A1=0, H8=63). Per-player: `pieces_bb[6]` (one per PieceType). Mailbox fallback: `own_board[64]` for validation.

**Zobrist hashing**: 781 pseudo-random 64-bit values (splitmix64-seeded via `splitmix64()` in `src/lib.rs`). One per piece/square/color combo (PieceType::Queen=4, etc.). Maintained incrementally in `update_board()` via XOR operations: `hash ^= zobrist.piece_sq[color][piece_type][sq]` — **color is the outer index**, then piece_type, then square (`piece_sq: [[[u64; 64]; 6]; 2]`, src/lib.rs:117-118). Collision probability < 1 in 10^9. See gotcha #1.

**Magic bitboards** (O(1) sliding move gen): Pre-computed tables indexed by occupancy. Rook/bishop entries contain `mask`, `magic_num`, `sig_bits`, `magic_table`. Generated at startup (~1s). Rook magics live in `src/pieces/mod_rook.rs`; bishop magics in `src/pieces/bishop.rs`.

## Move Generation

**Entry point**: `GameBoard::get_move_mask(sq, color, piece, board, white_pieces, black_pieces)` (src/game/board.rs:902) — dispatches by piece type. Takes the square, side-to-move color, the `PieceType`, the combined-occupancy `board` bitboard, and the canonical white/black piece bitboards.

**Sliding pieces** (Rook, Bishop, Queen): Magic bitboard tables indexed by occupancy.

**Knight/King**: Pre-computed lookup tables. **King limitation**: `get_move_mask` returns only 1-square moves. Castling is added separately (see below).

**Pawn moves**: Diagonals (capture), forward push, double-push from rank 2/7, promotion defaults to Queen.

**Special move flags** in `Move`: `castle_option`, `en_passant`, `promotion_piece_type`. See "Special Moves" below.

## Move Validation

**Pseudo-legal check**: From/to piece validation + piece can reach to square.

**Legal check**: Clone board, apply move, recalculate pins, check if own king in check.

**Pin detection**: For each enemy slider (rook/bishop/queen), check precomputed ray masks. Friendly piece blocking ray = pinned. Used by `calculate_checkmate()` and `calculate_stalemate()` to check if pinned pieces have escape moves.

## Board Initialization from FEN

`board_from_fen()` creates arbitrary positions from FEN. **Rank mapping**: FEN rank 8 = board rank 7, FEN rank 1 = board rank 0. Square = `board_rank * 8 + file`. Supports full FEN syntax (placement, color, castling, EP, clocks).

## Special Moves

**Castling**:
- King-side empty squares: f1, g1 (White) / f8, g8 (Black) — `castle_empty_squares[color][Kingside]` (src/lib.rs:187-192).
- Queen-side empty squares: **b1, c1, d1** (White) / **b8, c8, d8** (Black) — three squares each, c-file included — `castle_empty_squares[color][Queenside]` (src/lib.rs:189-194).
- Path squares (king passes through) also checked not-under-attack via `castle_path_squares`; king may not castle through or into check.
- Castling rights tracked per side (updated after king/rook moves).
- King moves 2 squares to the c- or g-file; rook moves to the adjacent square.
- **Not in the move mask**: `get_move_mask` returns 1-square king moves only. There is **no `get_castling_moves()` helper** — callers generate castling inline by looping `[CastleOption::Kingside, CastleOption::Queenside]`, building a `Move` with `castle_option: Some(opt)` and the king's target file (g=6 / c=2) on the king's rank, then running it through `validate_move()` (src/game/perft.rs:139-159, src/selfplay/game_task.rs:828-854).

**En passant**:
- `en_passant_target: Option<Square>` set after a pawn double-push
- If an opponent pawn sits on `from_sq ± 1`, capture with an EP move
- Captured pawn removed in `update_board()` (offset by rank)
- **Cosmetic flag**: `Move.en_passant` is unused; `update_board()` checks `en_passant_target` directly

**Promotion**:
- Only on the last rank (rank 7 for White, rank 0 for Black)
- Pawn removed, new piece added to the promoted square
- Promotion type carried in the `Move` struct; move-gen defaults to Queen

## Draw Rules

**50-move rule**: Halfmove clock incremented after every move (reset on a pawn move or capture). Drawn when the clock ≥ 100.

**Threefold repetition**: Zobrist hash of the position recorded after each move; drawn when the same position appears 3×. The initial position is inserted into `position_history` (fixes an off-by-one that previously required 4 occurrences).

**Insufficient material** (`is_insufficient_material()`, src/game/board.rs:1047-1094): Returns **not** a draw if either side has a pawn, rook, or queen. With only kings + minor pieces (knights/bishops), it is a draw only for: K vs K; K+single-minor vs K; or K+B vs K+B where **both bishops sit on same-colored squares** (`(rank+file)%2` equal). Any side with 2+ minor pieces (and the opponent with material) is **not** auto-drawn.

## Key Gotchas

1. **Zobrist maintains incrementally**: XOR the affected squares' zobrist values into the running hash after each move. Never recompute from scratch. Index order is `piece_sq[color][piece_type][sq]` (color outer).
2. **Square indexing**: Convert bitboard values to indices via `trailing_zeros() as usize`. Magic table offsets are baked in.
3. **Occupancy masks**: Exclude target-square edges. Mistakes cause hash collisions.
4. **Array + bitboard sync**: `update_board()` modifies both. Forgetting one causes subtle pin/check bugs. Always update in pairs.
5. **King square lookup**: `get_king_sq()` returns a bitboard (e.g. `1 << 60`), NOT an index. Apply `.trailing_zeros()` before array access.
6. **Castling not in move mask**: see "Special Moves". No `get_castling_moves()` exists — generation code must loop `CastleOption` variants inline and validate each candidate alongside `get_move_mask` output.
7. **En passant flag cosmetic**: `Move.en_passant` is unused; `update_board()` reads `en_passant_target`.
8. **Pins require all sliders**: `calculate_pins()` checks rook/bishop/queen. A missing slider caused false negatives in mate/stalemate detection.
9. **Stalemate parameter ordering**: `calculate_stalemate()` passes canonical `(white_pieces, black_pieces)` to `get_move_mask()`, not friendly/opponent bits. Black-to-move with flipped bits → wrong masks → missed escapes.
10. **Stalemate castling escape**: After checking 1-square king moves, also validate castling candidates as escape options (rare but legal).
11. **Same-color-bishop draw**: K+B vs K+B is a draw only when both bishops are on the same square color; opposite-colored bishops are not insufficient material.

## Related

- [Board Encoding](board-encoding.md) — observation tensor, action encoding/decoding
- [MCTS](mcts.md) — how the engine drives game loops
- [Self-Play Coordinator](selfplay-coordinator.md) — game termination paths

## Related Files

- `src/lib.rs` — ZobristTable (781 values, `piece_sq[color][piece][sq]`, splitmix64), `PrecomputedItems`, `castle_empty_squares`/`castle_path_squares`
- `src/game/board.rs` — move gen (`get_move_mask`), validation, pins, check/mate/draw, incremental zobrist, `is_insufficient_material`
- `src/game/fen.rs` — FEN parser
- `src/game/perft.rs` — perft driver (inline castling generation)
- `src/selfplay/game_task.rs` — self-play legal-move enumeration (inline castling generation)
- `src/game/playerobj.rs` — per-player bitboards
- `src/pieces/mod_rook.rs`, `src/pieces/bishop.rs` — magic table generation
- `src/pieces/{king,knight,pawn,queen}.rs` — non-sliding move tables
- `src/data/encoding.rs` — `action_to_move(action, board, color)`
- `src/bin/perft.rs` — CLI: `cargo run --release --bin perft -- [--divide|--moves|--status] <FEN> [depth]`
- `scripts/cross_validate.py` — python-chess validator (perft, moves, termination, fuzz)
