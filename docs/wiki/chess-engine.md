# Chess Engine (Bitboard + Magic Bitboards)

## Board Representation

**Bitboards**: `u64` where bit position = square (rank-major: `sq = rank*8 + file`, A1=0, H8=63)

**Magic bitboards**: Pre-computed lookup tables for rook/bishop move generation (O(1) per piece):
- `RookEntry` / `BishopEntry` contain: `mask`, `magic_num`, `sig_bits`, `magic_table`
- Lookup: `table[(mask & occupancy).wrapping_mul(magic) >> (64 - sig_bits)]`
- Pre-calculated at startup via `PrecomputedItems::begin_precomputing()` (takes ~1s)

**Piece representation**:
- Per-player: `pieces_bb[6]` (u64 array, one per PieceType)
- Mailbox fallback: `own_board[64]` (Option<Piece>), used for move validation

## Move Generation

**Entry point**: `GameBoard::get_move_mask(square, color)` — dispatches by piece type

**Sliding pieces** (Rook, Bishop, Queen):
- Call `get_sliding_moves()` with piece type
- Use magic bitboard table indexed by occupancy
- Returns bitboard of legal moves for that piece

**Knight/King**: Pre-computed lookup tables in `PrecomputedItems`

**Pawn moves**:
- Capture diagonally one square (forward based on color)
- Push forward one square (empty square check)
- Double-push from starting rank (rank 2 for White, rank 7 for Black)
- Promotion: always defaults to Queen (underpromotion added later as 4672 actions)

**Special moves encoded in `Move` struct**:
- `castle_option`: Some(WK/WQ/BK/BQ) for castling
- `en_passant`: true if EP capture
- `promotion_piece_type`: Some(Queen/Rook/Bishop/Knight)

## Move Validation

**Pseudo-legal check** (before applying):
- From square has own piece
- To square empty or enemy piece
- Piece can reach to square (via move generation)

**Legal check** (after clone + apply):
- Simulate the move on a cloned board
- Recalculate all pins
- Check if own king in check
- If so, undo and reject

**Pin detection** (`calculate_pins()`):
- For each enemy slider (rook/bishop), use precomputed ray masks
- Ray from enemy to own king: if friendly piece blocks, it's pinned
- Pinned pieces stored in `pin_masks[to_sq]` bitboard

## Special Moves

**Castling**:
- King-side: `castle_squares = [F1, G1, F8, G8]` (must be empty)
- Queen-side: `castle_squares = [B1, D1, B8, D8]` (must be empty)
- Path squares also checked empty
- Castling rights tracked per side (updated after king/rook moves)
- King moves 2 squares, rook moves to adjacent square

**En passant**:
- `en_passant_target: Option<Square>` set after pawn double-push
- If opponent pawn on (from_sq ± 1), capture with EP move
- Captured pawn removed from `update_board()` (offset by rank)

**Promotion**:
- Only on last rank (7 for White, 0 for Black)
- Pawn removed, new piece added to promoted square
- Promotion type in `Move` struct

## Draw Rules

**50-move rule**: Halfmove clock incremented after every move (reset on pawn move or capture). Game drawn if clock ≥ 100.

**Threefold repetition**: Hash of board position after each move. Game drawn if same position appears 3x.

**Insufficient material**: Neither side has queen, rook, or pawn; only kings and knights/bishops. Game drawn.

## Key Gotchas

1. **Square indexing rank-major**: Magic table base offsets baked in. Convert carefully between bitboard value and square index: `trailing_zeros() as usize`.

2. **Occupancy in magic lookup**: The mask and occupancy both exclude target square edges. Get this wrong = hash collisions.

3. **Array + bitboard sync**: `update_board()` modifies both bitboards and mailbox. If one is forgotten, subtle bugs (e.g., pin detection fails). Always update in pairs.

4. **in_check field unused**: Computed dynamically instead of cached. Not a bug, but means recalculation on every validation.

5. **King square lookup**: `get_king_sq()` returns a bitboard (e.g., `1 << 60`), NOT a square index. Convert with `.trailing_zeros()` before array indexing. See board_index_panic.md for history.

## Related Files

- `src/lib.rs` — types, precomputation
- `src/game/board.rs` — move generation, validation, pins, check/mate
- `src/game/playerobj.rs` — per-player bitboards
- `src/pieces/{rook,bishop}.rs` — magic table generation
