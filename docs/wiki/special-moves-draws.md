# Special Moves & Draw Rules

Chess-specific rules that affect game termination and piece movement.

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
- Note: `Move.en_passant` flag is cosmetic — `update_board()` checks `en_passant_target` directly

**Promotion**:
- Only on last rank (7 for White, 0 for Black)
- Pawn removed, new piece added to promoted square
- Promotion type in `Move` struct

## Draw Rules

**50-move rule**: Halfmove clock incremented after every move (reset on pawn move or capture). Game drawn if clock ≥ 100.

**Threefold repetition**: Hash of board position after each move. Game drawn if same position appears 3x. Zobrist hash used for fast comparisons.

**Insufficient material**: Neither side has queen, rook, or pawn; only kings and knights/bishops. Game drawn.

## Related

- [Chess Engine](chess-engine.md) — board representation, move generation
- [MCTS & Self-Play](mcts-selfplay.md) — game termination, move selection
