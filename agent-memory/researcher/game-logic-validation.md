# Game Logic Validation State

## Test Results (Latest)
- **Cargo test**: 24 passed, 0 failed, 3 ignored (PyO3 tests requiring Python env)
- **Status**: All tests passing, no broken game logic

## Game Logic Coverage

### What's Tested
1. **Move Generation**: 
   - `test_legal_moves_starting_position`: Verifies 20 legal moves from start (16 pawns + 4 knights) ✓
   - Implicitly tested through `test_play_game_completes` (async game plays to completion)

2. **Game Loop**:
   - `test_play_game_completes`: Confirms game executes, produces trajectory steps, outcomes ✓
   - Game state with MCTS integration verified (2 simulations used in test)

3. **Special Moves**:
   - Castling: Validation in `validate_move()` (lines 342+)
   - En Passant: `en_passant_target` field set on double-push, capture logic in `update_board()`
   - Promotion: Handled in Move struct with `promotion_piece_type` field
   - All logic compiled and functional in e2e tests

4. **Draw Rules**:
   - **50-move rule**: `halfmove_clock` logic (lines 145-153)
   - **Threefold repetition**: `position_history` HashMap with position hashing (lines 155-161)
   - **Insufficient material**: `is_insufficient_material()` method (line 164)
   - All enum variants exist: Checkmate, Stalemate, FiftyMoveRule, ThreefoldRepetition, InsufficientMaterial

5. **Check/Checkmate/Stalemate**:
   - `calculate_checkmate()` (line 286) — checks attacker count, evaluates legal moves
   - `calculate_stalemate()` (line 258) — checks not in check but has no legal moves
   - Pin detection `calculate_pins()` (line 197) — ray-based, both-side calculation

### What's NOT Explicitly Unit-Tested
- **No dedicated unit tests in `src/game/`** — no `#[cfg(test)]` blocks in board.rs, playerobj.rs, etc.
- Move validation corner cases (pins, discovered check, etc.) validated only through integration
- Castling with different board positions (blocked path, king in check, rook captured)
- En passant edge cases
- Promotion piece selection
- Draw rule detection against known problem positions

## E2E Validation
- **Script**: `scripts/e2e_test.sh` runs selfplay for 120s, extracts metrics
- **Latest run** (20260411_204914):
  - Games completed: 5
  - Avg game steps: 214 (all games running to completion)
  - Training steps: 13 (losses from 8.52 → 7.04)
  - No errors logged
  - Games not ending prematurely = basic game logic functional

## Known Issues & Gotchas (from wiki/chess-engine.md)

1. **King square lookup**: Must use `.trailing_zeros()` on bitboard value, not direct cast
   - Lines 266, 292 in board.rs show correct pattern
   
2. **Array + bitboard sync**: `board_arr` and `pieces_bb` must both be updated in `update_board()`
   - Complex method, risk of desync

3. **in_check field**: Marked `#[allow(dead_code)]`, computed dynamically not cached
   - Not a bug but impacts performance

4. **Occupancy in magic lookup**: Mask and occupancy both exclude target square edges
   - Pre-computed tables must match this convention

## Confidence Assessment
- **Core game logic working**: 95% confidence (all tests pass, e2e games complete)
- **Edge case coverage**: 70% confidence (no dedicated unit tests for corner cases)
- **Special moves complete**: 90% confidence (code exists and compiles, e2e doesn't fail)
- **Draw rules complete**: 85% confidence (all 5 conditions implemented, not stress-tested)

## Recommendations for Enhancement
1. Add dedicated unit tests in `src/game/board.rs` for:
   - Castling blocking/rights scenarios
   - En passant captures and double-push detection
   - Promotion piece selection
   - Threefold repetition hash correctness
   - Insufficient material edge cases

2. Stress-test long games to verify 50-move and threefold detection

3. Add regression tests for known gotchas (king sq lookup, pin calculation)
