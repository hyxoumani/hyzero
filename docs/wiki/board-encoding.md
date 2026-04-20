# Board Encoding & Observation Format

The board observation tensor encodes the game state from the current player's perspective, following the AlphaZero convention. This ensures the network learns symmetric value estimates and consistent policy targets regardless of which side is to move.

## Observation Planes (19 total)

**Planes 0–5: Current player's pieces** (piece-color-agnostic)
- Plane 0: Pawns
- Plane 1: Knights
- Plane 2: Bishops
- Plane 3: Rooks
- Plane 4: Queens
- Plane 5: Kings

**Planes 6–11: Opponent's pieces**
- Same structure, opponent's side

**For Black-to-move**: Planes are rank-mirrored so the current player always occupies the bottom two ranks (6–8) from the network's perspective, opponent occupies top ranks (1–3).

**Planes 12–15: Castling rights** (4 binary planes)
- Plane 12: White kingside
- Plane 13: White queenside
- Plane 14: Black kingside
- Plane 15: Black queenside

**Plane 16: En passant target square** (1 if ep-square exists, 0 else)

**Plane 17: Side to move** (0 = White, 1 = Black)

**Plane 18: Halfmove clock** (0–50, normalized to [0,1])

## AlphaZero Perspective Convention (Commit bb39db6)

**Key principle**: The observation always encodes from the side-to-move's perspective.

- **Input to h-network**: [B, 19, 8, 8] — board from current player's view
- **Output from f-network**: Policy [B, 4096] and value [B] in current-player space
- **MCTS action space**: Actions encoded relative to current player (0–4095)

**Rank mirroring (for Black-to-move)**:
- Board is flipped so Black's pieces appear in planes 0–5, White in 6–11
- Board is rotated 180° so rank 8 (Black's back rank) appears at the bottom
- En passant, castling, promotion targets all adjusted to match the rotated frame

## Action Encoding & Flipping

Actions are 4096-dimensional: 64 from-squares × 64 to-squares = 4096 (promotion defaults to Queen).

**At MCTS boundary** (game_task.rs):
1. Inference receives current-player-perspective observation
2. Network outputs policy logits for current-player actions
3. Action selected via MCTS visit counts
4. **Action is flipped** to absolute board space before applying to game board
5. Board state moves to opponent; next observation flipped to new current-player perspective

**flip_action(action, color)**: Transforms an action from current-player space to White-absolute space.
- If color = White, action unchanged
- If color = Black, both from-square and to-square are rotated 180° (from_sq → 63−from_sq, to_sq → 63−to_sq)

**Action Ordering and POV Symmetry (Commit 41f6681)**: `get_legal_moves()` returns actions in absolute-square iteration order, NOT in POV-symmetric order. This means the same move (e.g., Nc3 for White, Nc6 for Black) will appear at different indices in `legal_actions`. To present an action list with identical index geometry to both colors on mirror-equivalent positions, callers must sort: `legal_actions.sort_unstable()` after the POV-flip step. MCTS simulations and inference depend on consistent action indexing; if you modify action selection code, verify it works symmetrically on both colors using `test_legal_actions_ordering_is_color_symmetric_after_sort()`.

## action_to_notation Bug Fix

