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

## Representation Consistency Invariants

When implementing board transformations under color augmentation, verify these invariants to catch asymmetries:

1. **Action-spatial encode invariant**: `encode_action_spatial(flip_action(a), white_to_move=False) == flip_action_planes(encode_action_spatial(a, white_to_move=True))`
   - Ensures action encodings are symmetric under color flip
   - Particularly important for underpromotion, which has color-agnostic action IDs but color-specific output planes
   - Regression test: `test_encode_action_spatial_under_color_flip` (commit cc58506)

2. **Observation-flip invariant**: `encode_obs_planes(flip_board(b)) == flip_action_planes(encode_obs_planes(b))`
   - Ensure observation encoding transforms consistently with action encoding
   - Useful to extend regression tests once action invariant is verified

3. **Legal-mask consistency**: Legal-move mask construction must account for color perspective changes. Always call `get_legal_moves()` *after* perspective flips, or verify masks are color-independent.

**Bug fixed (commit cc58506)**: Underpromo action indices (4096–4671) are color-agnostic IDs, but `encode_action_spatial(action, white_to_move)` returns color-specific spatial planes (ranks 6→7 for White, ranks 1→0 for Black). Under color augmentation, this broke the encode-flip invariant for all 576 underpromotion actions. Fix: use `encode_action_spatial_for_color(action, white_to_move)` when color context matters.

## Encoder Validation

Python-side encoder available at `scripts/reward_probe.py` with helper functions `encode_board_python()` and `encode_action_spatial()`. Used for diagnostic probes on held-out checkpoints to validate distributional collapse (2026-04-21 session). Validated byte-identical to Rust encoder on initial position via trainer's `[start_value]` probe.

## Related

- [MCTS & Self-Play](mcts-selfplay.md) — action flipping at MCTS boundary, legal-actions ordering
- [Neural Networks](neural-networks.md) — current-player perspective in observation planes
- `docs/wiki/mistakes.md` — encoding asymmetry bugs and fixes (2026-04-17, 2026-04-20 entries)

