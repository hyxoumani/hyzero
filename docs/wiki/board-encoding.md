# Board Encoding & Observation Format

The board observation tensor encodes the game state from the current player's perspective, following the AlphaZero convention. This ensures the network learns symmetric value estimates and consistent policy targets regardless of which side is to move.

See `neural-networks.md` for network input shape; this page is the source of truth for plane definitions.

## Observation Planes (102 total)

`NUM_OBS_PLANES = 102` (defined in `src/data/types.rs`). Tensor shape is `[B, 102, 8, 8]` — 96 piece planes covering the current position plus 7 historical positions, then 6 game-state planes.

**Planes 0–11: Current position pieces** (12 planes, current-player perspective)
- Planes 0–5: Current player's pieces — Pawns, Knights, Bishops, Rooks, Queens, Kings
- Planes 6–11: Opponent's pieces — same order

For Black-to-move, planes are rank-mirrored so the current player's pieces always occupy the bottom ranks from the network's perspective; opponent occupies top ranks. Planes 0–5 always describe the side to move.

**Planes 12–95: Past positions** (7 history slots × 12 piece planes)
Each historical position fills the next 12-plane slot in temporal order (oldest at the highest planes; slot `i = 1..=7` occupies planes `(1+i)*12 .. (1+i)*12 + 12`). Same current-player perspective + rank-mirroring rules apply within each slot. Empty history slots are zero-filled.

**Planes 96–99: Castling rights** (4 binary planes, current-player perspective)
- Planes 96–97: Current player's kingside / queenside
- Planes 98–99: Opponent's kingside / queenside

Each plane is a constant fill (all 1s if the right is available, all 0s otherwise). Color flip swaps 96↔98 and 97↔99 (no rank mirror needed because constants).

**Plane 100: En passant target square** (one-hot, current position only). Rank-mirrored under color flip.

**Plane 101: Halfmove clock**, normalized as `clock / 100.0`, broadcast across all 64 squares. Unchanged under color flip.

**Side-to-move is NOT a plane.** It was removed in Phase 3b. Color information is implicit in the perspective convention (the network always sees "my" pieces in planes 0–5 of each slot). The `white_to_move` bool on `BoardObservation` is metadata used for action flipping at the MCTS boundary, not a network input.

## AlphaZero Perspective Convention (commit bb39db6)

**Key principle**: The observation always encodes from the side-to-move's perspective.

- **Input to h-network**: `[B, 102, 8, 8]` — board from current player's view
- **Output from f-network**: Policy `[B, 4096]` and value `[B]` in current-player space
- **MCTS action space**: Actions encoded relative to current player (0–4095 for base actions; 4096–4671 for underpromotions)

**Rank mirroring (for Black-to-move)**:
- Each 12-plane piece slot is flipped so the player to move appears in planes 0–5 within the slot
- Each 64-element plane block is rotated 180° so the side-to-move's back rank appears at the bottom
- En passant, castling, promotion targets all adjusted to match the rotated frame

## Action Encoding & Flipping

Action space: 4672 (4096 base + 576 underpromotion). Base actions are `from_sq * 64 + to_sq` (queen-default promotion). Underpromotion offsets at 4096–4671 cover knight/bishop/rook promotions across the 24 forward-pawn slots × 8 from-files × 4 underpromo pieces.

**At the MCTS boundary** (`game_task.rs`):
1. Inference receives current-player-perspective observation
2. Network outputs policy logits for current-player actions
3. Action selected via MCTS visit counts
4. **Action is flipped** to absolute board space before applying to game board
5. Board state moves to opponent; next observation flipped to the new current-player perspective

**`flip_action(action, color)`**: Transforms an action from current-player space to White-absolute space. If `color == White`, action unchanged. If `color == Black`, both from-square and to-square are rotated 180° (`from_sq → 63 − from_sq`, `to_sq → 63 − to_sq`).

**Action ordering and POV symmetry (commit 41f6681)**: `get_legal_moves()` returns actions in absolute-square iteration order, NOT in POV-symmetric order. The same move (e.g. Nc3 for White, Nc6 for Black) appears at different indices. To present an action list with identical index geometry on mirror-equivalent positions, callers must `legal_actions.sort_unstable()` after the POV-flip step. MCTS and inference depend on consistent action indexing; verify symmetry via `test_legal_actions_ordering_is_color_symmetric_after_sort()`.

## Representation Consistency Invariants

When implementing board transformations under color augmentation, verify these invariants to catch asymmetries:

1. **Action-spatial encode invariant**: `encode_action_spatial(flip_action(a), white_to_move=False) == flip_action_planes(encode_action_spatial(a, white_to_move=True))`
   - Ensures action encodings are symmetric under color flip
   - Particularly important for underpromotion, which has color-agnostic action IDs but color-specific output planes
   - Regression: `test_encode_action_spatial_under_color_flip` (commit cc58506)

2. **Observation-flip invariant**: `encode_obs_planes(flip_board(b)) == flip_action_planes(encode_obs_planes(b))`
   - Observation encoding transforms consistently with action encoding

3. **Legal-mask consistency**: Legal-move mask construction must account for color perspective changes. Always call `get_legal_moves()` *after* perspective flips, or verify masks are color-independent.

**Bug fixed (commit cc58506)**: Underpromo action indices (4096–4671) are color-agnostic IDs, but `encode_action_spatial(action, white_to_move)` returns color-specific spatial planes (ranks 6→7 for White, ranks 1→0 for Black). Under color augmentation, this broke the encode-flip invariant for all 576 underpromotion actions. Fix: use `encode_action_spatial_for_color(action, white_to_move)` when color context matters.

## Encoder Validation

Python-side encoder available at `scripts/reward_probe.py` with helper functions `encode_board_python()` and `encode_action_spatial()`. Used for diagnostic probes on held-out checkpoints to validate distributional collapse (2026-04-21 session). Validated byte-identical to Rust encoder on initial position via trainer's `[start_value]` probe.

## Related

- [Neural Networks](neural-networks.md) — h-network input shape `[B, 102, 8, 8]` and downstream tensor flow
- [MCTS & Self-Play](mcts-selfplay.md) — action flipping at MCTS boundary, legal-actions ordering
- `src/data/encoding.rs` — `encode_obs`, `encode_action_spatial_for_color`, `flip_action_planes`
- `src/data/types.rs` — `BoardObservation`, `NUM_OBS_PLANES = 102`
- `docs/wiki/mistakes.md` — encoding asymmetry bugs and fixes (2026-04-17, 2026-04-20 entries)
