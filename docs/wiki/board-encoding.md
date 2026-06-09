# Board Encoding & Observation Format

The board observation tensor encodes the game state from the current player's
perspective, following the AlphaZero convention. This makes the network learn
symmetric value estimates and consistent policy targets regardless of which side
is to move. See [Neural Networks](neural-networks.md) for the downstream network
input shape; this page is the source of truth for plane definitions.

## Observation Planes (102 total)

`NUM_OBS_PLANES = 102` (defined in `src/data/types.rs`). Tensor shape is `[B, 102, 8, 8]` — 96 piece planes covering the current position plus 7 historical positions, then 6 game-state planes. The Python mirror is documented in `python/hyzero/data/board_encoder.py`.

**Planes 0–11: Current position pieces** (12 planes, current-player perspective)
- Planes 0–5: Current player's pieces — Pawn, Knight, Bishop, Rook, Queen, King
- Planes 6–11: Opponent's pieces — same order

For Black-to-move, planes are rank-mirrored so the current player's pieces always occupy the bottom ranks from the network's perspective. Planes 0–5 always describe the side to move.

**Planes 12–95: Past positions** (7 history slots × 12 piece planes). Each historical position fills the next 12-plane slot in temporal order (oldest first; empty slots are zero-filled). Same perspective + rank-mirroring rules apply within each slot.

**Planes 96–99: Castling rights** (4 binary planes, current-player perspective)
- Planes 96–97: Current player's kingside / queenside
- Planes 98–99: Opponent's kingside / queenside

Each is a constant fill (all 1s if available, all 0s otherwise). Color flip swaps 96↔98 and 97↔99 (no rank mirror needed for constants).

**Plane 100: En passant target** (one-hot, current position only). Rank-mirrored under color flip.

**Plane 101: Halfmove clock**, normalized as `clock / 100.0`, broadcast across all 64 squares. Unchanged under color flip.

**Side-to-move is NOT a plane.** It was removed in Phase 3b; color is implicit in the perspective convention (the network always sees "my" pieces in planes 0–5 of each slot). The `white_to_move` bool on `StepRecord` / `BoardObservation` is metadata used for action flipping at the MCTS boundary and value-target ply-flipping in the trainer — not a network input.

## AlphaZero Perspective Convention

**Key principle**: The observation always encodes from the side-to-move's perspective.

- **Input to h-network**: `[B, 102, 8, 8]` — board from the current player's view
- **Output from f-network**: Policy `[B, 4672]` and value `[B]` in current-player space
- **MCTS action space**: Actions encoded relative to the current player

**Rank mirroring (Black-to-move)**:
- Each 12-plane piece slot is flipped so the player to move appears in planes 0–5
- Each 64-element plane block is rotated 180° so the side-to-move's back rank is at the bottom
- En passant, castling, and promotion targets are adjusted to match the rotated frame

## Action Encoding & Flipping

Action space: **4672** (`NUM_ACTIONS` in `src/data/types.rs`) = 4096 base (`NUM_BASE_ACTIONS`) + 576 underpromotion (`NUM_UNDERPROMO_ACTIONS`). Base actions are `from_sq * 64 + to_sq` (queen-default promotion). Underpromotion offsets at 4096–4671 cover knight/bishop/rook promotions: `4096 + piece_idx * 192 + from_file * 24 + ...` (piece_idx 0=Knight, 1=Bishop, 2=Rook).

**At the MCTS boundary** (`src/selfplay/game_task.rs`):
1. Inference receives the current-player-perspective observation
2. Network outputs policy logits for current-player actions
3. Action selected via MCTS visit counts
4. **Action is flipped** to absolute board space before applying to the board (`action_to_move(action, board, color)`)
5. Board state moves to the opponent; the next observation is flipped to the new current-player perspective

**`flip_action(action)`**: Rotates from/to squares 180° (`sq → 63 − sq`) for the base range, with color-specific handling for the underpromotion range. If the position is White-to-move, the action is effectively unchanged; for Black-to-move both squares are mirrored.

**Action ordering and POV symmetry**: `get_legal_moves()` returns actions in absolute-square iteration order, NOT POV-symmetric order. The same move (e.g. Nc3 for White, Nc6 for Black) appears at different indices. To present an action list with identical index geometry on mirror-equivalent positions, callers must `legal_actions.sort_unstable()` after the POV-flip step. MCTS and inference depend on this consistent indexing (see [MCTS](mcts.md) selection mechanics).

## Representation Consistency Invariants

When implementing board transforms under color augmentation, verify:

1. **Action-spatial encode invariant**: `encode_action_spatial_for_color(flip_action(a), !c) == flip_action_planes(encode_action_spatial_for_color(a, c))` for all actions `a` and both colors `c`. Underpromotion indices are color-agnostic at the action-ID level, but `encode_action_spatial_for_color` emits color-specific spatial planes (promotion ranks 6→7 for White, 1→0 for Black), so the POV color must be threaded through whenever it matters.
2. **Observation-flip invariant**: `flip_obs_planes(encode_board(b, c, hist)) == encode_board(b, !c, hist)` — flipping the observation planes (a plane involution that swaps my/opp piece groups and rank-mirrors, swaps castling planes 96↔98 / 97↔99, rank-mirrors the en-passant plane, and leaves the halfmove-clock plane unchanged) is equivalent to re-encoding the same board from the opposite color's perspective. `flip_obs_planes` is NOT a board flip; it operates purely on the encoded plane tensor.
3. **Legal-mask consistency**: build the legal-move mask *after* perspective flips, or verify it is color-independent.

The trainer (`src/py/training.rs::assemble_batch_arrays`) applies color augmentation per sample, gated by `HYZERO_DISABLE_COLOR_AUG`. When flipping, it negates `root_value`/`reward` targets and flips each action index before scattering into `target_policies` / `legal_masks`.

## Related

- [Neural Networks](neural-networks.md) — h-network input `[B, 102, 8, 8]`, policy output `[B, 4672]`
- [MCTS](mcts.md) — action flipping, legal-action ordering
- `src/data/encoding.rs` — `encode_board`, `flip_obs_planes`, `encode_action_spatial_for_color`, `flip_action_planes`, `action_to_move`
- `src/data/types.rs` — `BoardObservation`, `NUM_OBS_PLANES = 102`, `NUM_ACTIONS = 4672`
- `python/hyzero/data/board_encoder.py` — Python mirror of the encoder
