"""Python board encoder mirroring src/data/encoding.rs.

Piece-type plane index order matches pieces_bb in src/game/playerobj.rs:
  0=Pawn, 1=Knight, 2=Bishop, 3=Rook, 4=Queen, 5=King

Plane layout (110 planes, [110, 8, 8] float32):
  Groups 0-7 (12 planes each, 96 planes total): position history slots.
    Within each group:
      planes 0-5:  current player's pieces (Pawn, Knight, Bishop, Rook, Queen, King)
      planes 6-11: opponent's pieces (same order)
    Group 0 = current position; groups 1-7 = history (oldest first, zeros if absent).
  Planes 96-99: castling rights (current ks, current qs, opp ks, opp qs).
    Each is a constant 1.0 fill if the right exists, 0.0 otherwise.
  Plane 100: en passant target square (one-hot in rank-mirrored coords for Black).
  Plane 101: halfmove clock / 100.0 (fills entire 8x8 plane).
  Planes 102-109: lc0-style repetition flags, one per history position (constant
    1.0 fill if that position had occurred before in the game, else 0.0). Standalone
    encodes (e.g. TB samples) have no game history, so these stay all zeros.

All squares are encoded with current-player perspective (AlphaZero convention):
  - White to move: square indices are unchanged (rank 0 = rank 1 in chess notation).
  - Black to move: square indices are rank-mirrored (sq -> 63 - sq, i.e. (7-rank)*8+file).
"""

from __future__ import annotations

import numpy as np
import chess


# Number of observation planes and actions — must match Rust constants.
NUM_OBS_PLANES = 110
NUM_BASE_ACTIONS = 4096      # from_sq * 64 + to_sq
NUM_UNDERPROMO_ACTIONS = 576 # 3 pieces * 8 from_files * 24 slots
NUM_ACTIONS = NUM_BASE_ACTIONS + NUM_UNDERPROMO_ACTIONS  # 4672


# ─── Square helpers ───────────────────────────────────────────────────────────

def _flip_sq(sq: int) -> int:
    """Rank-mirror a square index: rank 0 <-> rank 7.

    sq = rank * 8 + file; flipped = (7 - rank) * 8 + file = 63 - sq + (file - (7-file))
    Equivalently: (7 - sq // 8) * 8 + (sq % 8).
    """
    return (7 - sq // 8) * 8 + (sq % 8)


def _encode_sq(sq: int, is_black: bool) -> int:
    """Return the observation-space square index for a board square.

    For Black to move, rank-mirrors the square so piece positions are
    from the current player's perspective.
    """
    return _flip_sq(sq) if is_black else sq


# ─── Piece type mapping ───────────────────────────────────────────────────────

# Map python-chess PieceType to hyzero plane index (0-5).
_PIECE_PLANE: dict[chess.PieceType, int] = {
    chess.PAWN:   0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK:   3,
    chess.QUEEN:  4,
    chess.KING:   5,
}


# ─── Board encoder ────────────────────────────────────────────────────────────

def encode_board_python(board: chess.Board) -> np.ndarray:
    """Encode a chess.Board into a [110, 8, 8] float32 observation.

    Mirrors src/data/encoding.rs::encode_board with side-to-move perspective
    (AlphaZero convention). History slots (groups 1-7) are all zeros — TB samples
    have no history. Repetition planes 102-109 likewise stay zero for standalone
    encodes (no game history to detect a repeat).

    Args:
        board: python-chess Board to encode.

    Returns:
        np.ndarray of shape [110, 8, 8] and dtype float32.
    """
    obs = np.zeros((NUM_OBS_PLANES, 8, 8), dtype=np.float32)
    is_black = (board.turn == chess.BLACK)

    my_color = board.turn
    opp_color = not my_color  # chess.WHITE ^ True == chess.BLACK, etc.

    # Group 0: current position (planes 0-11).
    _encode_position_group(obs, board, is_black, my_color, opp_color, group=0)

    # Groups 1-7: history — zeros (no history for TB samples).

    # Planes 96-99: castling rights.
    # Current player: ks=96, qs=97; opponent: ks=98, qs=99.
    if is_black:
        # Current player is Black.
        if board.has_kingside_castling_rights(chess.BLACK):
            obs[96, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK):
            obs[97, :, :] = 1.0
        if board.has_kingside_castling_rights(chess.WHITE):
            obs[98, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE):
            obs[99, :, :] = 1.0
    else:
        # Current player is White.
        if board.has_kingside_castling_rights(chess.WHITE):
            obs[96, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE):
            obs[97, :, :] = 1.0
        if board.has_kingside_castling_rights(chess.BLACK):
            obs[98, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK):
            obs[99, :, :] = 1.0

    # Plane 100: en passant target square (one-hot, rank-mirrored for Black).
    if board.ep_square is not None:
        ep_sq = _encode_sq(board.ep_square, is_black)
        ep_rank = ep_sq // 8
        ep_file = ep_sq % 8
        obs[100, ep_rank, ep_file] = 1.0

    # Plane 101: halfmove clock / 100.0 (fills entire 8x8 plane).
    clock_val = board.halfmove_clock / 100.0
    obs[101, :, :] = clock_val

    return obs


def _encode_position_group(
    obs: np.ndarray,
    board: chess.Board,
    is_black: bool,
    my_color: chess.Color,
    opp_color: chess.Color,
    group: int,
) -> None:
    """Fill a 12-plane history group in obs for the given board position.

    Planes group*12 + 0..5: current player's pieces.
    Planes group*12 + 6..11: opponent's pieces.
    """
    base = group * 12

    # Current player's pieces (planes base+0 .. base+5).
    for piece_type, plane_idx in _PIECE_PLANE.items():
        bb = board.pieces(piece_type, my_color)
        for sq in bb:
            esq = _encode_sq(sq, is_black)
            rank = esq // 8
            file = esq % 8
            obs[base + plane_idx, rank, file] = 1.0

    # Opponent's pieces (planes base+6 .. base+11).
    for piece_type, plane_idx in _PIECE_PLANE.items():
        bb = board.pieces(piece_type, opp_color)
        for sq in bb:
            esq = _encode_sq(sq, is_black)
            rank = esq // 8
            file = esq % 8
            obs[base + 6 + plane_idx, rank, file] = 1.0


# ─── Action encoder ───────────────────────────────────────────────────────────

def encode_action_spatial(action: int, white_to_move: bool) -> np.ndarray:
    """Encode an action index as 3 spatial planes [3, 8, 8] float32.

    Mirrors src/data/encoding.rs::encode_action_spatial_for_color.

    Plane 0: from-square one-hot.
    Plane 1: to-square one-hot.
    Plane 2: promotion flag (all 1.0 if promotion, all 0.0 otherwise).

    For underpromotion actions (>= NUM_BASE_ACTIONS), from/to squares are derived
    from the encoded from_file and to_file, using the correct promotion ranks for
    the given POV color. For base actions the from/to squares are explicit in the
    index. Queen promotions use the base encoding (no promotion flag set — mirrors Rust).

    Args:
        action:        Action index in [0, NUM_ACTIONS).
        white_to_move: True if encoding from White's POV.

    Returns:
        np.ndarray of shape [3, 8, 8] and dtype float32.
    """
    planes = np.zeros((3, 8, 8), dtype=np.float32)

    if action >= NUM_BASE_ACTIONS:
        # Underpromotion action: decode file indices.
        offset = action - NUM_BASE_ACTIONS
        piece_idx = offset // 192
        remainder = offset % 192
        from_file = remainder // 24
        to_file = remainder % 24  # Rust stores to_file directly (0-7); slots >= 8 are illegal

        if from_file < 8 and to_file < 8:
            # Use promotion ranks matching the POV color.
            if white_to_move:
                from_rank, to_rank = 6, 7
            else:
                from_rank, to_rank = 1, 0
            from_sq = from_rank * 8 + from_file
            to_sq = to_rank * 8 + to_file
            planes[0, from_sq // 8, from_sq % 8] = 1.0
            planes[1, to_sq // 8, to_sq % 8] = 1.0
        # Promotion flag always set for underpromo actions.
        planes[2, :, :] = 1.0
    else:
        # Base action: from_sq and to_sq are explicit.
        from_sq = action // 64
        to_sq = action % 64
        planes[0, from_sq // 8, from_sq % 8] = 1.0
        planes[1, to_sq // 8, to_sq % 8] = 1.0
        # No promotion flag for base actions (mirrors Rust — queen promos not flagged here).

    return planes


# ─── Move <-> action conversion ───────────────────────────────────────────────

def action_from_move(move: chess.Move, board: chess.Board) -> int:
    """Convert a python-chess Move to the hyzero 4672-action integer.

    Mirrors src/data/encoding.rs::move_to_action:
    - Queen promotions and non-promotion moves: from_sq * 64 + to_sq (base range 0..4095).
    - Knight/Bishop/Rook promotions: underpromotion encoding in range 4096..4671.

    Args:
        move:  python-chess Move object.
        board: Board context (used to determine color for correct EP / promo detection,
               but the action encoding itself only uses from/to squares + promo piece).

    Returns:
        Integer action index in [0, 4672).
    """
    from_sq = move.from_square   # 0-63
    to_sq = move.to_square       # 0-63
    promo = move.promotion       # None or chess.PieceType

    if promo is not None and promo in (chess.KNIGHT, chess.BISHOP, chess.ROOK):
        # Underpromotion encoding: piece_idx * 192 + from_file * 24 + to_file
        piece_idx = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}[promo]
        from_file = from_sq % 8
        to_file = to_sq % 8
        return NUM_BASE_ACTIONS + piece_idx * 192 + from_file * 24 + to_file
    else:
        # Base encoding (queen promotions and all non-promotion moves).
        return from_sq * 64 + to_sq
