#!/usr/bin/env python3
"""Build a Syzygy tablebase position cache for training supervision.

Usage:
    HYZERO_TABLEBASE_PATH=/path/to/syzygy python3 scripts/build_tablebase_cache.py

Environment variables:
    HYZERO_TABLEBASE_PATH:       Directory containing .rtbw/.rtbz files (required).
    HYZERO_TABLEBASE_CACHE_PATH: Output path for pickle. Default: data/syzygy/cache.pkl.
    HYZERO_TB_N_TOTAL:           Total positions to generate. Default: 500000.

Output: pickled list[TBSample] at HYZERO_TABLEBASE_CACHE_PATH.

Endgame classes and target counts (total ~500k):
    KQK: 80k, KRK: 80k, KBBK: 40k, KBNK: 40k, KPK: 80k
    KRKP: 60k, KQKR: 60k, KRKR: 60k

Each position is probed for WDL + DTZ + mating moves. Positions where the
tablebase probe fails (MissingTableError or invalid position) are skipped.
"""

from __future__ import annotations

import os
import sys
import pickle
import random
import math
from dataclasses import dataclass

# Ensure hyzero package is importable when run from repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import chess
import chess.syzygy

from hyzero.data.board_encoder import action_from_move


# ─── Config ───────────────────────────────────────────────────────────────────

TB_PATH = os.environ.get("HYZERO_TABLEBASE_PATH")
CACHE_PATH = os.environ.get("HYZERO_TABLEBASE_CACHE_PATH", "data/syzygy/cache.pkl")
N_TOTAL = int(os.environ.get("HYZERO_TB_N_TOTAL", "500000"))

# Target counts per endgame class (sum = N_TOTAL default).
ENDGAME_CLASSES: list[tuple[str, int]] = [
    ("KQK",  80_000),
    ("KRK",  80_000),
    ("KBBK", 40_000),
    ("KBNK", 40_000),
    ("KPK",  80_000),
    ("KRKP", 60_000),
    ("KQKR", 60_000),
    ("KRKR", 60_000),
]


# ─── TBSample dataclass (must match python/hyzero/data/tablebase.py) ─────────

@dataclass
class TBSample:
    fen: str                        # Position FEN.
    target_value: float             # +1, 0, or -1 from side-to-move POV.
    mating_actions: list[int]       # Action indices of mate-in-1 moves (may be empty).
    optimal_actions: list[int]      # Action indices of optimal-DTZ moves.
    all_legal_actions: list[int]    # All legal action indices.


# ─── Position generators ──────────────────────────────────────────────────────

def _king_distance(sq1: int, sq2: int) -> int:
    """Chebyshev distance between two squares."""
    r1, f1 = sq1 // 8, sq1 % 8
    r2, f2 = sq2 // 8, sq2 % 8
    return max(abs(r1 - r2), abs(f1 - f2))


def _place_kings() -> tuple[int, int]:
    """Place two kings on non-adjacent squares (Chebyshev distance > 1)."""
    while True:
        k1 = random.randint(0, 63)
        k2 = random.randint(0, 63)
        if k1 != k2 and _king_distance(k1, k2) > 1:
            return k1, k2


def _generate_kqk(tb: chess.syzygy.Tablebase, n: int) -> list[TBSample]:
    """Generate KQK positions (white has Q, black just has K)."""
    samples: list[TBSample] = []
    attempts = 0
    max_attempts = n * 20
    while len(samples) < n and attempts < max_attempts:
        attempts += 1
        wk, bk = _place_kings()
        # Place queen on any non-king square.
        empty = [s for s in range(64) if s != wk and s != bk]
        if not empty:
            continue
        wq = random.choice(empty)
        for turn in (chess.WHITE, chess.BLACK):
            board = chess.Board(None)
            board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
            board.set_piece_at(wq, chess.Piece(chess.QUEEN, chess.WHITE))
            board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
            board.turn = turn
            board.castling_rights = 0
            sample = _probe_position(tb, board)
            if sample is not None:
                samples.append(sample)
                if len(samples) >= n:
                    break
    return samples


def _generate_krk(tb: chess.syzygy.Tablebase, n: int) -> list[TBSample]:
    """Generate KRK positions."""
    samples: list[TBSample] = []
    attempts = 0
    max_attempts = n * 20
    while len(samples) < n and attempts < max_attempts:
        attempts += 1
        wk, bk = _place_kings()
        empty = [s for s in range(64) if s != wk and s != bk]
        if not empty:
            continue
        wr = random.choice(empty)
        for turn in (chess.WHITE, chess.BLACK):
            board = chess.Board(None)
            board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
            board.set_piece_at(wr, chess.Piece(chess.ROOK, chess.WHITE))
            board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
            board.turn = turn
            board.castling_rights = 0
            sample = _probe_position(tb, board)
            if sample is not None:
                samples.append(sample)
                if len(samples) >= n:
                    break
    return samples


def _generate_kbbk(tb: chess.syzygy.Tablebase, n: int) -> list[TBSample]:
    """Generate KBBK positions (two bishops, need on different colors for mate)."""
    samples: list[TBSample] = []
    attempts = 0
    max_attempts = n * 30
    while len(samples) < n and attempts < max_attempts:
        attempts += 1
        wk, bk = _place_kings()
        empty = [s for s in range(64) if s != wk and s != bk]
        if len(empty) < 2:
            continue
        # Pick two bishops on different colored squares.
        bishop_sqs = random.sample(empty, 2)
        if (bishop_sqs[0] + bishop_sqs[0] // 8) % 2 == (bishop_sqs[1] + bishop_sqs[1] // 8) % 2:
            continue  # Same color — no mate possible, skip
        for turn in (chess.WHITE, chess.BLACK):
            board = chess.Board(None)
            board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
            board.set_piece_at(bishop_sqs[0], chess.Piece(chess.BISHOP, chess.WHITE))
            board.set_piece_at(bishop_sqs[1], chess.Piece(chess.BISHOP, chess.WHITE))
            board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
            board.turn = turn
            board.castling_rights = 0
            sample = _probe_position(tb, board)
            if sample is not None:
                samples.append(sample)
                if len(samples) >= n:
                    break
    return samples


def _generate_kbnk(tb: chess.syzygy.Tablebase, n: int) -> list[TBSample]:
    """Generate KBNK positions."""
    samples: list[TBSample] = []
    attempts = 0
    max_attempts = n * 30
    while len(samples) < n and attempts < max_attempts:
        attempts += 1
        wk, bk = _place_kings()
        empty = [s for s in range(64) if s != wk and s != bk]
        if len(empty) < 2:
            continue
        piece_sqs = random.sample(empty, 2)
        for turn in (chess.WHITE, chess.BLACK):
            board = chess.Board(None)
            board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
            board.set_piece_at(piece_sqs[0], chess.Piece(chess.BISHOP, chess.WHITE))
            board.set_piece_at(piece_sqs[1], chess.Piece(chess.KNIGHT, chess.WHITE))
            board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
            board.turn = turn
            board.castling_rights = 0
            sample = _probe_position(tb, board)
            if sample is not None:
                samples.append(sample)
                if len(samples) >= n:
                    break
    return samples


def _generate_kpk(tb: chess.syzygy.Tablebase, n: int) -> list[TBSample]:
    """Generate KPK positions. Pawns cannot be on rank 0 or rank 7."""
    samples: list[TBSample] = []
    attempts = 0
    max_attempts = n * 20
    # Valid pawn squares: ranks 1-6 (not rank 0 or 7 to avoid auto-promotion complications)
    pawn_squares = [s for s in range(8, 56)]  # ranks 1-6
    while len(samples) < n and attempts < max_attempts:
        attempts += 1
        wk, bk = _place_kings()
        # Pick a pawn square not occupied by kings.
        valid_pawn = [s for s in pawn_squares if s != wk and s != bk]
        if not valid_pawn:
            continue
        wp = random.choice(valid_pawn)
        for turn in (chess.WHITE, chess.BLACK):
            board = chess.Board(None)
            board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
            board.set_piece_at(wp, chess.Piece(chess.PAWN, chess.WHITE))
            board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
            board.turn = turn
            board.castling_rights = 0
            sample = _probe_position(tb, board)
            if sample is not None:
                samples.append(sample)
                if len(samples) >= n:
                    break
    return samples


def _generate_krkp(tb: chess.syzygy.Tablebase, n: int) -> list[TBSample]:
    """Generate KRKP positions (white has K+R, black has K+P)."""
    samples: list[TBSample] = []
    attempts = 0
    max_attempts = n * 20
    pawn_squares = [s for s in range(8, 56)]
    while len(samples) < n and attempts < max_attempts:
        attempts += 1
        wk, bk = _place_kings()
        occupied = {wk, bk}
        empty = [s for s in range(64) if s not in occupied]
        if not empty:
            continue
        wr = random.choice(empty)
        occupied.add(wr)
        valid_pawn = [s for s in pawn_squares if s not in occupied]
        if not valid_pawn:
            continue
        bp = random.choice(valid_pawn)
        for turn in (chess.WHITE, chess.BLACK):
            board = chess.Board(None)
            board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
            board.set_piece_at(wr, chess.Piece(chess.ROOK, chess.WHITE))
            board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
            board.set_piece_at(bp, chess.Piece(chess.PAWN, chess.BLACK))
            board.turn = turn
            board.castling_rights = 0
            sample = _probe_position(tb, board)
            if sample is not None:
                samples.append(sample)
                if len(samples) >= n:
                    break
    return samples


def _generate_kqkr(tb: chess.syzygy.Tablebase, n: int) -> list[TBSample]:
    """Generate KQKR positions."""
    samples: list[TBSample] = []
    attempts = 0
    max_attempts = n * 20
    while len(samples) < n and attempts < max_attempts:
        attempts += 1
        wk, bk = _place_kings()
        occupied = {wk, bk}
        empty = [s for s in range(64) if s not in occupied]
        if len(empty) < 2:
            continue
        piece_sqs = random.sample(empty, 2)
        wq, br = piece_sqs[0], piece_sqs[1]
        for turn in (chess.WHITE, chess.BLACK):
            board = chess.Board(None)
            board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
            board.set_piece_at(wq, chess.Piece(chess.QUEEN, chess.WHITE))
            board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
            board.set_piece_at(br, chess.Piece(chess.ROOK, chess.BLACK))
            board.turn = turn
            board.castling_rights = 0
            sample = _probe_position(tb, board)
            if sample is not None:
                samples.append(sample)
                if len(samples) >= n:
                    break
    return samples


def _generate_krkr(tb: chess.syzygy.Tablebase, n: int) -> list[TBSample]:
    """Generate KRKR positions."""
    samples: list[TBSample] = []
    attempts = 0
    max_attempts = n * 20
    while len(samples) < n and attempts < max_attempts:
        attempts += 1
        wk, bk = _place_kings()
        occupied = {wk, bk}
        empty = [s for s in range(64) if s not in occupied]
        if len(empty) < 2:
            continue
        piece_sqs = random.sample(empty, 2)
        wr, br = piece_sqs[0], piece_sqs[1]
        for turn in (chess.WHITE, chess.BLACK):
            board = chess.Board(None)
            board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
            board.set_piece_at(wr, chess.Piece(chess.ROOK, chess.WHITE))
            board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
            board.set_piece_at(br, chess.Piece(chess.ROOK, chess.BLACK))
            board.turn = turn
            board.castling_rights = 0
            sample = _probe_position(tb, board)
            if sample is not None:
                samples.append(sample)
                if len(samples) >= n:
                    break
    return samples


_GENERATORS = {
    "KQK":  _generate_kqk,
    "KRK":  _generate_krk,
    "KBBK": _generate_kbbk,
    "KBNK": _generate_kbnk,
    "KPK":  _generate_kpk,
    "KRKP": _generate_krkp,
    "KQKR": _generate_kqkr,
    "KRKR": _generate_krkr,
}


# ─── TB probe ─────────────────────────────────────────────────────────────────

def _probe_position(tb: chess.syzygy.Tablebase, board: chess.Board) -> TBSample | None:
    """Probe a position; return TBSample or None if invalid/unprobeble.

    Validation checks:
    - board.is_valid(): basic legality (no pawns on rank 0/7, etc.)
    - Side not to move is not in check (would be illegal).
    - Both sides have exactly one king.
    """
    if not board.is_valid():
        return None

    # Reject if side NOT to move is in check (the position would be illegal after the
    # prior move left their king in check).
    board_copy = board.copy()
    board_copy.turn = not board.turn
    if board_copy.is_check():
        return None

    # Collect legal moves.
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None  # Stalemate or checkmate — skip.

    all_legal_actions = [action_from_move(m, board) for m in legal_moves]

    # Probe WDL.
    try:
        wdl = tb.probe_wdl(board)
    except Exception:
        return None

    target_value = 1.0 if wdl > 0 else (-1.0 if wdl < 0 else 0.0)

    # Find mating moves (mate-in-1): gives check AND results in checkmate.
    mating_actions: list[int] = []
    for move in legal_moves:
        if board.gives_check(move):
            board.push(move)
            if board.is_checkmate():
                mating_actions.append(action_from_move(move, board))  # Note: push, then encode
            board.pop()

    # Re-encode mating actions with the un-pushed board (from-position perspective).
    mating_actions = []
    for move in legal_moves:
        if board.gives_check(move):
            board.push(move)
            if board.is_checkmate():
                board.pop()
                mating_actions.append(action_from_move(move, board))
            else:
                board.pop()

    # Probe DTZ for optimal move policy.
    optimal_actions: list[int] = []
    try:
        # For each legal move, probe DTZ after pushing.
        dtz_after: list[tuple[int, chess.Move]] = []
        for move in legal_moves:
            board.push(move)
            try:
                dtz_val = tb.probe_dtz(board)
                dtz_after.append((abs(dtz_val), move))
            except Exception:
                dtz_after.append((999999, move))  # Large fallback
            board.pop()

        if dtz_after:
            min_dtz = min(d for d, _ in dtz_after)
            optimal_moves = [m for d, m in dtz_after if d == min_dtz]
            optimal_actions = [action_from_move(m, board) for m in optimal_moves]
    except Exception:
        # Fall back to uniform over all legal moves.
        optimal_actions = all_legal_actions[:]

    if not optimal_actions:
        optimal_actions = all_legal_actions[:]

    fen = board.fen()
    return TBSample(
        fen=fen,
        target_value=target_value,
        mating_actions=mating_actions,
        optimal_actions=optimal_actions,
        all_legal_actions=all_legal_actions,
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    if TB_PATH is None:
        print("ERROR: HYZERO_TABLEBASE_PATH must be set", file=sys.stderr)
        sys.exit(1)

    if not os.path.isdir(TB_PATH):
        print(f"ERROR: HYZERO_TABLEBASE_PATH={TB_PATH!r} is not a directory", file=sys.stderr)
        sys.exit(1)

    print(f"[build_cache] Opening tablebase at {TB_PATH!r}")
    try:
        tb = chess.syzygy.open_tablebase(TB_PATH)
    except Exception as e:
        print(f"ERROR: Failed to open tablebase: {e}", file=sys.stderr)
        sys.exit(1)

    # Scale target counts if N_TOTAL != default sum (500k).
    default_total = sum(n for _, n in ENDGAME_CLASSES)
    scale = N_TOTAL / default_total
    scaled_classes = [(name, max(1, int(n * scale))) for name, n in ENDGAME_CLASSES]
    print(f"[build_cache] Target: {N_TOTAL} positions total (scale={scale:.2f})")

    all_samples: list[TBSample] = []
    for name, target_n in scaled_classes:
        gen_fn = _GENERATORS[name]
        print(f"[build_cache] Generating {target_n} positions for {name} ...", flush=True)
        samples = gen_fn(tb, target_n)
        print(f"[build_cache]   Got {len(samples)} samples for {name}")
        all_samples.extend(samples)

    print(f"[build_cache] Total samples: {len(all_samples)}")

    # Ensure output directory exists.
    cache_dir = os.path.dirname(CACHE_PATH)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    print(f"[build_cache] Writing cache to {CACHE_PATH!r} ...")
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(all_samples, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[build_cache] Done. Cache contains {len(all_samples)} positions.")


if __name__ == "__main__":
    main()
