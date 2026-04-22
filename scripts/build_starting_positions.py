#!/usr/bin/env python3
"""Build a diverse self-play starting-position dataset for hyzero.

Mix (default):
  30% standard opening   — the initial position (drives opening coverage).
  40% middlegame         — random-play positions at ply 15–25 with material
                            asymmetry |Δ| ≥ 2 pawn equivalents (forces decisive
                            play, gives the reward head terminal signal).
  30% endgame            — positions from the Syzygy trajectory cache with
                            7–12 pieces (games that start near trained TB
                            terrain and actually reach mate).

Output: data/starting_positions.txt — one FEN per line, shuffled. The Rust
self-play code loads this via HYZERO_STARTS_FILE.

Env vars:
  HYZERO_STARTS_OUTPUT     — output path (default: data/starting_positions.txt)
  HYZERO_STARTS_N          — total FEN count (default: 100000)
  HYZERO_STARTS_FRAC_START — opening fraction (default: 0.30)
  HYZERO_STARTS_FRAC_MID   — middlegame fraction (default: 0.40)
  HYZERO_STARTS_FRAC_END   — endgame fraction (default: 0.30)
  HYZERO_STARTS_TB_CACHE   — TB cache path (default: data/syzygy/cache_trajectories.pkl)
  HYZERO_STARTS_SEED       — PRNG seed (default: 42)
"""

from __future__ import annotations

import os
import sys
import pickle
import random
import time

import chess


OUTPUT = os.environ.get("HYZERO_STARTS_OUTPUT", "data/starting_positions.txt")
N_TOTAL = int(os.environ.get("HYZERO_STARTS_N", "100000"))
FRAC_START = float(os.environ.get("HYZERO_STARTS_FRAC_START", "0.30"))
FRAC_MID   = float(os.environ.get("HYZERO_STARTS_FRAC_MID",   "0.40"))
FRAC_END   = float(os.environ.get("HYZERO_STARTS_FRAC_END",   "0.30"))
TB_CACHE   = os.environ.get("HYZERO_STARTS_TB_CACHE", "data/syzygy/cache_trajectories.pkl")
SEED       = int(os.environ.get("HYZERO_STARTS_SEED", "42"))


# Standard piece-value table used to compute material asymmetry. King = 0 so it
# cancels out of the diff (same for both sides).
_PIECE_VALUE = {
    chess.PAWN:   1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK:   5,
    chess.QUEEN:  9,
    chess.KING:   0,
}


def _material_count(board: chess.Board, color: chess.Color) -> int:
    total = 0
    for piece_type, val in _PIECE_VALUE.items():
        total += val * len(board.pieces(piece_type, color))
    return total


def _total_pieces(board: chess.Board) -> int:
    return chess.popcount(board.occupied)


# ─── Opening (standard start) ─────────────────────────────────────────────────

def build_start_positions(n: int) -> list[str]:
    """Standard opening: just the initial position N times."""
    return [chess.STARTING_FEN] * n


# ─── Middlegame from random play + material filter ──────────────────────────

def build_middlegame_positions(n: int, max_attempts: int | None = None) -> list[str]:
    """Random-play to ply 15-25, accept if |Δmaterial| ≥ 2 and game not over.

    Pieces-per-side must be ≥ 7 so we don't generate endgame-like positions
    that would better belong to the endgame bucket.
    """
    if max_attempts is None:
        max_attempts = n * 50
    fens: list[str] = []
    attempts = 0
    while len(fens) < n and attempts < max_attempts:
        attempts += 1
        board = chess.Board()
        # Random depth in [15, 25].
        target_ply = random.randint(15, 25)
        ok = True
        for _ in range(target_ply):
            if board.is_game_over():
                ok = False
                break
            legal = list(board.legal_moves)
            if not legal:
                ok = False
                break
            board.push(random.choice(legal))
        if not ok or board.is_game_over():
            continue
        # Filters: material asymmetry ≥ 2 and both sides still have middlegame piece counts.
        w_mat = _material_count(board, chess.WHITE)
        b_mat = _material_count(board, chess.BLACK)
        if abs(w_mat - b_mat) < 2:
            continue
        total_pieces = _total_pieces(board)
        if total_pieces < 14:  # too few pieces — belongs in endgame bucket
            continue
        fens.append(board.fen())
    if len(fens) < n:
        print(f"[starts] WARN: only produced {len(fens)}/{n} middlegame positions "
              f"(max_attempts={max_attempts})", file=sys.stderr)
    return fens


# ─── Endgame from TB cache ───────────────────────────────────────────────────

def build_endgame_positions(n: int) -> list[str]:
    """Sample step-0 FENs from the TB trajectory cache, filter to 7-12 pieces."""
    if not os.path.exists(TB_CACHE):
        print(f"[starts] WARN: TB cache not found at {TB_CACHE!r}; "
              f"endgame bucket will be empty", file=sys.stderr)
        return []

    # Import-shim so the pickle resolves to TBTrajectory.
    import sys as _sys
    import types as _types
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
    from hyzero.data.tablebase import TBTrajectory
    _shim = _types.ModuleType("__main__")
    _shim.TBTrajectory = TBTrajectory
    _prev = _sys.modules.get("__main__")
    _sys.modules["__main__"] = _shim
    try:
        with open(TB_CACHE, "rb") as f:
            trajectories = pickle.load(f)
    finally:
        if _prev is not None:
            _sys.modules["__main__"] = _prev

    print(f"[starts] loaded {len(trajectories)} trajectories from TB cache", flush=True)

    # Sample step-0 FENs with piece-count filter.
    # Piece counts in the TB cache span 3-5 (KvK + at most 2 others). Relax the
    # filter floor to let small endgames through — those still need conversion.
    fens: list[str] = []
    random.shuffle(trajectories)
    for t in trajectories:
        if len(fens) >= n:
            break
        fen = t.fens[0]
        if fen is None:
            continue
        try:
            board = chess.Board(fen)
        except Exception:
            continue
        p_count = _total_pieces(board)
        if p_count < 3 or p_count > 12:
            continue
        fens.append(fen)

    if len(fens) < n:
        print(f"[starts] WARN: only produced {len(fens)}/{n} endgame positions "
              f"from TB cache", file=sys.stderr)
    return fens


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    random.seed(SEED)
    assert abs(FRAC_START + FRAC_MID + FRAC_END - 1.0) < 1e-6, \
        f"fractions must sum to 1.0, got {FRAC_START + FRAC_MID + FRAC_END}"

    n_start = int(N_TOTAL * FRAC_START)
    n_mid   = int(N_TOTAL * FRAC_MID)
    n_end   = N_TOTAL - n_start - n_mid

    print(f"[starts] target: {N_TOTAL} total "
          f"(start={n_start}, mid={n_mid}, end={n_end})", flush=True)

    t = time.time()
    print(f"[starts] building {n_start} opening positions ...", flush=True)
    start_fens = build_start_positions(n_start)
    print(f"[starts]   got {len(start_fens)} in {time.time()-t:.1f}s", flush=True)

    t = time.time()
    print(f"[starts] building {n_mid} middlegame positions (random-play + filter) ...", flush=True)
    mid_fens = build_middlegame_positions(n_mid)
    print(f"[starts]   got {len(mid_fens)} in {time.time()-t:.1f}s", flush=True)

    t = time.time()
    print(f"[starts] building {n_end} endgame positions (from TB cache) ...", flush=True)
    end_fens = build_endgame_positions(n_end)
    print(f"[starts]   got {len(end_fens)} in {time.time()-t:.1f}s", flush=True)

    all_fens = start_fens + mid_fens + end_fens
    random.shuffle(all_fens)
    print(f"[starts] writing {len(all_fens)} FENs to {OUTPUT!r} ...", flush=True)

    out_dir = os.path.dirname(OUTPUT)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(OUTPUT, "w") as f:
        for fen in all_fens:
            f.write(fen + "\n")

    size_kb = os.path.getsize(OUTPUT) / 1024
    print(f"[starts] done. {size_kb:.1f} KB", flush=True)


if __name__ == "__main__":
    main()
