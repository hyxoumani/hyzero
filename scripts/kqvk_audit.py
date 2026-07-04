#!/usr/bin/env python3
"""KQvK (K+Q vs K) conversion audit for self-play PGNs.

Scans a PGN file, selects games whose *starting* position has exactly K+Q vs K
material (either color), replays each to its final position, and classifies the
terminal state (checkmate / stalemate / insufficient material / repetition /
other). Emits a single JSON line summarizing conversion:

    {"kqvk_games": N, "mates": n1, "insufficient_material": n2,
     "repetition": n3, "stalemate": n4, "other": n5, "mate_rate": r,
     "valid": bool}

``valid`` is false when fewer than 30 KQvK games were found, signaling the
sample is too small for ``mate_rate`` to be meaningful.

Robust to interleaving corruption: games that fail to parse or replay (illegal
SAN, truncated move text) are skipped rather than aborting the whole scan.

Usage:
    python scripts/kqvk_audit.py logs/selfplay_sample.pgn
    python scripts/kqvk_audit.py logs/selfplay_sample.pgn --material KQ
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

import chess
import chess.pgn

# Letter -> piece type for the strong side's non-king material spec.
_PIECE_LETTER = {
    "Q": chess.QUEEN,
    "R": chess.ROOK,
    "B": chess.BISHOP,
    "N": chess.KNIGHT,
    "P": chess.PAWN,
}

_NON_KING = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)


def parse_material(spec: str) -> Counter:
    """Parse a material spec like ``"KQ"`` into the strong side's non-king pieces.

    The spec must contain exactly one ``K``; remaining letters are the strong
    side's extra material. Returns a ``Counter`` keyed by ``chess`` piece type.
    """
    letters = list(spec.strip().upper())
    if letters.count("K") != 1:
        raise ValueError(f"material spec must contain exactly one K: {spec!r}")
    extra = Counter()
    for c in letters:
        if c == "K":
            continue
        if c not in _PIECE_LETTER:
            raise ValueError(f"unknown piece letter {c!r} in material spec {spec!r}")
        extra[_PIECE_LETTER[c]] += 1
    return extra


def _non_king_counts(board: chess.Board, color: bool) -> Counter:
    counts = Counter()
    for pt in _NON_KING:
        n = len(board.pieces(pt, color))
        if n:
            counts[pt] = n
    return counts


def is_target_material(board: chess.Board, strong_extra: Counter) -> bool:
    """True if one side has K + ``strong_extra`` and the other has a bare king."""
    if len(board.pieces(chess.KING, chess.WHITE)) != 1:
        return False
    if len(board.pieces(chess.KING, chess.BLACK)) != 1:
        return False
    white = _non_king_counts(board, chess.WHITE)
    black = _non_king_counts(board, chess.BLACK)
    return (white == strong_extra and not black) or (
        black == strong_extra and not white
    )


def classify_terminal(board: chess.Board) -> str:
    """Classify the final board into one of the audit outcome buckets."""
    if board.is_checkmate():
        return "mates"
    if board.is_stalemate():
        return "stalemate"
    if board.is_insufficient_material():
        return "insufficient_material"
    if board.is_repetition(3):
        return "repetition"
    return "other"


def audit_pgn(path: str, strong_extra: Counter | None = None) -> dict:
    """Audit ``path`` and return the conversion-summary dict.

    Malformed games (parse errors or illegal moves) are skipped silently so a
    single corrupt game cannot abort the scan.
    """
    if strong_extra is None:
        strong_extra = parse_material("KQ")

    counts = {
        "kqvk_games": 0,
        "mates": 0,
        "insufficient_material": 0,
        "repetition": 0,
        "stalemate": 0,
        "other": 0,
    }

    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        while True:
            try:
                game = chess.pgn.read_game(handle)
            except Exception:
                # Corruption while parsing this game — advance to the next one.
                continue
            if game is None:
                break
            # Skip games the parser flagged as malformed (illegal SAN, etc.).
            if game.errors:
                continue
            try:
                start = game.board()
                if not is_target_material(start, strong_extra):
                    continue
                final = game.end().board()
            except Exception:
                # Replay failed (illegal move mid-game) — treat as malformed.
                continue

            counts["kqvk_games"] += 1
            counts[classify_terminal(final)] += 1

    n = counts["kqvk_games"]
    counts["mate_rate"] = (counts["mates"] / n) if n else 0.0
    # `valid` tells downstream consumers whether the sample is large enough to
    # trust mate_rate. Fewer than 30 KQvK games is too small to be meaningful.
    counts["valid"] = n >= 30
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="KQvK conversion audit for a PGN file.")
    parser.add_argument("pgn", help="path to the PGN file to audit")
    parser.add_argument(
        "--material",
        default="KQ",
        help="strong-side material spec (default: KQ)",
    )
    args = parser.parse_args(argv)

    strong_extra = parse_material(args.material)
    result = audit_pgn(args.pgn, strong_extra)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
