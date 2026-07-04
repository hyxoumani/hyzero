#!/usr/bin/env python3
"""Basic-mate conversion audit for self-play PGNs.

Scans a PGN file, selects games whose *starting* position is one of the tracked
basic-mate classes — K+Q vs K (``KQvK``) or K+R vs K (``KRvK``), either color —
replays each to its final position, and classifies the terminal state
(checkmate / stalemate / insufficient material / repetition / other). Emits a
single JSON line summarizing conversion per class plus a combined roll-up:

    {"classes": {
        "KQvK": {"games": N, "mates": n1, "insufficient_material": n2,
                 "repetition": n3, "stalemate": n4, "other": n5,
                 "mate_rate": r},
        "KRvK": {...same fields...}},
     "combined": {...same fields, summed across classes, mate_rate recomputed...},
     "valid": bool}

``valid`` is false when fewer than 30 games total were found (``combined.games``
< 30), signaling the sample is too small for ``mate_rate`` to be meaningful.

Robust to interleaving corruption: games that fail to parse or replay (illegal
SAN, truncated move text) are skipped rather than aborting the whole scan.

Usage:
    python scripts/kqvk_audit.py logs/selfplay_sample.pgn
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

# Outcome buckets tracked per class, in output order (before mate_rate).
_OUTCOME_FIELDS = (
    "games",
    "mates",
    "insufficient_material",
    "repetition",
    "stalemate",
    "other",
)


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


# Tracked start classes: name -> strong-side non-king material spec.
_CLASS_SPECS = {
    "KQvK": parse_material("KQ"),
    "KRvK": parse_material("KR"),
}


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


def _empty_class_counts() -> dict:
    return {field: 0 for field in _OUTCOME_FIELDS}


def audit_pgn(path: str, class_specs: dict | None = None) -> dict:
    """Audit ``path`` and return the conversion-summary dict.

    Each game's starting position is matched against the tracked classes
    (``KQvK``, ``KRvK`` by default); the first match is counted. Malformed games
    (parse errors or illegal moves) are skipped silently so a single corrupt
    game cannot abort the scan.
    """
    if class_specs is None:
        class_specs = _CLASS_SPECS

    classes = {name: _empty_class_counts() for name in class_specs}

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
                matched = None
                for name, spec in class_specs.items():
                    if is_target_material(start, spec):
                        matched = name
                        break
                if matched is None:
                    continue
                final = game.end().board()
            except Exception:
                # Replay failed (illegal move mid-game) — treat as malformed.
                continue

            bucket = classes[matched]
            bucket["games"] += 1
            bucket[classify_terminal(final)] += 1

    for bucket in classes.values():
        n = bucket["games"]
        bucket["mate_rate"] = (bucket["mates"] / n) if n else 0.0

    combined = _empty_class_counts()
    for bucket in classes.values():
        for field in _OUTCOME_FIELDS:
            combined[field] += bucket[field]
    n = combined["games"]
    combined["mate_rate"] = (combined["mates"] / n) if n else 0.0

    # `valid` tells downstream consumers whether the sample is large enough to
    # trust mate_rate. Fewer than 30 games total is too small to be meaningful.
    return {"classes": classes, "combined": combined, "valid": n >= 30}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Basic-mate (KQvK / KRvK) conversion audit for a PGN file."
    )
    parser.add_argument("pgn", help="path to the PGN file to audit")
    args = parser.parse_args(argv)

    result = audit_pgn(args.pgn)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
