#!/usr/bin/env python3
"""PGN quality report for hyzero self-play / eval logs.

Emits a single JSON summary describing:

  * ``termination``   — distribution of ``[Termination "..."]`` headers across
                        all games (checkmate / stalemate / repetition / ...).
  * ``endgame``       — per-class (KQvK / KRvK) conversion: how many tracked
                        basic-mate starts reach an actual checkmate vs shuffle
                        to a draw, plus a combined roll-up and ``mate_rate``.
  * ``std_start``     — stats for games from the standard opening position:
                        count, result distribution, mean ply length.
  * ``repair``        — legacy-corruption repair counters (see below).

Legacy repair heuristic
------------------------
Before the pgn-writer fix (``selfplay: fix spurious promotion suffix ...``) a
non-pawn move landing on rank 1/8 was logged with a spurious ``q`` suffix
(``a7a8q`` for a rook), which python-chess rejects as an illegal promotion and
truncates the game. This report retries any illegal 5-char coordinate token as
its 4-char form so legacy PGNs replay to their true terminal position. The
``repair`` block reports how many tokens/games were repaired; new (fixed) PGNs
report zero.

Usage:
    python3 scripts/diagnostics/pgn_quality.py logs/selfplay_sample.pgn
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter

import chess

# ─── Material classification (KQvK / KRvK, either color) ──────────────────────

_NON_KING = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)

# name -> strong-side non-king material (Counter of piece_type -> count).
_CLASS_SPECS = {
    "KQvK": Counter({chess.QUEEN: 1}),
    "KRvK": Counter({chess.ROOK: 1}),
}

_OUTCOME_FIELDS = (
    "games",
    "mates",
    "insufficient_material",
    "repetition",
    "stalemate",
    "other",
)

_HEADER_RE = re.compile(r'^\[(\w+)\s+"(.*)"\]\s*$')
_MOVENUM_RE = re.compile(r"^\d+\.(\.\.)?$")
_RESULTS = {"1-0", "0-1", "1/2-1/2", "*"}
_COORD_RE = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$")


def _non_king_counts(board: chess.Board, color: bool) -> Counter:
    counts = Counter()
    for pt in _NON_KING:
        n = len(board.pieces(pt, color))
        if n:
            counts[pt] = n
    return counts


def _match_class(board: chess.Board) -> str | None:
    """Return the tracked class name matching ``board`` material, or None."""
    if len(board.pieces(chess.KING, chess.WHITE)) != 1:
        return None
    if len(board.pieces(chess.KING, chess.BLACK)) != 1:
        return None
    white = _non_king_counts(board, chess.WHITE)
    black = _non_king_counts(board, chess.BLACK)
    for name, spec in _CLASS_SPECS.items():
        if (white == spec and not black) or (black == spec and not white):
            return name
    return None


def _classify_terminal(board: chess.Board) -> str:
    if board.is_checkmate():
        return "mates"
    if board.is_stalemate():
        return "stalemate"
    if board.is_insufficient_material():
        return "insufficient_material"
    if board.is_repetition(3):
        return "repetition"
    return "other"


# ─── Manual game parsing (needed for the token-level repair) ──────────────────


def _iter_games(text: str):
    """Yield ``(headers: dict, tokens: list[str])`` for each game in ``text``.

    A game is a header block (``[Key "Value"]`` lines) followed by movetext.
    Games are delimited by the next ``[Event`` header after movetext was seen,
    mirroring how the audit splits the append-only log.
    """
    headers: dict[str, str] = {}
    tokens: list[str] = []
    seen_moves = False

    def flush():
        nonlocal headers, tokens, seen_moves
        if headers or tokens:
            yield_val = (headers, tokens)
            headers = {}
            tokens = []
            seen_moves = False
            return yield_val
        return None

    for raw in text.splitlines():
        line = raw.strip()
        m = _HEADER_RE.match(line)
        if m:
            if seen_moves:
                out = flush()
                if out is not None:
                    yield out
            headers[m.group(1)] = m.group(2)
            continue
        if not line:
            continue
        # Movetext line: split into tokens, drop move numbers and results.
        for tok in line.split():
            if _MOVENUM_RE.match(tok) or tok in _RESULTS:
                continue
            tokens.append(tok)
            seen_moves = True

    if headers or tokens:
        yield (headers, tokens)


def _replay(headers: dict, tokens: list[str]) -> tuple[chess.Board, chess.Board, int]:
    """Replay ``tokens`` from the game's start position with legacy repair.

    Returns ``(start_board, final_board, repaired)`` where ``repaired`` counts
    tokens that only parsed after dropping a spurious promotion suffix.
    """
    fen = headers.get("FEN")
    start = chess.Board(fen) if fen else chess.Board()
    board = start.copy()
    repaired = 0
    for tok in tokens:
        if not _COORD_RE.match(tok):
            break
        try:
            mv = chess.Move.from_uci(tok)
        except ValueError:
            break
        if mv in board.legal_moves:
            board.push(mv)
            continue
        # Legacy repair: retry a 5-char token as its 4-char (no-promotion) form.
        if len(tok) == 5:
            try:
                mv4 = chess.Move.from_uci(tok[:4])
            except ValueError:
                break
            if mv4 in board.legal_moves:
                board.push(mv4)
                repaired += 1
                continue
        break
    return start, board, repaired


def _empty_class_counts() -> dict:
    return {f: 0 for f in _OUTCOME_FIELDS}


def analyze_pgn(path: str) -> dict:
    """Analyze ``path`` and return the quality-report dict."""
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        text = handle.read()

    termination = Counter()
    classes = {name: _empty_class_counts() for name in _CLASS_SPECS}
    std_results = Counter()
    std_plies_total = 0
    std_games = 0
    total_games = 0
    repaired_tokens = 0
    repaired_games = 0

    for headers, tokens in _iter_games(text):
        if not tokens and "FEN" not in headers and "Result" not in headers:
            continue
        total_games += 1
        termination[headers.get("Termination", "unknown")] += 1

        try:
            start, final, repaired = _replay(headers, tokens)
        except Exception:
            continue
        if repaired:
            repaired_tokens += repaired
            repaired_games += 1

        cls = _match_class(start)
        if cls is not None:
            bucket = classes[cls]
            bucket["games"] += 1
            bucket[_classify_terminal(final)] += 1

        if start.fen() == chess.STARTING_FEN:
            std_games += 1
            std_plies_total += len(final.move_stack)
            std_results[headers.get("Result", "*")] += 1

    for bucket in classes.values():
        n = bucket["games"]
        bucket["mate_rate"] = (bucket["mates"] / n) if n else 0.0

    combined = _empty_class_counts()
    for bucket in classes.values():
        for f in _OUTCOME_FIELDS:
            combined[f] += bucket[f]
    cn = combined["games"]
    combined["mate_rate"] = (combined["mates"] / cn) if cn else 0.0

    return {
        "total_games": total_games,
        "termination": dict(termination),
        "endgame": {"classes": classes, "combined": combined},
        "std_start": {
            "games": std_games,
            "results": dict(std_results),
            "mean_plies": (std_plies_total / std_games) if std_games else 0.0,
        },
        "repair": {
            "repaired_tokens": repaired_tokens,
            "repaired_games": repaired_games,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="PGN quality report (termination / endgame conversion / repair)."
    )
    parser.add_argument("pgn", help="path to the PGN file to analyze")
    args = parser.parse_args(argv)
    print(json.dumps(analyze_pgn(args.pgn)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
