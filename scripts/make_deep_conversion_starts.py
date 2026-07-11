#!/usr/bin/env python3
"""Sample DEEP won conversion starts across endgame classes for the demo set.

Generates random legal won endgames chosen so the *defending* king sits near the
centre while the attacker's king and piece(s) start FAR away. Far-apart starts
force long conversions, which is exactly the signal the warm-start demos are
meant to teach. The endgame class is selected with ``--class`` (see ``CLASSES``);
the default reproduces the original KQvK / KRvK (lone-defender) mix.

Each candidate is:

  1. Built from random squares under the far-apart constraints below, with the
     attacker (always White) holding the winning material for the class.
  2. Rejected unless python-chess considers it a legal, non-terminal position
     with the winning side to move.
  3. Probed by playing Stockfish vs Stockfish from the start (``--probe-ms``
     per move). The start is DISCARDED when the game ends in fewer than
     ``--min-plies`` plies (too shallow), more than ``--max-mate-plies`` plies
     (too deep, when set), or — with ``--require-mate`` — does not end in a
     checkmate delivered by the winning (White) side. ``--require-mate`` is
     essential for classes where the defender holds a piece (e.g. KQvKR,
     KRvKB/N), since random such starts are frequently drawn and must be
     filtered so the demo mate rate stays ~100%.

Accepted FENs are written one-per-line to the output starts file, consumable by
``scripts/generate_sf_demos.py``. A stats line reports candidates / accepted /
average accepted conversion length.

Example:

    python scripts/make_deep_conversion_starts.py data/probe_deep_starts.txt \
        --class KQvKR --require-mate \
        --target 150 --sample 400 --min-plies 8 --probe-ms 50
"""

from __future__ import annotations

import argparse
import random
import sys
import time

import chess
import chess.engine


# Central squares the DEFENDING king is placed on: the inner 4x4 block
# (files c-f, ranks 3-6). Keeping the defender central lengthens conversions.
_CENTER_FILES = range(2, 6)   # c, d, e, f
_CENTER_RANKS = range(2, 6)   # 3rd .. 6th rank


# Endgame classes: name -> (white_extra_piece_types, black_extra_piece_types).
# The two kings are implicit; the attacker (winning side) is always White. Piece
# classes with a defender piece (KQvKR, KRvKB/N) are only reliably won from a
# subset of positions, so sample them with ``--require-mate`` to drop draws.
CLASSES: dict[str, tuple[list[int], list[int]]] = {
    "KQvK":  ([chess.QUEEN], []),
    "KRvK":  ([chess.ROOK], []),
    "KQvKR": ([chess.QUEEN], [chess.ROOK]),
    "KRvKB": ([chess.ROOK], [chess.BISHOP]),
    "KRvKN": ([chess.ROOK], [chess.KNIGHT]),
    "K2RvK": ([chess.ROOK, chess.ROOK], []),
    "KQRvK": ([chess.QUEEN, chess.ROOK], []),
}


def _chebyshev(a: int, b: int) -> int:
    """Chebyshev (king-move) distance between two 0-63 squares."""
    return max(
        abs(chess.square_file(a) - chess.square_file(b)),
        abs(chess.square_rank(a) - chess.square_rank(b)),
    )


def sample_candidate(
    rng: random.Random,
    *,
    min_king_dist: int,
    min_piece_dist: int,
    white_extra: list[int] | None = None,
    black_extra: list[int] | None = None,
) -> chess.Board | None:
    """Build one random legal far-apart won start, or None on rejection.

    White (attacker) is to move. The black king is central; the white king and
    each white piece are placed at least ``min_king_dist`` / ``min_piece_dist``
    (Chebyshev) from it. ``white_extra`` / ``black_extra`` are the non-king piece
    types held by each side (see ``CLASSES``). When both are ``None`` the legacy
    lone-defender mix is used: one random white queen-or-rook vs a bare black
    king. Defender (black) pieces are dropped on any free square; legality is
    left to ``is_valid``.
    """
    center = [
        chess.square(f, r) for f in _CENTER_FILES for r in _CENTER_RANKS
    ]
    bk = rng.choice(center)
    occupied = {bk}

    king_far = [
        s for s in range(64)
        if s not in occupied and _chebyshev(s, bk) >= min_king_dist
    ]
    if not king_far:
        return None
    wk = rng.choice(king_far)
    occupied.add(wk)

    if white_extra is None and black_extra is None:
        white_extra = [rng.choice([chess.QUEEN, chess.ROOK])]
        black_extra = []
    white_extra = white_extra or []
    black_extra = black_extra or []

    board = chess.Board(fen=None)
    board.clear()
    board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
    board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))

    for piece_type in white_extra:
        piece_far = [
            s for s in range(64)
            if s not in occupied and _chebyshev(s, bk) >= min_piece_dist
        ]
        if not piece_far:
            return None
        ps = rng.choice(piece_far)
        occupied.add(ps)
        board.set_piece_at(ps, chess.Piece(piece_type, chess.WHITE))

    for piece_type in black_extra:
        free = [s for s in range(64) if s not in occupied]
        if not free:
            return None
        ps = rng.choice(free)
        occupied.add(ps)
        board.set_piece_at(ps, chess.Piece(piece_type, chess.BLACK))

    board.turn = chess.WHITE

    # Legal, non-terminal, and the defender (side not to move) is not in check.
    if not board.is_valid():
        return None
    if board.is_game_over():
        return None
    if board.is_check():
        return None
    return board


def probe_playout(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    *,
    probe_ms: int,
    max_plies: int,
) -> tuple[int, chess.Board]:
    """Play SF vs SF from ``board`` and return (plies_to_end, final_board).

    Truncates after ``max_plies`` plies without terminating (such starts are
    clearly not shallow). The returned board lets callers inspect the terminal
    outcome (e.g. checkmate for a win requirement).
    """
    b = board.copy()
    limit = chess.engine.Limit(time=probe_ms / 1000.0)
    plies = 0
    while not b.is_game_over():
        if plies >= max_plies:
            break
        move = engine.play(b, limit).move
        if move is None:
            break
        b.push(move)
        plies += 1
    return plies, b


def probe_plies_to_end(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    *,
    probe_ms: int,
    max_plies: int,
) -> int:
    """Play SF vs SF from ``board`` and return the number of plies to game over."""
    plies, _ = probe_playout(engine, board, probe_ms=probe_ms, max_plies=max_plies)
    return plies


def _white_delivered_mate(final: chess.Board) -> bool:
    """True when ``final`` is a checkmate with Black (the defender) to move."""
    return final.is_checkmate() and final.turn == chess.BLACK


def generate_starts(
    engine: chess.engine.SimpleEngine,
    rng: random.Random,
    *,
    target: int,
    sample: int,
    min_plies: int,
    probe_ms: int,
    max_plies: int,
    min_king_dist: int,
    min_piece_dist: int,
    white_extra: list[int] | None = None,
    black_extra: list[int] | None = None,
    require_mate: bool = False,
    max_mate_plies: int | None = None,
) -> tuple[list[str], dict[str, float]]:
    """Sample and probe candidates until ``target`` accepted or ``sample`` tried.

    ``white_extra`` / ``black_extra`` select the endgame class (see
    ``sample_candidate``). A probed start is accepted only when its playout runs
    at least ``min_plies`` and, when set, at most ``max_mate_plies`` plies; with
    ``require_mate`` it must also end in a checkmate by the winning (White) side.

    Returns the accepted FENs and a stats dict with keys ``candidates``,
    ``accepted``, ``rejected_illegal``, ``rejected_shallow``, ``rejected_deep``,
    ``rejected_nonmate``, ``avg_plies``.
    """
    accepted: list[str] = []
    accepted_plies: list[int] = []
    candidates = 0
    rejected_illegal = 0
    rejected_shallow = 0
    rejected_deep = 0
    rejected_nonmate = 0

    while len(accepted) < target and candidates < sample:
        board = sample_candidate(
            rng,
            min_king_dist=min_king_dist,
            min_piece_dist=min_piece_dist,
            white_extra=white_extra,
            black_extra=black_extra,
        )
        candidates += 1
        if board is None:
            rejected_illegal += 1
            continue

        plies, final = probe_playout(
            engine, board, probe_ms=probe_ms, max_plies=max_plies
        )
        if plies < min_plies:
            rejected_shallow += 1
            continue
        if max_mate_plies is not None and plies > max_mate_plies:
            rejected_deep += 1
            continue
        if require_mate and not _white_delivered_mate(final):
            rejected_nonmate += 1
            continue

        accepted.append(board.fen())
        accepted_plies.append(plies)

    stats = {
        "candidates": candidates,
        "accepted": len(accepted),
        "rejected_illegal": rejected_illegal,
        "rejected_shallow": rejected_shallow,
        "rejected_deep": rejected_deep,
        "rejected_nonmate": rejected_nonmate,
        "avg_plies": (sum(accepted_plies) / len(accepted_plies))
        if accepted_plies else 0.0,
    }
    return accepted, stats


def write_starts(starts: list[str], out_path: str) -> None:
    """Write accepted FENs one per line to ``out_path``."""
    with open(out_path, "w", encoding="utf-8") as f:
        for fen in starts:
            f.write(fen + "\n")


def format_stats(stats: dict[str, float]) -> str:
    """One-line human-readable summary of a deep-start sampling run."""
    line = (
        f"[deep_starts] candidates={int(stats['candidates'])} "
        f"accepted={int(stats['accepted'])} "
        f"rejected_illegal={int(stats['rejected_illegal'])} "
        f"rejected_shallow={int(stats['rejected_shallow'])}"
    )
    if "rejected_deep" in stats:
        line += f" rejected_deep={int(stats['rejected_deep'])}"
    if "rejected_nonmate" in stats:
        line += f" rejected_nonmate={int(stats['rejected_nonmate'])}"
    line += f" avg_plies={stats['avg_plies']:.1f}"
    return line


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample deep far-apart KQvK / KRvK conversion starts."
    )
    parser.add_argument("out_path", help="File to write accepted starting FENs.")
    parser.add_argument(
        "--class", dest="class_name", default=None, choices=sorted(CLASSES),
        help="Endgame class to sample (default: legacy KQvK/KRvK lone-defender mix).",
    )
    parser.add_argument(
        "--require-mate", action="store_true",
        help="Discard probes that don't end in checkmate by the winning (White) side.",
    )
    parser.add_argument(
        "--max-mate-plies", type=int, default=None,
        help="Discard starts whose SF playout runs longer than this (caps depth).",
    )
    parser.add_argument(
        "--stockfish-bin", default="stockfish",
        help="Stockfish binary (on PATH or a full path; default 'stockfish').",
    )
    parser.add_argument(
        "--target", type=int, default=150,
        help="Stop once this many starts are accepted (default 150).",
    )
    parser.add_argument(
        "--sample", type=int, default=400,
        help="Cap on candidates sampled before giving up (default 400).",
    )
    parser.add_argument(
        "--min-plies", type=int, default=8,
        help="Discard starts whose SF playout mates in fewer plies (default 8).",
    )
    parser.add_argument(
        "--probe-ms", type=int, default=50,
        help="Per-move Stockfish time for the probe playout (default 50).",
    )
    parser.add_argument(
        "--max-plies", type=int, default=200,
        help="Truncate a probe playout after this many plies (default 200).",
    )
    parser.add_argument(
        "--min-king-dist", type=int, default=4,
        help="Min Chebyshev distance of the white king from the black king.",
    )
    parser.add_argument(
        "--min-piece-dist", type=int, default=3,
        help="Min Chebyshev distance of the white piece from the black king.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="PRNG seed for sampling (default 0).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    white_extra, black_extra = (None, None)
    if args.class_name is not None:
        white_extra, black_extra = CLASSES[args.class_name]

    rng = random.Random(args.seed)
    t0 = time.time()
    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish_bin)
    try:
        starts, stats = generate_starts(
            engine,
            rng,
            target=args.target,
            sample=args.sample,
            min_plies=args.min_plies,
            probe_ms=args.probe_ms,
            max_plies=args.max_plies,
            min_king_dist=args.min_king_dist,
            min_piece_dist=args.min_piece_dist,
            white_extra=white_extra,
            black_extra=black_extra,
            require_mate=args.require_mate,
            max_mate_plies=args.max_mate_plies,
        )
    finally:
        engine.quit()

    write_starts(starts, args.out_path)
    cls = args.class_name or "KQvK/KRvK"
    print(
        f"{format_stats(stats)} class={cls} "
        f"out={args.out_path} ({time.time() - t0:.0f}s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
