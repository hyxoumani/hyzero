#!/usr/bin/env python3
"""Sample DEEP KQvK / KRvK conversion starts for the campaign demo set.

Generates random legal two-piece won endgames (attacker = king + queen or
king + rook, defender = lone king) chosen so the *defending* king sits near the
centre while the attacker's king and piece start FAR away. Far-apart starts
force long conversions, which is exactly the signal the warm-start demos are
meant to teach.

Each candidate is:

  1. Built from random squares under the far-apart constraints below.
  2. Rejected unless python-chess considers it a legal, non-terminal position
     with the winning side to move.
  3. Probed by playing Stockfish vs Stockfish from the start (``--probe-ms``
     per move). The start is DISCARDED when the game ends in fewer than
     ``--min-plies`` plies (too shallow) — i.e. SF mate distance < min-plies.

Accepted FENs are written one-per-line to the output starts file, consumable by
``scripts/generate_sf_demos.py``. A stats line reports candidates / accepted /
average accepted conversion length.

Example:

    python scripts/make_deep_conversion_starts.py data/probe_deep_starts.txt \
        --target 150 --sample 200 --min-plies 8 --probe-ms 50
"""

from __future__ import annotations

import argparse
import random
import sys
import time

import chess
import chess.engine


# Central squares the DEFENDING (lone) king is placed on: the inner 4x4 block
# (files c-f, ranks 3-6). Keeping the defender central lengthens conversions.
_CENTER_FILES = range(2, 6)   # c, d, e, f
_CENTER_RANKS = range(2, 6)   # 3rd .. 6th rank


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
) -> chess.Board | None:
    """Build one random legal far-apart KQvK / KRvK start, or None on rejection.

    White (attacker) is to move from a king + queen-or-rook vs lone black king.
    The black king is central; the white king and piece are placed at least
    ``min_king_dist`` / ``min_piece_dist`` (Chebyshev) from it.
    """
    center = [
        chess.square(f, r) for f in _CENTER_FILES for r in _CENTER_RANKS
    ]
    bk = rng.choice(center)

    king_far = [
        s for s in range(64)
        if _chebyshev(s, bk) >= min_king_dist
    ]
    if not king_far:
        return None
    wk = rng.choice(king_far)

    piece_far = [
        s for s in range(64)
        if s not in (bk, wk) and _chebyshev(s, bk) >= min_piece_dist
    ]
    if not piece_far:
        return None
    ps = rng.choice(piece_far)

    piece_type = rng.choice([chess.QUEEN, chess.ROOK])

    board = chess.Board(fen=None)
    board.clear()
    board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))
    board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(ps, chess.Piece(piece_type, chess.WHITE))
    board.turn = chess.WHITE

    # Legal, non-terminal, and the defender (side not to move) is not in check.
    if not board.is_valid():
        return None
    if board.is_game_over():
        return None
    if board.is_check():
        return None
    return board


def probe_plies_to_end(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    *,
    probe_ms: int,
    max_plies: int,
) -> int:
    """Play SF vs SF from ``board`` and return the number of plies to game over.

    Returns ``max_plies`` when the game is truncated without terminating (which
    keeps such starts, since they are clearly not shallow).
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
    return plies


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
) -> tuple[list[str], dict[str, float]]:
    """Sample and probe candidates until ``target`` accepted or ``sample`` tried.

    Returns the accepted FENs and a stats dict with keys ``candidates``,
    ``accepted``, ``rejected_illegal``, ``rejected_shallow``, ``avg_plies``.
    """
    accepted: list[str] = []
    accepted_plies: list[int] = []
    candidates = 0
    rejected_illegal = 0
    rejected_shallow = 0

    while len(accepted) < target and candidates < sample:
        board = sample_candidate(
            rng,
            min_king_dist=min_king_dist,
            min_piece_dist=min_piece_dist,
        )
        candidates += 1
        if board is None:
            rejected_illegal += 1
            continue

        plies = probe_plies_to_end(
            engine, board, probe_ms=probe_ms, max_plies=max_plies
        )
        if plies < min_plies:
            rejected_shallow += 1
            continue

        accepted.append(board.fen())
        accepted_plies.append(plies)

    stats = {
        "candidates": candidates,
        "accepted": len(accepted),
        "rejected_illegal": rejected_illegal,
        "rejected_shallow": rejected_shallow,
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
    return (
        f"[deep_starts] candidates={int(stats['candidates'])} "
        f"accepted={int(stats['accepted'])} "
        f"rejected_illegal={int(stats['rejected_illegal'])} "
        f"rejected_shallow={int(stats['rejected_shallow'])} "
        f"avg_plies={stats['avg_plies']:.1f}"
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample deep far-apart KQvK / KRvK conversion starts."
    )
    parser.add_argument("out_path", help="File to write accepted starting FENs.")
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
        )
    finally:
        engine.quit()

    write_starts(starts, args.out_path)
    print(f"{format_stats(stats)} out={args.out_path} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
