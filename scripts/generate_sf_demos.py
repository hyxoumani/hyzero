#!/usr/bin/env python3
"""Generate Stockfish demonstration games for the conversion campaign.

Reads a file of starting FENs (one per line; blank lines and ``#`` comments
ignored) — typically won-endgame probe starts — and has Stockfish play BOTH
sides of each start to termination. The result is a standard PGN with
``Result`` / ``Termination`` headers plus the ``FEN`` / ``SetUp`` headers that
let a custom start replay cleanly. The output is consumable as-is by
``python/hyzero/data/pgn_ingest.py`` (``ingest_pgn_stream``): decisive games
carry a known ``Result`` so ingest replays them into warm-start trajectories.

Example — build the real demo set from the 120 won starts:

    python scripts/generate_sf_demos.py data/probe_won_starts_120.txt \
        demos/sf_won_120.pgn --mirror --movetime-ms 100

Stockfish from a won KQvK / KRvK start mates ~100% of the time; the printed
mate% is a sanity check that the starts really are won and SF is converting.
"""

from __future__ import annotations

import argparse
import sys
import time

import chess
import chess.engine
import chess.pgn


def read_starts(path: str) -> list[str]:
    """Read starting FENs from a file, skipping blanks and ``#`` comments."""
    starts: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            starts.append(line)
    return starts


def mirror_fen(fen: str) -> str:
    """Return the color-mirrored FEN (winning side handed to the other color)."""
    return chess.Board(fen).mirror().fen()


def configure_engine(engine: chess.engine.SimpleEngine, skill: int) -> None:
    """Set Stockfish ``Skill Level`` when below the maximum (20)."""
    if skill < 20:
        engine.configure({"Skill Level": skill})


def _limit(movetime_ms: int, depth: int | None) -> chess.engine.Limit:
    """Build a per-move search limit: fixed depth if given, else movetime."""
    if depth is not None:
        return chess.engine.Limit(depth=depth)
    return chess.engine.Limit(time=movetime_ms / 1000.0)


def select_move(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    limit: chess.engine.Limit,
    rng,
    multipv_jitter: int,
) -> chess.Move | None:
    """Pick Stockfish's move, optionally jittering among near-best moves.

    With ``multipv_jitter <= 0`` this is deterministic best-move play. Otherwise
    it analyses the top few moves and randomly picks one scoring within
    ``multipv_jitter`` centipawns of the best (from the side-to-move POV), which
    diversifies repeated games while staying near-optimal so wins still convert.
    """
    if multipv_jitter <= 0:
        return engine.play(board, limit).move

    infos = engine.analyse(board, limit, multipv=4)
    if not infos:
        return engine.play(board, limit).move

    scored: list[tuple[chess.Move, int]] = []
    for info in infos:
        pv = info.get("pv")
        if not pv:
            continue
        score = info["score"].pov(board.turn).score(mate_score=100000)
        scored.append((pv[0], score))
    if not scored:
        return engine.play(board, limit).move

    best = max(s for _, s in scored)
    candidates = [mv for mv, s in scored if best - s <= multipv_jitter]
    return rng.choice(candidates)


def result_and_termination(board: chess.Board, truncated: bool) -> tuple[str, str]:
    """Map a final board to PGN ``Result`` and ``Termination`` header values."""
    if truncated and not board.is_game_over():
        return "*", "unterminated"
    return board.result(claim_draw=True), "normal"


def play_demo_game(
    engine: chess.engine.SimpleEngine,
    start_fen: str,
    *,
    movetime_ms: int,
    depth: int | None,
    max_plies: int,
    rng,
    multipv_jitter: int,
) -> tuple[chess.pgn.Game, chess.Board]:
    """Play one Stockfish-vs-Stockfish game from ``start_fen`` to termination.

    Returns the finished ``chess.pgn.Game`` (with Result / Termination / FEN /
    SetUp headers) and the terminal ``chess.Board`` for stats.
    """
    board = chess.Board(start_fen)
    limit = _limit(movetime_ms, depth)
    plies = 0
    truncated = False
    while not board.is_game_over():
        if plies >= max_plies:
            truncated = True
            break
        move = select_move(engine, board, limit, rng, multipv_jitter)
        if move is None:
            break
        board.push(move)
        plies += 1

    game = chess.pgn.Game.from_board(board)
    result, termination = result_and_termination(board, truncated)
    game.headers["Event"] = "hyzero SF demo"
    game.headers["White"] = "Stockfish"
    game.headers["Black"] = "Stockfish"
    game.headers["Result"] = result
    game.headers["Termination"] = termination
    return game, board


def generate_demos(
    starts: list[str],
    engine: chess.engine.SimpleEngine,
    *,
    movetime_ms: int,
    depth: int | None,
    max_plies: int,
    mirror: bool,
    games_per_start: int,
    multipv_jitter: int,
    rng,
) -> tuple[list[chess.pgn.Game], dict[str, float]]:
    """Play demo games for every start (both colors if ``mirror``).

    Returns the list of games and a stats dict with keys ``games``, ``mates``,
    ``mate_rate``, ``avg_plies``.
    """
    positions: list[str] = []
    for fen in starts:
        positions.append(fen)
        if mirror:
            positions.append(mirror_fen(fen))

    games: list[chess.pgn.Game] = []
    mates = 0
    total_plies = 0
    for fen in positions:
        for _ in range(games_per_start):
            game, board = play_demo_game(
                engine,
                fen,
                movetime_ms=movetime_ms,
                depth=depth,
                max_plies=max_plies,
                rng=rng,
                multipv_jitter=multipv_jitter,
            )
            games.append(game)
            if board.is_checkmate():
                mates += 1
            total_plies += board.ply() - chess.Board(fen).ply()

    n = len(games)
    stats = {
        "games": n,
        "mates": mates,
        "mate_rate": (mates / n) if n else 0.0,
        "avg_plies": (total_plies / n) if n else 0.0,
    }
    return games, stats


def write_pgn(games: list[chess.pgn.Game], out_path: str) -> None:
    """Write games to ``out_path`` as a standard multi-game PGN."""
    with open(out_path, "w", encoding="utf-8") as f:
        for game in games:
            print(game, file=f, end="\n\n")


def format_stats(stats: dict[str, float]) -> str:
    """One-line human-readable summary of a demo-generation run."""
    return (
        f"[sf_demos] games={int(stats['games'])} "
        f"mates={int(stats['mates'])} "
        f"mate_rate={stats['mate_rate'] * 100:.1f}% "
        f"avg_plies={stats['avg_plies']:.1f}"
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Stockfish demonstration games from a FEN starts file."
    )
    parser.add_argument("starts_path", help="File of starting FENs (one per line).")
    parser.add_argument("out_path", help="Path to write the output PGN.")
    parser.add_argument(
        "--stockfish-bin", default="stockfish",
        help="Stockfish binary (on PATH or a full path; default 'stockfish').",
    )
    parser.add_argument(
        "--movetime-ms", type=int, default=100,
        help="Per-move search time in milliseconds (default 100).",
    )
    parser.add_argument(
        "--depth", type=int, default=None,
        help="Fixed search depth per move (overrides --movetime-ms when set).",
    )
    parser.add_argument(
        "--skill", type=int, default=20,
        help="Stockfish Skill Level 0-20 (default 20 = max).",
    )
    parser.add_argument(
        "--max-plies", type=int, default=200,
        help="Truncate a game after this many plies (default 200).",
    )
    parser.add_argument(
        "--mirror", action="store_true",
        help="Also play the color-mirrored start (both colors from each start).",
    )
    parser.add_argument(
        "--games-per-start", type=int, default=1,
        help="Games per start position (>1 only varies with --sf-multipv-jitter).",
    )
    parser.add_argument(
        "--sf-multipv-jitter", type=int, default=0,
        help="Randomize among moves within this many centipawns of best (0 = off).",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="PRNG seed for move jitter (default 0).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import random

    args = _parse_args(argv if argv is not None else sys.argv[1:])

    if args.games_per_start > 1 and args.sf_multipv_jitter <= 0:
        print(
            "[sf_demos] warning: --games-per-start > 1 without --sf-multipv-jitter "
            "produces identical games (Stockfish is deterministic).",
            file=sys.stderr,
        )

    starts = read_starts(args.starts_path)
    if not starts:
        print(f"[sf_demos] no starts found in {args.starts_path}", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    t0 = time.time()
    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish_bin)
    try:
        configure_engine(engine, args.skill)
        games, stats = generate_demos(
            starts,
            engine,
            movetime_ms=args.movetime_ms,
            depth=args.depth,
            max_plies=args.max_plies,
            mirror=args.mirror,
            games_per_start=args.games_per_start,
            multipv_jitter=args.sf_multipv_jitter,
            rng=rng,
        )
    finally:
        engine.quit()

    write_pgn(games, args.out_path)
    print(f"{format_stats(stats)} out={args.out_path} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
