#!/usr/bin/env python3
"""Mine mate-in-1 puzzles from the Lichess puzzle database.

Lichess puzzles are structured as "opponent blunders, you play the winning move(s)."
Moves column: space-separated UCI moves, starting with opponent's move.

For mate-in-1 puzzles:
    FEN = position BEFORE opponent's move
    Moves = "opp_move our_mate_move"
    We supervise on (board_after_opp_move, our_mate_move).

We filter by the "mateIn1" theme and verify each position with python-chess:
    - Apply opp_move to FEN → get supervised FEN
    - Verify our_mate_move is legal AND is checkmate
    - Keep if verified

Usage:
    python3 scripts/mine_lichess_mate_in_1.py \\
        --in data/lichess/lichess_db_puzzle.csv.zst \\
        --out data/lichess_mates.pkl \\
        --max 200000
"""
from __future__ import annotations
import argparse, os, pickle, subprocess, sys, time, csv

import chess


def mine(zst_path: str, out_path: str, max_puzzles: int, theme_filter: str) -> None:
    """Stream the .zst file via `zstdcat`, parse rows, filter, verify."""
    print(f"streaming {zst_path}")
    t0 = time.time()

    # Use zstd CLI for decompression (zstandard pip package not installed)
    proc = subprocess.Popen(
        ["zstdcat", zst_path],
        stdout=subprocess.PIPE,
        text=True,
    )
    reader = csv.reader(proc.stdout)
    header = next(reader)
    col = {name: i for i, name in enumerate(header)}

    puzzles = []
    total_rows = 0
    themed = 0
    verified = 0
    for row in reader:
        total_rows += 1
        if total_rows % 500_000 == 0:
            print(f"  scanned {total_rows:,} rows, {themed:,} themed, "
                  f"{verified:,} verified ({time.time() - t0:.0f}s)")

        themes = row[col["Themes"]]
        if theme_filter not in themes:
            continue
        themed += 1

        fen = row[col["FEN"]]
        moves_str = row[col["Moves"]].strip()
        if not moves_str:
            continue
        moves = moves_str.split()
        if len(moves) < 2:
            continue
        opp_uci, our_uci = moves[0], moves[1]

        # Verify: apply opp_move; check our_move is a legal mate
        try:
            board = chess.Board(fen)
            opp_move = chess.Move.from_uci(opp_uci)
            if opp_move not in board.legal_moves:
                continue
            board.push(opp_move)

            our_move = chess.Move.from_uci(our_uci)
            if our_move not in board.legal_moves:
                continue
            board.push(our_move)
            if board.is_checkmate():
                board.pop()
                puzzles.append((board.fen(), our_uci))
                verified += 1
                if verified >= max_puzzles:
                    break
        except Exception:
            continue

    proc.stdout.close()
    proc.wait()

    print(f"done. scanned={total_rows:,} themed={themed:,} verified={verified:,} "
          f"({time.time() - t0:.0f}s)")

    # Diagnostic: piece-count distribution + side distribution
    by_pieces = {}
    w_count = b_count = 0
    for fen, _ in puzzles[:5000]:
        b = chess.Board(fen)
        n = len(b.piece_map())
        by_pieces[n] = by_pieces.get(n, 0) + 1
        if b.turn == chess.WHITE:
            w_count += 1
        else:
            b_count += 1
    print("piece-count distribution (5k sample):")
    for n in sorted(by_pieces):
        bar = "#" * int(30 * by_pieces[n] / sum(by_pieces.values()))
        print(f"  {n:2d} pieces: {by_pieces[n]:5d}  {bar}")
    print(f"side-to-move: W={w_count}, B={b_count}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(puzzles, f)
    print(f"wrote {len(puzzles):,} puzzles to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path",
                    default="data/lichess/lichess_db_puzzle.csv.zst")
    ap.add_argument("--out", default="data/lichess_mates.pkl")
    ap.add_argument("--max", type=int, default=200000)
    ap.add_argument("--theme", default="mateIn1")
    args = ap.parse_args()
    mine(args.in_path, args.out, args.max, args.theme)


if __name__ == "__main__":
    main()
