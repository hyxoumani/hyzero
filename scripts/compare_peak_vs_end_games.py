#!/usr/bin/env python3
"""Compare eval games during the 'peak' (when promotions happened) vs the
'end' (locked-in draw cycles) of a 24h training run.

Text-only analyzer — reads the raw PGN without trying to validate moves
through python-chess (eval games start from non-standard FENs but don't
include [FEN] headers, so python-chess would silently drop them).
"""
from __future__ import annotations
import argparse, re
from collections import Counter

CYCLE_RE = re.compile(r'\[Event "Eval Cycle (\d+) Game \d+"\]')
RESULT_RE = re.compile(r'\[Result "([^"]+)"\]')
WHITE_RE = re.compile(r'\[White "([^"]+)"\]')
# A move token is e.g. "e2e4", "e7e8q", "g1f3" — UCI-style.
MOVE_TOKEN = re.compile(r'\b([a-h][1-8][a-h][1-8][qrbn]?)\b')


def parse_games(path: str) -> list[dict]:
    """Yield dicts: {cycle, result, white, plies, moves[: list of UCI strings]}"""
    out: list[dict] = []
    cur: dict | None = None
    move_lines: list[str] = []
    with open(path) as f:
        for line in f:
            if line.startswith('[Event '):
                # flush previous
                if cur is not None:
                    moves = MOVE_TOKEN.findall(' '.join(move_lines))
                    cur["plies"] = len(moves)
                    cur["moves"] = moves
                    out.append(cur)
                m = CYCLE_RE.search(line)
                cycle = int(m.group(1)) if m else -1
                cur = {"cycle": cycle, "result": "*", "white": "", "plies": 0, "moves": []}
                move_lines = []
            elif line.startswith('[Result '):
                if cur is not None:
                    m = RESULT_RE.search(line)
                    if m:
                        cur["result"] = m.group(1)
            elif line.startswith('[White '):
                if cur is not None:
                    m = WHITE_RE.search(line)
                    if m:
                        cur["white"] = m.group(1)
            elif line.startswith('['):
                pass
            else:
                if cur is not None and line.strip():
                    move_lines.append(line)
        if cur is not None:
            moves = MOVE_TOKEN.findall(' '.join(move_lines))
            cur["plies"] = len(moves)
            cur["moves"] = moves
            out.append(cur)
    return out


def detect_shuffle_uci(moves: list[str], window: int = 12) -> bool:
    """Last `window` plies all between ≤4 distinct squares as either source or
    destination, AND all unique moves ≤ 6, AND no move that looks like a capture
    (we can't tell captures from UCI, so just check repetition pattern)."""
    if len(moves) < window:
        return False
    last = moves[-window:]
    sources = {m[:2] for m in last}
    dests = {m[2:4] for m in last}
    distinct_moves = len(set(last))
    return len(sources) <= 4 and len(dests) <= 4 and distinct_moves <= 6


def is_central_first_white(moves: list[str], plies: int = 9) -> int:
    """Count central first moves on white's turns (indices 0,2,4,6,8 in UCI list).
    Central = pawn pushes to e3/e4/d3/d4 or knights to f3/c3 or bishops to c4/d3."""
    central_squares = {"e3", "e4", "d3", "d4", "f3", "c3", "c4"}
    count = 0
    for i in range(0, min(plies, len(moves)), 2):
        if moves[i][2:4] in central_squares:
            count += 1
    return count


def first_white_uci(moves: list[str]) -> str | None:
    return moves[0] if moves else None


def analyze_bucket(games: list[dict], label: str) -> None:
    if not games:
        print(f"\n=== {label} (0 games) ===")
        return

    n = len(games)
    results = Counter(g["result"] for g in games)
    plies = [g["plies"] for g in games]
    avg_plies = sum(plies) / n
    median_plies = sorted(plies)[n // 2]

    # Result distribution
    decisive = results.get("1-0", 0) + results.get("0-1", 0)
    draws = results.get("1/2-1/2", 0)
    decisive_pct = 100 * decisive / n

    # First-4-move variety
    first4 = [tuple(g["moves"][:4]) for g in games if g["plies"] >= 4]
    first4_unique = len(set(first4))

    # First white move histogram
    first_white = Counter(first_white_uci(g["moves"]) for g in games if g["moves"])

    # Central-first-5-white moves
    central_per_game = (
        sum(is_central_first_white(g["moves"]) for g in games) / n
    )

    # Shuffle endings
    shuffle_count = sum(detect_shuffle_uci(g["moves"]) for g in games)

    # Game length distribution
    short_games = sum(1 for p in plies if p < 10)
    very_long = sum(1 for p in plies if p > 100)

    print(f"\n=== {label} ({n} games) ===")
    print(f"  Result distribution:    1-0={results.get('1-0',0)}  0-1={results.get('0-1',0)}  draw={draws}  *={results.get('*',0)}")
    print(f"  Decisive rate:          {decisive_pct:.1f}%")
    print(f"  Avg plies:              {avg_plies:.1f}   median {median_plies}")
    print(f"  Plies <10  (instant draw):    {short_games}/{n} ({100*short_games/n:.1f}%)")
    print(f"  Plies >100 (long shuffle):    {very_long}/{n} ({100*very_long/n:.1f}%)")
    print(f"  First-4-move sequences unique: {first4_unique}/{len([g for g in games if g['plies'] >= 4])}")
    print(f"  Top-5 first white moves: {first_white.most_common(5)}")
    print(f"  Central first-5 white moves/game: {central_per_game:.2f}")
    print(f"  Shuffle endings (last-12 ply pattern): {shuffle_count}/{n} ({100*shuffle_count/n:.1f}%)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", default="logs/eval_games.pgn")
    ap.add_argument("--peak-lo", type=int, default=15)
    ap.add_argument("--peak-hi", type=int, default=48)
    ap.add_argument("--end-lo", type=int, default=240)
    ap.add_argument("--end-hi", type=int, default=268)
    ap.add_argument("--show-samples", type=int, default=2,
                    help="show this many sample games per bucket")
    args = ap.parse_args()

    print(f"loading {args.pgn}...")
    games = parse_games(args.pgn)
    print(f"  {len(games)} games loaded")

    peak = [g for g in games if args.peak_lo <= g["cycle"] <= args.peak_hi]
    end = [g for g in games if args.end_lo <= g["cycle"] <= args.end_hi]

    analyze_bucket(peak, f"PEAK (cycles {args.peak_lo}-{args.peak_hi})")
    analyze_bucket(end, f"END  (cycles {args.end_lo}-{args.end_hi})")

    if args.show_samples > 0:
        print(f"\n--- Sample games (first {args.show_samples} of each bucket) ---")
        print("\nPeak:")
        for g in peak[:args.show_samples]:
            mvs = " ".join(g["moves"][:30])
            print(f"  cycle={g['cycle']} result={g['result']} plies={g['plies']}: {mvs}{'...' if g['plies']>30 else ''}")
        print("\nEnd:")
        for g in end[:args.show_samples]:
            mvs = " ".join(g["moves"][:30])
            print(f"  cycle={g['cycle']} result={g['result']} plies={g['plies']}: {mvs}{'...' if g['plies']>30 else ''}")


if __name__ == "__main__":
    main()
