#!/usr/bin/env python3
"""Merge opening + real middlegame (Stockfish-generated) + endgame FENs into
data/starting_positions.txt.

Replaces build_starting_positions.py's middlegame bucket (random-play, bad
coverage) with Stockfish-generated middlegame FENs from
data/middlegame_stockfish.txt.

Env vars:
  HYZERO_STARTS_OUTPUT     — output path (default: data/starting_positions.txt)
  HYZERO_STARTS_N          — total FEN count (default: 100000)
  HYZERO_STARTS_FRAC_START — opening fraction (default: 0.30)
  HYZERO_STARTS_FRAC_MID   — middlegame fraction (default: 0.40)
  HYZERO_STARTS_FRAC_END   — endgame fraction (default: 0.30)
  HYZERO_STARTS_MG_FILE    — middlegame source (default: data/middlegame_stockfish.txt)
  HYZERO_STARTS_TB_CACHE   — TB cache for endgame bucket (default: data/syzygy/cache_trajectories.pkl)
  HYZERO_STARTS_SEED       — PRNG seed (default: 42)
"""

from __future__ import annotations
import os, sys, pickle, random, types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import chess
from hyzero.data.tablebase import TBTrajectory, TBSample

OUTPUT = os.environ.get("HYZERO_STARTS_OUTPUT", "data/starting_positions.txt")
N_TOTAL = int(os.environ.get("HYZERO_STARTS_N", "100000"))
FRAC_START = float(os.environ.get("HYZERO_STARTS_FRAC_START", "0.30"))
FRAC_MID   = float(os.environ.get("HYZERO_STARTS_FRAC_MID",   "0.40"))
FRAC_END   = float(os.environ.get("HYZERO_STARTS_FRAC_END",   "0.30"))
MG_FILE    = os.environ.get("HYZERO_STARTS_MG_FILE", "data/middlegame_stockfish.txt")
TB_CACHE   = os.environ.get("HYZERO_STARTS_TB_CACHE", "data/syzygy/cache_trajectories.pkl")
SEED       = int(os.environ.get("HYZERO_STARTS_SEED", "42"))


def load_mg() -> list[str]:
    if not os.path.exists(MG_FILE):
        print(f"[merge] WARN: {MG_FILE} missing — middlegame bucket will be empty", file=sys.stderr)
        return []
    with open(MG_FILE) as f:
        fens = [l.strip() for l in f if l.strip()]
    print(f"[merge] loaded {len(fens)} middlegame FENs from {MG_FILE}")
    return fens


def load_tb_root_fens() -> list[str]:
    if not os.path.exists(TB_CACHE):
        print(f"[merge] WARN: {TB_CACHE} missing — endgame bucket will be empty", file=sys.stderr)
        return []
    shim = types.ModuleType("__main__")
    shim.TBTrajectory = TBTrajectory
    shim.TBSample = TBSample
    _prev = sys.modules.get("__main__")
    sys.modules["__main__"] = shim
    try:
        with open(TB_CACHE, "rb") as f:
            trajs = pickle.load(f)
    finally:
        if _prev is not None:
            sys.modules["__main__"] = _prev
    fens = []
    for t in trajs:
        if hasattr(t, "fens"):
            root = t.fens[0]
            if root is not None:
                fens.append(root)
        elif hasattr(t, "fen"):
            fens.append(t.fen)
    print(f"[merge] loaded {len(fens)} endgame root FENs from {TB_CACHE}")
    return fens


def main() -> None:
    random.seed(SEED)
    assert abs(FRAC_START + FRAC_MID + FRAC_END - 1.0) < 1e-6

    n_start = int(N_TOTAL * FRAC_START)
    n_mid   = int(N_TOTAL * FRAC_MID)
    n_end   = N_TOTAL - n_start - n_mid

    print(f"[merge] target: {N_TOTAL} (start={n_start}, mid={n_mid}, end={n_end})")

    # Opening bucket: standard initial position repeated.
    INIT = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    start_fens = [INIT] * n_start

    # Middlegame bucket: Stockfish output (sampled with replacement if too few).
    mg_pool = load_mg()
    if len(mg_pool) == 0:
        mid_fens = []
    elif len(mg_pool) >= n_mid:
        mid_fens = random.sample(mg_pool, n_mid)
    else:
        print(f"[merge] WARN: only {len(mg_pool)} middlegame FENs available; "
              f"sampling with replacement to reach {n_mid}")
        mid_fens = random.choices(mg_pool, k=n_mid)

    # Endgame bucket: TB cache root FENs.
    tb_pool = load_tb_root_fens()
    if len(tb_pool) == 0:
        end_fens = []
    elif len(tb_pool) >= n_end:
        end_fens = random.sample(tb_pool, n_end)
    else:
        end_fens = random.choices(tb_pool, k=n_end)

    all_fens = start_fens + mid_fens + end_fens
    random.shuffle(all_fens)

    # Sanity: piece-count distribution
    buckets = {}
    for f in all_fens[:5000]:  # sample
        try:
            b = chess.Board(f)
            n = len(b.piece_map())
            key = (n // 4) * 4
            buckets[key] = buckets.get(key, 0) + 1
        except Exception:
            pass
    print(f"[merge] piece-count distribution (5k sample):")
    for k in sorted(buckets):
        pct = 100 * buckets[k] / sum(buckets.values())
        print(f"  {k:2d}-{k+3}: {buckets[k]:,} ({pct:.1f}%)")

    os.makedirs(os.path.dirname(OUTPUT) or ".", exist_ok=True)
    with open(OUTPUT, "w") as f:
        for fen in all_fens:
            f.write(fen + "\n")
    print(f"[merge] wrote {len(all_fens)} FENs to {OUTPUT}")


if __name__ == "__main__":
    main()
