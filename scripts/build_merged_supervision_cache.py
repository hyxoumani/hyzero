#!/usr/bin/env python3
"""Merge TB trajectory cache + Lichess mate-in-1 puzzles into a single
supervision cache readable by the trainer's TablebaseCache.

Lichess puzzles are converted to 1-step TBTrajectory objects:
  - fens[0] = pre-mate FEN, fens[1..K] = None (absorbing)
  - actions[0] = mating action, actions[1..K-1] = -1
  - target_values[0] = +1 (mover wins), rest = 0
  - target_rewards[1] = +1 (mating transition), rest = 0
  - mate_step = 1

Output: a single pickle file consumable by TablebaseCache unchanged.

Usage:
    python3 scripts/build_merged_supervision_cache.py \\
        --tb data/syzygy/cache_trajectories.pkl \\
        --mates data/lichess_mates.pkl \\
        --out data/syzygy/cache_tb_plus_mates.pkl \\
        --k-steps 5
"""
from __future__ import annotations
import argparse, os, pickle, sys, time, types

import chess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from hyzero.data.tablebase import TBSample, TBTrajectory
from hyzero.data.board_encoder import action_from_move


def puzzle_to_trajectory(fen: str, mate_uci: str, k_steps: int) -> TBTrajectory | None:
    """Convert a (fen, mate_uci) puzzle into a 1-step TBTrajectory."""
    try:
        board = chess.Board(fen)
        move = chess.Move.from_uci(mate_uci)
        if move not in board.legal_moves:
            return None
        mate_action = action_from_move(move, board)
    except Exception:
        return None

    fens = [fen] + [None] * k_steps         # step 0 real, 1..K absorbing
    actions = [mate_action] + [-1] * (k_steps - 1)
    target_values = [1.0] + [0.0] * k_steps  # mover winning at root; absorbing below
    target_rewards = [0.0] * (k_steps + 1)
    target_rewards[1] = 1.0                   # reward fires at the mate step

    legal_acts = [action_from_move(m, board) for m in board.legal_moves]
    legal_actions = [legal_acts] + [[] for _ in range(k_steps)]
    optimal_actions = [[mate_action]] + [[] for _ in range(k_steps)]

    return TBTrajectory(
        fens=fens,
        actions=actions,
        target_values=target_values,
        target_rewards=target_rewards,
        legal_actions=legal_actions,
        optimal_actions=optimal_actions,
        mate_step=1,
    )


def load_tb_cache(path: str) -> list[TBTrajectory]:
    """Load existing TB trajectory cache (shim __main__ for unpickling)."""
    shim = types.ModuleType("__main__")
    shim.TBSample = TBSample
    shim.TBTrajectory = TBTrajectory
    _prev = sys.modules.get("__main__")
    sys.modules["__main__"] = shim
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    finally:
        if _prev is not None:
            sys.modules["__main__"] = _prev
    # Normalize into proper TBTrajectory objects (handles stale class identities)
    normalized = []
    for t in data:
        if hasattr(t, "fens"):
            normalized.append(TBTrajectory(
                fens=list(t.fens),
                actions=list(t.actions),
                target_values=[float(v) for v in t.target_values],
                target_rewards=[float(r) for r in t.target_rewards],
                legal_actions=[list(a) for a in t.legal_actions],
                optimal_actions=[list(a) for a in t.optimal_actions],
                mate_step=t.mate_step,
            ))
    return normalized


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tb", default="data/syzygy/cache_trajectories.pkl")
    ap.add_argument("--mates", default="data/lichess_mates.pkl")
    ap.add_argument("--out", default="data/syzygy/cache_tb_plus_mates.pkl")
    ap.add_argument("--k-steps", type=int, default=5)
    ap.add_argument("--max-mates", type=int, default=0,
                    help="Subsample puzzles to this count; 0 = use all")
    args = ap.parse_args()

    t0 = time.time()
    print(f"loading TB trajectories from {args.tb}...")
    tb_trajs = load_tb_cache(args.tb) if os.path.exists(args.tb) else []
    print(f"  {len(tb_trajs):,} TB trajectories loaded")

    print(f"loading mate puzzles from {args.mates}...")
    with open(args.mates, "rb") as f:
        raw_mates = pickle.load(f)
    print(f"  {len(raw_mates):,} mate puzzles loaded")

    if args.max_mates > 0 and len(raw_mates) > args.max_mates:
        import random
        rng = random.Random(42)
        raw_mates = rng.sample(raw_mates, args.max_mates)
        print(f"  subsampled to {len(raw_mates):,}")

    print(f"converting puzzles to trajectories (K={args.k_steps})...")
    mate_trajs = []
    skipped = 0
    for i, (fen, uci) in enumerate(raw_mates):
        tr = puzzle_to_trajectory(fen, uci, args.k_steps)
        if tr is None:
            skipped += 1
            continue
        mate_trajs.append(tr)
        if (i + 1) % 20_000 == 0:
            print(f"  {i+1:,}/{len(raw_mates):,} processed ({time.time()-t0:.0f}s)")
    print(f"  converted {len(mate_trajs):,} (skipped {skipped:,} invalid)")

    merged = tb_trajs + mate_trajs
    # Shuffle so sampling is balanced
    import random
    random.Random(0).shuffle(merged)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(merged, f)
    print(f"wrote {len(merged):,} trajectories to {args.out} "
          f"(TB={len(tb_trajs):,}, mates={len(mate_trajs):,}) "
          f"in {time.time()-t0:.0f}s")

    # Report fraction of mates in final cache
    mate_frac = len(mate_trajs) / max(1, len(merged))
    print(f"  mate-trajectory fraction: {100*mate_frac:.1f}%")


if __name__ == "__main__":
    main()
