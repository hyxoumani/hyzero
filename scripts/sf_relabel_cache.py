"""Relabel a PGN warm-start cache with dense Stockfish value targets.

The elite PGN cache (``data/corpus/pgn_cache_elite.pkl``) stores per-position
K-step ``PGNTrajectory`` windows whose ``target_values`` are the smeared game
outcome from each step's side-to-move POV (+1 / 0 / -1, optionally discounted).
That signal is coarse: an equal middlegame in a game White eventually won still
carries a +1 value target. This script replaces each per-step value target with
a REAL Stockfish evaluation of that position, mapped ``cp -> tanh(cp/scale)``
from the side-to-move POV (the same POV the outcome targets already use), so the
value head learns a graded position evaluation rather than the game result.

The one-hot played-move policy targets are left UNCHANGED — only ``target_values``
are rewritten. An optional ``--outcome-blend b`` mixes the original outcome value
back in: ``v = (1 - b) * sf_value + b * outcome_value`` (default 0 = pure SF).

Since overlapping K-step windows share positions, the ~1.16M FEN slots dedup to
~195k unique FENs; each unique FEN is evaluated once and cached (incrementally
persisted to a sidecar so an interrupted run resumes). Absorbing (``None``)
padding steps keep their zero targets.

Output is a pickled ``list[PGNTrajectory]`` with the SAME schema as the input,
loadable by ``hyzero.data.pgn_ingest.PGNCache``.

Run:
    python scripts/sf_relabel_cache.py \
        data/corpus/pgn_cache_elite.pkl \
        data/corpus/pgn_cache_elite_sfval.pkl --stats
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import shutil
import sys
import time
import types
from pathlib import Path

import chess
import chess.engine

# Make the hyzero package importable when run as a bare script from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "python"))

from hyzero.data.pgn_ingest import PGNTrajectory  # noqa: E402

# Default Stockfish binary: env override, then PATH, then the known local install.
_DEFAULT_STOCKFISH = (
    os.environ.get("HYZERO_STOCKFISH")
    or shutil.which("stockfish")
    or "/home/devs/.local/bin/stockfish"
)

# Mate value band. Mate scores map to a magnitude in [MATE_VALUE_LO, MATE_VALUE_HI]
# so they stay inside the categorical value support ([-1, 1]) yet remain clearly
# separated from ordinary large-cp evaluations.
MATE_VALUE_HI = 1.0
MATE_VALUE_LO = 0.95


def cp_to_value(cp: int, scale: float = 400.0) -> float:
    """Map a side-to-move centipawn score to a value in [-1, 1] via ``tanh``.

    Args:
        cp:    Centipawn score from the side-to-move POV.
        scale: ``tanh`` divisor (default 400 — the conventional Elo/pawn scale).

    Returns:
        ``clamp(tanh(cp / scale), -1, 1)`` as a Python float.
    """
    v = math.tanh(cp / scale)
    if v > 1.0:
        return 1.0
    if v < -1.0:
        return -1.0
    return v


def mate_to_value(mate: int, eps: float = 0.01) -> float:
    """Map a side-to-move mate distance to a value in the mate band.

    A forced mate delivered by the side to move maps to ``+`` band, a forced mate
    against it to ``-`` band. Deeper mates decay slightly toward the band floor by
    ``eps`` per move so that mate-in-1 outranks mate-in-8.

    Args:
        mate: Signed mate distance (moves), positive if the side to move mates.
        eps:  Per-move decay of the value magnitude within the band.

    Returns:
        A value ``sign * mag`` with ``mag`` clamped to
        ``[MATE_VALUE_LO, MATE_VALUE_HI]``.
    """
    sign = 1.0 if mate > 0 else -1.0
    mag = MATE_VALUE_HI - eps * (abs(mate) - 1)
    mag = min(MATE_VALUE_HI, max(MATE_VALUE_LO, mag))
    return sign * mag


def score_to_value(
    pov_score: chess.engine.PovScore,
    scale: float = 400.0,
    mate_eps: float = 0.01,
) -> float:
    """Map a Stockfish ``PovScore`` to a side-to-move value in [-1, 1]."""
    rel = pov_score.relative
    if rel.is_mate():
        return mate_to_value(rel.mate(), eps=mate_eps)
    return cp_to_value(rel.score(), scale=scale)


def _load_trajectories(path: str) -> list[PGNTrajectory]:
    """Load a pickled ``list[PGNTrajectory]``, tolerating a ``__main__`` pickle."""
    shim = types.ModuleType("__main__")
    shim.PGNTrajectory = PGNTrajectory  # type: ignore[attr-defined]
    prev = sys.modules.get("__main__")
    sys.modules["__main__"] = shim  # type: ignore[assignment]
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    finally:
        if prev is not None:
            sys.modules["__main__"] = prev
        else:
            del sys.modules["__main__"]
    return data


def _atomic_pickle(obj: object, path: str) -> None:
    """Pickle ``obj`` to ``path`` atomically (write tmp, then rename)."""
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def _unique_fens(trajectories: list[PGNTrajectory]) -> list[str]:
    """Return the unique non-absorbing FENs in first-appearance order."""
    seen: set[str] = set()
    order: list[str] = []
    for traj in trajectories:
        for fen in traj.fens:
            if fen is not None and fen not in seen:
                seen.add(fen)
                order.append(fen)
    return order


def evaluate_fens(
    fens: list[str],
    engine: chess.engine.SimpleEngine,
    limit: chess.engine.Limit,
    scale: float,
    mate_eps: float,
    cache: dict[str, float],
    cache_path: str,
    save_every: int,
) -> dict[str, float]:
    """Evaluate every FEN not already cached, persisting incrementally.

    Args:
        fens:       Unique FENs to evaluate.
        engine:     An open Stockfish engine.
        limit:      Per-position search limit (movetime or depth).
        scale:      ``cp -> value`` tanh scale.
        mate_eps:   Mate-band per-move decay.
        cache:      Existing ``fen -> value`` map (resumed from a prior run).
        cache_path: Sidecar path for incremental cache persistence.
        save_every: Persist the cache after this many new evaluations.

    Returns:
        The updated ``fen -> value`` cache.
    """
    total = len(fens)
    todo = [f for f in fens if f not in cache]
    done = total - len(todo)
    print(
        f"[sf_relabel] {total} unique FENs; {done} already cached, "
        f"{len(todo)} to evaluate",
        flush=True,
    )

    start = time.time()
    new_since_save = 0
    for i, fen in enumerate(todo):
        info = engine.analyse(chess.Board(fen), limit)
        cache[fen] = score_to_value(info["score"], scale=scale, mate_eps=mate_eps)
        new_since_save += 1

        if new_since_save >= save_every:
            _atomic_pickle(cache, cache_path)
            new_since_save = 0
            elapsed = time.time() - start
            rate = (i + 1) / max(elapsed, 1e-9)
            eta = (len(todo) - (i + 1)) / max(rate, 1e-9)
            print(
                f"[sf_relabel] evaluated {i + 1}/{len(todo)} "
                f"({rate:.0f}/s, ETA {eta / 60:.1f} min)",
                flush=True,
            )

    if new_since_save > 0:
        _atomic_pickle(cache, cache_path)
    print(
        f"[sf_relabel] evaluation complete in {(time.time() - start) / 60:.1f} min",
        flush=True,
    )
    return cache


def relabel_trajectories(
    trajectories: list[PGNTrajectory],
    values: dict[str, float],
    outcome_blend: float,
) -> None:
    """Rewrite each trajectory's ``target_values`` in place with SF evals.

    Non-absorbing steps get ``(1 - b) * sf_value + b * outcome_value``; absorbing
    (``None``) steps keep their existing (zero) target. Policy targets untouched.
    """
    for traj in trajectories:
        new_values: list[float] = []
        for fen, outcome_v in zip(traj.fens, traj.target_values):
            if fen is None:
                new_values.append(outcome_v)
                continue
            sf_v = values[fen]
            new_values.append((1.0 - outcome_blend) * sf_v + outcome_blend * outcome_v)
        traj.target_values = new_values


def _spot_check(
    trajectories: list[PGNTrajectory],
    values: dict[str, float],
    scale: float,
    n: int,
) -> None:
    """Print FEN / SF value / approx-cp / old outcome for the first n positions."""
    print(f"[sf_relabel] spot-check (first {n} positions):", flush=True)
    shown = 0
    for traj in trajectories:
        for fen, new_v, old_v in zip(
            traj.fens, traj.target_values, _original_outcomes(traj)
        ):
            if fen is None:
                continue
            sf_v = values[fen]
            # Recover an approximate cp from the mapped value for eyeballing
            # (exact for non-mate values; mate band shows as saturated).
            approx_cp = scale * math.atanh(max(-0.999999, min(0.999999, sf_v)))
            print(
                f"  {fen}\n"
                f"    sf_value={sf_v:+.4f}  (~cp={approx_cp:+.0f})  "
                f"new_target={new_v:+.4f}  old_outcome={old_v:+.1f}",
                flush=True,
            )
            shown += 1
            if shown >= n:
                return


# During spot-check the trajectories have already been relabeled in place, so the
# original outcome value is no longer stored. It is reconstructed here only for
# the printed sanity comparison: outcome targets alternate +/- along a window and
# equal the sign of the played-side result. Rather than reconstruct, we snapshot
# the originals before relabeling via this module-level side table.
_ORIGINAL_OUTCOMES: dict[int, list[float]] = {}


def _snapshot_outcomes(trajectories: list[PGNTrajectory]) -> None:
    """Record original outcome target_values keyed by object id (for spot-check)."""
    _ORIGINAL_OUTCOMES.clear()
    for traj in trajectories:
        _ORIGINAL_OUTCOMES[id(traj)] = list(traj.target_values)


def _original_outcomes(traj: PGNTrajectory) -> list[float]:
    """Return the pre-relabel outcome values for a trajectory (or the current)."""
    return _ORIGINAL_OUTCOMES.get(id(traj), list(traj.target_values))


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Relabel a PGN cache with dense Stockfish value targets."
    )
    parser.add_argument(
        "in_path",
        nargs="?",
        default="data/corpus/pgn_cache_elite.pkl",
        help="Input pickled list[PGNTrajectory] (default elite cache).",
    )
    parser.add_argument(
        "out_path",
        nargs="?",
        default="data/corpus/pgn_cache_elite_sfval.pkl",
        help="Output pickled list[PGNTrajectory] with SF value targets.",
    )
    parser.add_argument(
        "--stockfish", default=_DEFAULT_STOCKFISH,
        help="Path to the Stockfish UCI binary.",
    )
    parser.add_argument(
        "--movetime-ms", type=float, default=15.0,
        help="Per-position search time in ms (used unless --depth is set).",
    )
    parser.add_argument(
        "--depth", type=int, default=None,
        help="Fixed search depth (overrides --movetime-ms when set).",
    )
    parser.add_argument(
        "--value-scale", type=float, default=400.0,
        help="cp->value tanh divisor (default 400).",
    )
    parser.add_argument(
        "--mate-eps", type=float, default=0.01,
        help="Per-move decay of the mate value magnitude within the band.",
    )
    parser.add_argument(
        "--outcome-blend", type=float, default=0.0,
        help="Blend weight for the original outcome value (0 = pure SF).",
    )
    parser.add_argument(
        "--save-every", type=int, default=20000,
        help="Persist the eval sidecar cache after this many new evaluations.",
    )
    parser.add_argument(
        "--cache-path", default=None,
        help="Eval sidecar cache path (default: <out_path>.evalcache.pkl).",
    )
    parser.add_argument(
        "--threads", type=int, default=1,
        help="Stockfish Threads option (keep low on a shared machine).",
    )
    parser.add_argument(
        "--spot-check", type=int, default=10,
        help="Print this many (FEN, value) samples after relabeling.",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Print a summary after relabeling.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    cache_path = args.cache_path or f"{args.out_path}.evalcache.pkl"

    if not (0.0 <= args.outcome_blend <= 1.0):
        raise ValueError("--outcome-blend must be in [0, 1]")

    print(f"[sf_relabel] loading {args.in_path}", flush=True)
    trajectories = _load_trajectories(args.in_path)
    fens = _unique_fens(trajectories)

    cache: dict[str, float] = {}
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        print(f"[sf_relabel] resumed {len(cache)} evals from {cache_path}", flush=True)

    if args.depth is not None:
        limit = chess.engine.Limit(depth=args.depth)
        setting = f"depth={args.depth}"
    else:
        limit = chess.engine.Limit(time=args.movetime_ms / 1000.0)
        setting = f"movetime={args.movetime_ms:.0f}ms"
    print(f"[sf_relabel] stockfish={args.stockfish} setting={setting}", flush=True)

    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    try:
        if args.threads and args.threads != 1:
            engine.configure({"Threads": args.threads})
        evaluate_fens(
            fens, engine, limit,
            scale=args.value_scale, mate_eps=args.mate_eps,
            cache=cache, cache_path=cache_path, save_every=args.save_every,
        )
    finally:
        engine.quit()

    _snapshot_outcomes(trajectories)
    relabel_trajectories(trajectories, cache, args.outcome_blend)
    _atomic_pickle(trajectories, args.out_path)
    print(f"[sf_relabel] wrote {len(trajectories)} trajectories -> {args.out_path}",
          flush=True)

    if args.spot_check > 0:
        _spot_check(trajectories, cache, args.value_scale, args.spot_check)

    if args.stats:
        print(
            f"[sf_relabel] trajectories={len(trajectories)} "
            f"unique_fens={len(fens)} evals_cached={len(cache)} "
            f"setting={setting} value_scale={args.value_scale} "
            f"outcome_blend={args.outcome_blend} out={args.out_path}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
