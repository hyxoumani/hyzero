#!/usr/bin/env python3
"""Campaign-3 driver: build a large, non-memorizable curriculum of won endgames.

Round-robins over (endgame class x depth band) cells, sampling far-apart won
starts with ``make_deep_conversion_starts`` and probing each with Stockfish vs
Stockfish (``--probe-ms`` per move) to bucket by conversion length. Accepted
FENs are de-duplicated (board + side-to-move key) against each other AND against
the fixed probe/holdout start files, then streamed to the output file with an
incremental flush every ``--flush-every`` accepts so a killed run still leaves a
usable partial set.

Depth mix (default): ~40% shallow (8-15 plies) / ~60% deep (15-45 plies).
Classes (default): KQvK, KRvK, KQvKR, K2RvK, KRvKB, KRvKN. The generator's
CLASSES are fixed material templates, so arbitrary 6-7 piece material-up
positions are not supported without editing CLASSES; only the named classes are
sampled here.

The run stops when every cell reaches its target, all cells are exhausted, or
the wall-clock ``--budget-s`` elapses -- whichever comes first -- so it always
"delivers what fits". A stats block (per class/band distribution + generation
rate) is printed and written to ``--report``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import chess
import chess.engine

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_deep_conversion_starts import (  # noqa: E402
    CLASSES,
    probe_playout,
    sample_candidate,
    _white_delivered_mate,
)


# Depth bands: name -> probe/geometry params. Shallow bands keep the kings
# closer so SF mates quickly; deep bands push them apart for long conversions.
BANDS: dict[str, dict[str, int]] = {
    "shallow": {
        "min_plies": 8, "max_mate_plies": 15, "max_plies": 17,
        "min_king_dist": 2, "min_piece_dist": 1,
    },
    "deep": {
        "min_plies": 15, "max_mate_plies": 45, "max_plies": 47,
        "min_king_dist": 4, "min_piece_dist": 3,
    },
}

DEFAULT_CLASSES = ["KQvK", "KRvK", "KQvKR", "K2RvK", "KRvKB", "KRvKN"]


def board_stm_key(fen: str) -> str:
    """De-dup key: piece placement + side to move (ignores clocks/castling/ep)."""
    parts = fen.split()
    return f"{parts[0]} {parts[1]}"


def load_keys(paths: list[str]) -> set[str]:
    """Board+STM keys for every FEN found in the given files (missing = skipped)."""
    keys: set[str] = set()
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    keys.add(board_stm_key(line))
    return keys


def split_targets(total: int, n: int) -> list[int]:
    """Split ``total`` across ``n`` cells as evenly as possible (remainder first)."""
    base, rem = divmod(total, n)
    return [base + (1 if i < rem else 0) for i in range(n)]


def build_cells(
    classes: list[str], total: int, shallow_frac: float, seed: int
) -> list[dict]:
    """One state dict per (class, band) cell with target, rng seed and params."""
    shallow_total = round(total * shallow_frac)
    deep_total = total - shallow_total
    shallow_targets = split_targets(shallow_total, len(classes))
    deep_targets = split_targets(deep_total, len(classes))

    cells: list[dict] = []
    for ci, name in enumerate(classes):
        white_extra, black_extra = CLASSES[name]
        for bi, band in enumerate(("shallow", "deep")):
            target = (shallow_targets if band == "shallow" else deep_targets)[ci]
            cells.append({
                "class": name,
                "band": band,
                "white_extra": white_extra,
                "black_extra": black_extra,
                "params": BANDS[band],
                "target": target,
                "rng": __import__("random").Random(seed + ci * 100 + bi),
                "accepted": 0,
                "candidates": 0,
                "plies_sum": 0,
                "exhausted": False,
            })
    return cells


def run(
    engine: chess.engine.SimpleEngine,
    cells: list[dict],
    out_path: str,
    *,
    seen: set[str],
    probe_ms: int,
    budget_s: float,
    flush_every: int,
    batch: int,
    attempts_cap: int,
    max_candidates_per_cell: int,
    log_every: int,
) -> int:
    """Round-robin sample/probe until targets met, cells exhausted, or budget out."""
    t0 = time.time()
    deadline = t0 + budget_s
    total = 0
    since_flush = 0
    out = open(out_path, "w", encoding="utf-8")
    try:
        while True:
            if time.time() >= deadline:
                print(f"[curriculum] budget {budget_s:.0f}s reached; stopping")
                break
            active = [c for c in cells
                      if not c["exhausted"] and c["accepted"] < c["target"]]
            if not active:
                print("[curriculum] all cells complete or exhausted")
                break

            for cell in active:
                if time.time() >= deadline:
                    break
                p = cell["params"]
                got = 0
                attempts = 0
                while got < batch and attempts < attempts_cap:
                    if cell["accepted"] >= cell["target"]:
                        break
                    if time.time() >= deadline:
                        break
                    attempts += 1
                    cell["candidates"] += 1
                    if cell["candidates"] >= max_candidates_per_cell:
                        cell["exhausted"] = True
                        break
                    board = sample_candidate(
                        cell["rng"],
                        min_king_dist=p["min_king_dist"],
                        min_piece_dist=p["min_piece_dist"],
                        white_extra=cell["white_extra"],
                        black_extra=cell["black_extra"],
                    )
                    if board is None:
                        continue
                    plies, final = probe_playout(
                        engine, board, probe_ms=probe_ms, max_plies=p["max_plies"]
                    )
                    if plies < p["min_plies"] or plies > p["max_mate_plies"]:
                        continue
                    if not _white_delivered_mate(final):
                        continue
                    fen = board.fen()
                    key = board_stm_key(fen)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.write(fen + "\n")
                    cell["accepted"] += 1
                    cell["plies_sum"] += plies
                    total += 1
                    got += 1
                    since_flush += 1
                    if since_flush >= flush_every:
                        out.flush()
                        os.fsync(out.fileno())
                        since_flush = 0
                    if total % log_every == 0:
                        rate = total / max(time.time() - t0, 1e-9)
                        print(f"[curriculum] accepted={total} "
                              f"rate={rate:.2f}/s elapsed={time.time()-t0:.0f}s",
                              flush=True)
        out.flush()
        os.fsync(out.fileno())
    finally:
        out.close()
    return total


def format_report(cells: list[dict], total: int, elapsed: float) -> str:
    """Human-readable per-cell distribution + generation-rate report."""
    lines = [
        "# Curriculum endgame 10k -- generation report",
        "",
        f"total_accepted: {total}",
        f"elapsed_s: {elapsed:.0f}",
        f"rate_per_s: {total / max(elapsed, 1e-9):.3f}",
        "",
        "| class | band | accepted | target | candidates | avg_plies |",
        "|-------|------|----------|--------|------------|-----------|",
    ]
    for c in cells:
        avg = c["plies_sum"] / c["accepted"] if c["accepted"] else 0.0
        lines.append(
            f"| {c['class']} | {c['band']} | {c['accepted']} | {c['target']} "
            f"| {c['candidates']} | {avg:.1f} |"
        )
    by_band: dict[str, int] = {}
    by_class: dict[str, int] = {}
    for c in cells:
        by_band[c["band"]] = by_band.get(c["band"], 0) + c["accepted"]
        by_class[c["class"]] = by_class.get(c["class"], 0) + c["accepted"]
    lines += ["", "band_totals: " + ", ".join(
        f"{k}={v}" for k, v in sorted(by_band.items()))]
    lines += ["class_totals: " + ", ".join(
        f"{k}={v}" for k, v in sorted(by_class.items()))]
    return "\n".join(lines) + "\n"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out_path", help="Output curriculum starts file.")
    p.add_argument("--total", type=int, default=10000)
    p.add_argument("--shallow-frac", type=float, default=0.40)
    p.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES,
                   choices=sorted(CLASSES))
    p.add_argument("--seed", type=int, default=20260711)
    p.add_argument("--probe-ms", type=int, default=30)
    p.add_argument("--budget-s", type=float, default=6 * 3600.0)
    p.add_argument("--flush-every", type=int, default=500)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--attempts-cap", type=int, default=200)
    p.add_argument("--max-candidates-per-cell", type=int, default=60000)
    p.add_argument("--log-every", type=int, default=250)
    p.add_argument("--stockfish-bin", default="stockfish")
    p.add_argument("--report", default=None,
                   help="Path to write the markdown stats report.")
    p.add_argument(
        "--exclude", nargs="*",
        default=[
            "/home/devs/workspace/hyzero/data/probe_won_starts_120.txt",
            "/home/devs/workspace/hyzero/data/probe_holdout_starts_150.txt",
        ],
        help="Files whose board+STM keys must be excluded from the output.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    seen = load_keys(args.exclude)
    print(f"[curriculum] loaded {len(seen)} exclusion keys from "
          f"{len(args.exclude)} file(s)")
    cells = build_cells(args.classes, args.total, args.shallow_frac, args.seed)

    t0 = time.time()
    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish_bin)
    try:
        total = run(
            engine, cells, args.out_path,
            seen=seen,
            probe_ms=args.probe_ms,
            budget_s=args.budget_s,
            flush_every=args.flush_every,
            batch=args.batch,
            attempts_cap=args.attempts_cap,
            max_candidates_per_cell=args.max_candidates_per_cell,
            log_every=args.log_every,
        )
    finally:
        engine.quit()

    elapsed = time.time() - t0
    report = format_report(cells, total, elapsed)
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
