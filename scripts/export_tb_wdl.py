#!/usr/bin/env python3
"""Export a Syzygy WDL lookup CSV for Rust-side self-play value rescoring.

Reads the supervision cache (list[TBSample] snapshot OR list[TBTrajectory]) and
emits `data/syzygy/tb_wdl.csv`, one `normfen,wdl` line per distinct position:

    normfen = the first four FEN fields (piece placement, active color, castling,
              en passant target) space-joined. The halfmove/fullmove clocks are
              dropped so the key is clock-invariant. The en passant field is
              reconstructed from the RAW ep target square (`board.ep_square`), NOT
              python-chess's legality-filtered `board.fen()` field, so it matches
              the Rust `GameBoard::to_normfen` emitter byte-for-byte.
    wdl     = the position's Syzygy WDL from the SIDE-TO-MOVE point of view,
              rounded to an int in {-1, 0, 1}. This is exactly the POV the cache's
              target_value carries (STM POV) and the POV the Rust value targets
              use, so the Rust loader stores it with no sign flip.

Idempotent: regeneration is skipped when the CSV already exists and is newer than
the input cache. Delete the CSV (or touch the cache) to force a rebuild.

Usage:
    python3 scripts/export_tb_wdl.py

Environment variables:
    HYZERO_TABLEBASE_CACHE_PATH: Input cache. Default data/syzygy/cache_tb_plus_mates.pkl.
    HYZERO_TB_WDL_PATH:          Output CSV.  Default data/syzygy/tb_wdl.csv.

Output:
    Prints: "export_tb_wdl: wrote N entries to <path>" (or a skip/empty notice).
"""

from __future__ import annotations

import os
import sys

import chess


def normfen(fen: str) -> str:
    """Return the clock-invariant first-four-field FEN with a RAW ep target.

    Matches the Rust `GameBoard::to_normfen` emitter: placement + active color +
    castling come from python-chess's FEN, but the en passant field is taken from
    the raw `board.ep_square` (set after any double push, regardless of whether a
    capture is legal) rather than the legality-filtered `board.fen()` ep field.
    """
    board = chess.Board(fen)
    placement, color, castling = board.fen().split(" ")[:3]
    ep = chess.square_name(board.ep_square) if board.ep_square is not None else "-"
    return f"{placement} {color} {castling} {ep}"


def iter_positions(cache) -> "list[tuple[str, int]]":
    """Yield (fen, wdl_int) pairs from a snapshot or trajectory cache.

    Trajectory caches carry per-step `fens`/`target_values`; absorbing steps have
    `fen is None` and are skipped. WDL is clamped/rounded into {-1, 0, 1}.
    """
    out: list[tuple[str, int]] = []
    if cache.is_trajectory_format:
        for traj in cache._trajectories:  # noqa: SLF001 (internal access by design)
            for fen, val in zip(traj.fens, traj.target_values):
                if fen is None:
                    continue
                out.append((fen, _wdl_int(val)))
    else:
        for sample in cache._samples:  # noqa: SLF001
            out.append((sample.fen, _wdl_int(sample.target_value)))
    return out


def _wdl_int(value: float) -> int:
    """Round a target_value into a WDL int in {-1, 0, 1}."""
    if value > 0.5:
        return 1
    if value < -0.5:
        return -1
    return 0


def main() -> int:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
    from hyzero.data.tablebase import TablebaseCache

    cache_path = os.environ.get(
        "HYZERO_TABLEBASE_CACHE_PATH", "data/syzygy/cache_tb_plus_mates.pkl"
    )
    out_path = os.environ.get("HYZERO_TB_WDL_PATH", "data/syzygy/tb_wdl.csv")

    if not os.path.exists(cache_path):
        print(f"export_tb_wdl: cache {cache_path} missing — nothing to do")
        return 0

    # Idempotency: skip when the CSV exists and is newer than the cache.
    if os.path.exists(out_path) and os.path.getmtime(out_path) >= os.path.getmtime(
        cache_path
    ):
        print(f"export_tb_wdl: {out_path} up to date (newer than cache) — skipping")
        return 0

    cache = TablebaseCache(cache_path)

    # Dedup by normfen; a position's WDL is invariant across occurrences.
    table: dict[str, int] = {}
    for fen, wdl in iter_positions(cache):
        table[normfen(fen)] = wdl

    if not table:
        print(f"export_tb_wdl: cache {cache_path} produced 0 positions — writing empty CSV")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp_path = f"{out_path}.tmp"
    with open(tmp_path, "w", encoding="ascii") as f:
        for nf, wdl in table.items():
            f.write(f"{nf},{wdl}\n")
    os.replace(tmp_path, out_path)

    print(f"export_tb_wdl: wrote {len(table)} entries to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
