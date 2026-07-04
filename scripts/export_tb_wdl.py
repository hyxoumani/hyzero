#!/usr/bin/env python3
"""Export a Syzygy value lookup CSV for Rust-side self-play value rescoring.

Reads the supervision cache (list[TBSample] snapshot OR list[TBTrajectory]) and
emits `data/syzygy/tb_wdl.csv`, one `normfen,value` line per distinct position:

    normfen = the first four FEN fields (piece placement, active color, castling,
              en passant target) space-joined. The halfmove/fullmove clocks are
              dropped so the key is clock-invariant. The en passant field is
              reconstructed from the RAW ep target square (`board.ep_square`), NOT
              python-chess's legality-filtered `board.fen()` field, so it matches
              the Rust `GameBoard::to_normfen` emitter byte-for-byte.
    value   = the position's Syzygy result from the SIDE-TO-MOVE point of view.
              In the default (ungraded) mode this is the plain WDL int {-1, 0, 1}.
              With HYZERO_TB_WDL_GRADED truthy the winning/losing magnitude is
              graded by distance-to-zeroing: v = sign(wdl) * (V_MIN + (1-V_MIN) *
              max(0, 1 - dtz/DTZ_MAX)), so near-mate wins score close to ±1 and
              distant wins decay toward ±V_MIN. Drawn positions (wdl=0) stay 0.0.
              This is exactly the POV the cache's target_value carries (STM POV)
              and the POV the Rust value targets use, so the Rust loader (an f32
              parser) stores it with no sign flip.

DTZ source: the graded scaling needs a per-record `dtz` field. Current caches do
not carry one (the trajectory builder probes DTZ to pick optimal moves but does
not store it), so every nonzero-WDL entry falls back to the full ±1 magnitude and
the fallback count is reported. A future cache that stores `dtz` grades automatically.

Idempotent: regeneration is skipped when the CSV already exists and is newer than
the input cache. Delete the CSV (or touch the cache) to force a rebuild.

Usage:
    python3 scripts/export_tb_wdl.py

Environment variables:
    HYZERO_TABLEBASE_CACHE_PATH: Input cache. Default data/syzygy/cache_tb_plus_mates.pkl.
    HYZERO_TB_WDL_PATH:          Output CSV.  Default data/syzygy/tb_wdl.csv.
    HYZERO_TB_WDL_GRADED:        Truthy ⇒ emit DTZ-graded f32 values. Default off
                                 (plain WDL ints), so behavior is unchanged unless set.

Output:
    Prints: "export_tb_wdl: wrote N entries to <path>" (or a skip/empty notice).
"""

from __future__ import annotations

import os
import sys

import chess

# DTZ grading shape (STM-POV magnitude). A near-mate win (dtz=1) scores
# ~V_MIN + (1-V_MIN)*(1-1/DTZ_MAX) = 0.9925; a distant win (dtz>=DTZ_MAX) floors
# at V_MIN. Matches the prior campaign's steepened-DTZ value gradient.
V_MIN = 0.25
DTZ_MAX = 100.0


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


def iter_positions(cache) -> "list[tuple[str, int, int | None]]":
    """Yield (fen, wdl_int, dtz) tuples from a snapshot or trajectory cache.

    Trajectory caches carry per-step `fens`/`target_values`; absorbing steps have
    `fen is None` and are skipped. WDL is clamped/rounded into {-1, 0, 1}. `dtz`
    is read from a per-record `dtz` field when present (used only in graded mode);
    caches without it yield `None`, forcing the full ±1 fallback.
    """
    out: list[tuple[str, int, int | None]] = []
    if cache.is_trajectory_format:
        for traj in cache._trajectories:  # noqa: SLF001 (internal access by design)
            dtz = getattr(traj, "dtz", None)
            for fen, val in zip(traj.fens, traj.target_values):
                if fen is None:
                    continue
                out.append((fen, _wdl_int(val), dtz))
    else:
        for sample in cache._samples:  # noqa: SLF001
            dtz = getattr(sample, "dtz", None)
            out.append((sample.fen, _wdl_int(sample.target_value), dtz))
    return out


def _wdl_int(value: float) -> int:
    """Round a target_value into a WDL int in {-1, 0, 1}."""
    if value > 0.5:
        return 1
    if value < -0.5:
        return -1
    return 0


def _graded_enabled() -> bool:
    """Whether HYZERO_TB_WDL_GRADED requests DTZ-graded output (off by default)."""
    v = os.environ.get("HYZERO_TB_WDL_GRADED", "0").strip().lower()
    return v not in ("", "0", "false", "no")


def graded_value(wdl: int, dtz: "int | None") -> "tuple[float, bool]":
    """Grade a WDL into an STM-POV value by DTZ distance; return (value, fell_back).

    v = sign(wdl) * (V_MIN + (1-V_MIN) * max(0, 1 - |dtz|/DTZ_MAX)). Drawn
    positions (wdl=0) map to 0.0. A missing `dtz` falls back to the full ±1
    magnitude and flags `fell_back=True` so the caller can count coverage.
    """
    if wdl == 0:
        return 0.0, False
    if dtz is None:
        return float(wdl), True  # no DTZ ⇒ full ±1 fallback
    scale = V_MIN + (1.0 - V_MIN) * max(0.0, 1.0 - abs(dtz) / DTZ_MAX)
    return (scale if wdl > 0 else -scale), False


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
    graded = _graded_enabled()

    # Dedup by normfen; a position's result is invariant across occurrences.
    # In graded mode `table` holds f32 values; otherwise plain WDL ints.
    table: dict[str, float] = {}
    fell_back: set[str] = set()
    for fen, wdl, dtz in iter_positions(cache):
        nf = normfen(fen)
        if graded:
            value, fb = graded_value(wdl, dtz)
            table[nf] = value
            if fb:
                fell_back.add(nf)
            else:
                fell_back.discard(nf)
        else:
            table[nf] = wdl

    if not table:
        print(f"export_tb_wdl: cache {cache_path} produced 0 positions — writing empty CSV")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp_path = f"{out_path}.tmp"
    with open(tmp_path, "w", encoding="ascii") as f:
        for nf, value in table.items():
            if graded:
                f.write(f"{nf},{value:.4f}\n")
            else:
                f.write(f"{nf},{value}\n")
    os.replace(tmp_path, out_path)

    mode = "graded" if graded else "wdl"
    print(f"export_tb_wdl: wrote {len(table)} entries to {out_path} ({mode} mode)")
    if graded and fell_back:
        print(
            f"export_tb_wdl: {len(fell_back)} of {len(table)} entries lacked dtz "
            f"— fell back to full ±1"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
