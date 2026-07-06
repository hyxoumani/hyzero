#!/usr/bin/env python3
"""Live watcher for a hyzero training run's self-play output.

Reads the append-only ``logs/selfplay_sample.pgn`` and the newest
``logs/baseline_*.log`` to surface — in one place — how the run is *playing*
(self-play games) and *learning* (training loss). It is strictly READ-ONLY: it
never writes checkpoints, logs, or caches, and it needs no GPU.

Modes
-----
``--snapshot`` (default)
    One-shot report on the CURRENT run: games so far, termination mix
    (counts + pct), endgame-class (KQvK/KRvK) conversion, avg game length, and a
    mate-rate trend comparing the run's first window of games against its most
    recent window (is mate conversion improving *within* the run?). Also tails
    the newest ``logs/baseline_*.log`` for the last few training loss lines so
    the training state is visible in the same report.

``--game [N|last]``
    Pretty-print one game in SAN with move numbers and its termination, so a
    human can eyeball play quality.

``--follow [--interval S]``
    Loop the snapshot every ``S`` seconds (default 300) until Ctrl-C, parsing
    the growing PGN incrementally.

Live-file handling
------------------
The PGN is parsed by splitting on ``[Event`` boundaries and always holding back
the final block as a *carry* until the next game begins — so a game that is
still being written (a truncated final game) is never half-counted or allowed to
crash the report. ``PgnTail`` remembers a byte offset between polls, so follow
mode only reads the newly-appended bytes and tolerates the file growing (and
resets cleanly if the file is rotated/truncated).

Usage:
    python3 scripts/diagnostics/watch.py                       # snapshot
    python3 scripts/diagnostics/watch.py --game last
    python3 scripts/diagnostics/watch.py --follow --interval 120
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import chess

# ─── Reuse the PGN parsing / repair helpers from pgn_quality ──────────────────

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("pgn_quality", _HERE / "pgn_quality.py")
pgn_quality = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pgn_quality)

DEFAULT_PGN = "logs/selfplay_sample.pgn"
BASELINE_GLOB = "logs/baseline_*.log"
TRAIN_PREFIX = "[py_training] step"
MATE_TERMINATION = "checkmate"
DEFAULT_INTERVAL = 300
DEFAULT_WINDOW = 40

_EVENT_SPLIT_RE = re.compile(r"(?=^\[Event )", re.MULTILINE)
_RESULT_TAIL_RE = re.compile(r"(1-0|0-1|1/2-1/2|\*)\s*$")


# ─── Incremental, truncation-tolerant PGN reader ──────────────────────────────


def _split_on_event(text: str) -> tuple[list[str], str]:
    """Split ``text`` into complete game blocks and a trailing carry.

    Blocks are delimited by ``[Event`` at line start. The LAST block is always
    returned as the carry (never treated as complete) so a still-being-written
    final game is held back until the next game begins.
    """
    parts = [p for p in _EVENT_SPLIT_RE.split(text) if p.strip()]
    if not parts:
        return [], ""
    return parts[:-1], parts[-1]


def _parse_block(block: str):
    """Parse one game block into ``(headers, tokens)`` or ``None`` if empty."""
    for headers, tokens in pgn_quality._iter_games(block):
        if not tokens and "FEN" not in headers and "Result" not in headers:
            continue
        return headers, tokens
    return None


class PgnTail:
    """Byte-offset tail over an append-only PGN.

    ``poll()`` returns games completed since the last poll. The final,
    possibly-incomplete game is kept in ``carry``; ``pending()`` exposes it only
    once it looks complete (movetext ends with a result token).
    """

    def __init__(self, path: str):
        self.path = path
        self.offset = 0
        self.carry = ""

    def poll(self) -> list[tuple[dict, list[str]]]:
        try:
            size = os.path.getsize(self.path)
        except OSError:
            return []
        if size < self.offset:  # file rotated / truncated — restart
            self.offset = 0
            self.carry = ""
        with open(self.path, "r", encoding="utf-8", errors="replace") as handle:
            handle.seek(self.offset)
            chunk = handle.read()
            self.offset = handle.tell()
        complete, self.carry = _split_on_event(self.carry + chunk)
        games = []
        for block in complete:
            parsed = _parse_block(block)
            if parsed is not None:
                games.append(parsed)
        return games

    def pending(self) -> tuple[dict, list[str]] | None:
        """Return the carry as a game iff it already ended with a result."""
        if not _RESULT_TAIL_RE.search(self.carry.strip()):
            return None
        return _parse_block(self.carry)


def load_all_games(path: str) -> list[tuple[dict, list[str]]]:
    """One-shot read of every game so far, dropping a truncated final game."""
    tail = PgnTail(path)
    games = tail.poll()
    pend = tail.pending()
    if pend is not None:
        games.append(pend)
    return games


# ─── Per-game analysis ────────────────────────────────────────────────────────


def _safe_move(board: chess.Board, tok: str) -> chess.Move | None:
    """Parse ``tok`` against ``board`` with the legacy 5→4 char promotion repair."""
    if not pgn_quality._COORD_RE.match(tok):
        return None
    try:
        mv = chess.Move.from_uci(tok)
    except ValueError:
        return None
    if mv in board.legal_moves:
        return mv
    if len(tok) == 5:
        try:
            mv4 = chess.Move.from_uci(tok[:4])
        except ValueError:
            return None
        if mv4 in board.legal_moves:
            return mv4
    return None


def _game_record(headers: dict, tokens: list[str]) -> dict:
    """Summarize one game: termination, length, endgame class + conversion."""
    term = headers.get("Termination", "unknown")
    try:
        start, final, _ = pgn_quality._replay(headers, tokens)
    except Exception:
        start = final = None
    cls = pgn_quality._match_class(start) if start is not None else None
    return {
        "termination": term,
        "plies": len(tokens),
        "is_mate": term == MATE_TERMINATION,
        "class": cls,
        "class_mate": bool(final is not None and final.is_checkmate()) if cls else False,
    }


def analyze(games: list[tuple[dict, list[str]]]) -> list[dict]:
    return [_game_record(h, t) for h, t in games]


def _mate_rate(records: list[dict]) -> float:
    return (sum(r["is_mate"] for r in records) / len(records)) if records else 0.0


# ─── SAN rendering for --game ─────────────────────────────────────────────────


def game_to_san(headers: dict, tokens: list[str]) -> str:
    """Render movetext as ``1. e4 e5 2. Nf3 ...`` (with legacy repair)."""
    fen = headers.get("FEN")
    board = chess.Board(fen) if fen else chess.Board()
    parts: list[str] = []
    started = False
    for tok in tokens:
        mv = _safe_move(board, tok)
        if mv is None:
            break
        num = board.fullmove_number
        white = board.turn == chess.WHITE
        san = board.san(mv)
        if white:
            parts.append(f"{num}. {san}")
        elif not started:
            parts.append(f"{num}... {san}")
        else:
            parts[-1] += f" {san}"
        started = True
        board.push(mv)
    return " ".join(parts)


# ─── Training-log tail ────────────────────────────────────────────────────────


def newest_baseline(pattern: str = BASELINE_GLOB) -> str | None:
    matches = glob.glob(pattern)
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def _tail_bytes(path: str, nbytes: int = 65536) -> str:
    size = os.path.getsize(path)
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        if size > nbytes:
            handle.seek(size - nbytes)
        return handle.read()


def training_tail(path: str | None, n: int = 5) -> list[str]:
    """Last ``n`` training loss lines from ``path`` (falls back to any 'step' line)."""
    if path is None:
        return []
    text = _tail_bytes(path)
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    loss = [ln for ln in lines if ln.startswith(TRAIN_PREFIX)]
    if not loss:
        loss = [ln for ln in lines if "step" in ln]
    return loss[-n:]


# ─── Snapshot report ──────────────────────────────────────────────────────────


def snapshot_report(
    pgn_path: str = DEFAULT_PGN,
    baseline_pattern: str = BASELINE_GLOB,
    window: int = DEFAULT_WINDOW,
) -> str:
    lines: list[str] = []
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"=== hyzero run snapshot @ {stamp} ===")

    if not os.path.exists(pgn_path):
        lines.append(f"PGN not found: {pgn_path}")
        records = []
    else:
        games = load_all_games(pgn_path)
        records = analyze(games)
        lines.append(f"PGN: {pgn_path}")
        lines.append(f"Games so far: {len(records)}")

    if records:
        term_counts = Counter(r["termination"] for r in records)
        total = len(records)
        lines.append("Termination mix:")
        for name, count in term_counts.most_common():
            lines.append(f"  {name:<22} {count:>5}  ({100.0 * count / total:5.1f}%)")

        lines.append("Endgame conversion:")
        combined_games = combined_mates = 0
        for cls in ("KQvK", "KRvK"):
            crecs = [r for r in records if r["class"] == cls]
            n = len(crecs)
            mates = sum(r["class_mate"] for r in crecs)
            rate = (100.0 * mates / n) if n else 0.0
            combined_games += n
            combined_mates += mates
            lines.append(f"  {cls:<6} games={n:<4} mates={mates:<4} rate={rate:5.1f}%")
        crate = (100.0 * combined_mates / combined_games) if combined_games else 0.0
        lines.append(
            f"  combined mates={combined_mates}/{combined_games} rate={crate:5.1f}%"
        )

        avg_plies = sum(r["plies"] for r in records) / total
        lines.append(f"Avg game length: {avg_plies:.1f} plies")

        w = min(window, total)
        start_rate = 100.0 * _mate_rate(records[:w])
        recent_rate = 100.0 * _mate_rate(records[-w:])
        delta = recent_rate - start_rate
        trend = "improving" if delta > 0 else ("flat" if delta == 0 else "declining")
        lines.append(
            f"Mate-rate trend (window={w}): start={start_rate:5.1f}%  "
            f"recent={recent_rate:5.1f}%  Δ={delta:+5.1f}%  ({trend})"
        )

    baseline = newest_baseline(baseline_pattern)
    lines.append("")
    if baseline is None:
        lines.append("Training: no baseline_*.log found")
    else:
        lines.append(f"Training ({baseline}):")
        tail = training_tail(baseline)
        if not tail:
            lines.append("  (no training step lines yet)")
        for ln in tail:
            lines.append(f"  {ln}")

    return "\n".join(lines)


def game_report(pgn_path: str, which: str) -> str:
    if not os.path.exists(pgn_path):
        return f"PGN not found: {pgn_path}"
    games = load_all_games(pgn_path)
    if not games:
        return "No complete games in PGN yet."
    if which == "last":
        idx = len(games) - 1
    else:
        try:
            idx = int(which) - 1
        except ValueError:
            return f"Invalid game selector: {which!r} (use N or 'last')"
        if not 0 <= idx < len(games):
            return f"Game {which} out of range (1..{len(games)})"
    headers, tokens = games[idx]
    lines = [
        f"Game #{idx + 1} / {len(games)}",
        f"Event: {headers.get('Event', '?')}   "
        f"Result: {headers.get('Result', '*')}   "
        f"Termination: {headers.get('Termination', 'unknown')}",
    ]
    if "FEN" in headers:
        lines.append(f"FEN: {headers['FEN']}")
    lines.append("")
    lines.append(game_to_san(headers, tokens) or "(no legal moves parsed)")
    return "\n".join(lines)


def follow(pgn_path: str, baseline_pattern: str, interval: int, window: int) -> int:
    print(f"[watch] following {pgn_path} every {interval}s (Ctrl-C to stop)")
    try:
        while True:
            print(snapshot_report(pgn_path, baseline_pattern, window))
            print()
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[watch] stopped")
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Live watcher for a hyzero training run (read-only)."
    )
    parser.add_argument("--pgn", default=DEFAULT_PGN, help="self-play PGN path")
    parser.add_argument(
        "--baseline-glob", default=BASELINE_GLOB, help="training log glob"
    )
    parser.add_argument(
        "--window", type=int, default=DEFAULT_WINDOW, help="trend window size (games)"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--snapshot", action="store_true", help="one-shot report (default)")
    mode.add_argument("--game", metavar="N|last", help="pretty-print one game in SAN")
    mode.add_argument("--follow", action="store_true", help="loop the snapshot")
    parser.add_argument(
        "--interval", type=int, default=DEFAULT_INTERVAL, help="--follow seconds"
    )
    args = parser.parse_args(argv)

    if args.game is not None:
        print(game_report(args.pgn, args.game))
        return 0
    if args.follow:
        return follow(args.pgn, args.baseline_glob, args.interval, args.window)
    print(snapshot_report(args.pgn, args.baseline_glob, args.window))
    return 0


if __name__ == "__main__":
    sys.exit(main())
