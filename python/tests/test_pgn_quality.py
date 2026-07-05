"""Tests for scripts/diagnostics/pgn_quality.py.

Builds an inline PGN fixture exercising: a clean KQvK checkmate, a standard-start
draw, and a LEGACY-corrupted KQvK game whose rook back-rank move was logged with
a spurious ``q`` suffix (``a1a8q``). Asserts the report classifies conversions,
summarizes standard-start games, and repairs the corrupted token so the game
still replays to its true (checkmate) terminal.

Run with: cd python && pytest tests/test_pgn_quality.py -v
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_MOD_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "diagnostics" / "pgn_quality.py"
)
_spec = importlib.util.spec_from_file_location("pgn_quality", _MOD_PATH)
pgn_quality = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pgn_quality)


# A clean KQvK mate, a standard-start drawn game, and a KRvK game corrupted by
# the legacy spurious-promotion bug: the rook move a1-a8# was logged as "a1a8q".
_FIXTURE = """[Event "Selfplay"]
[White "w"]
[Black "b"]
[Result "1-0"]
[Termination "checkmate"]
[SetUp "1"]
[FEN "5k2/7Q/5K2/8/8/8/8/8 w - - 0 1"]

1. h7h8 1-0

[Event "Selfplay"]
[White "w"]
[Black "b"]
[Result "1/2-1/2"]
[Termination "repetition"]

1. e2e4 e7e5 2. e1e2 e8e7 1/2-1/2

[Event "Eval"]
[White "w"]
[Black "b"]
[Result "1-0"]
[Termination "checkmate"]
[SetUp "1"]
[FEN "7k/8/6K1/8/8/8/8/R7 w - - 0 1"]

1. a1a8q 1-0
"""


def test_report_classifies_repairs_and_summarizes(tmp_path):
    pgn = tmp_path / "sample.pgn"
    pgn.write_text(_FIXTURE, encoding="utf-8")

    report = pgn_quality.analyze_pgn(str(pgn))

    assert report["total_games"] == 3
    assert report["termination"]["checkmate"] == 2
    assert report["termination"]["repetition"] == 1

    # The legacy "a1a8q" rook token is repaired to "a1a8" and the game replays to
    # checkmate — so the KRvK class is counted as a mate, not dropped.
    assert report["repair"]["repaired_tokens"] == 1
    assert report["repair"]["repaired_games"] == 1

    endgame = report["endgame"]
    assert endgame["classes"]["KQvK"]["games"] == 1
    assert endgame["classes"]["KQvK"]["mates"] == 1
    assert endgame["classes"]["KRvK"]["games"] == 1
    assert endgame["classes"]["KRvK"]["mates"] == 1
    assert endgame["combined"]["mates"] == 2
    assert abs(endgame["combined"]["mate_rate"] - 1.0) < 1e-9

    std = report["std_start"]
    assert std["games"] == 1
    assert std["results"]["1/2-1/2"] == 1
    assert std["mean_plies"] == 4.0


def test_clean_pgn_reports_zero_repairs(tmp_path):
    """A well-formed 4-char token PGN needs no repair."""
    pgn = tmp_path / "clean.pgn"
    pgn.write_text(
        '[Event "e"]\n[Result "1-0"]\n[Termination "checkmate"]\n'
        '[SetUp "1"]\n[FEN "7k/8/6K1/8/8/8/8/R7 w - - 0 1"]\n\n1. a1a8 1-0\n',
        encoding="utf-8",
    )
    report = pgn_quality.analyze_pgn(str(pgn))
    assert report["repair"]["repaired_tokens"] == 0
    assert report["endgame"]["classes"]["KRvK"]["mates"] == 1
