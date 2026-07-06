"""Tests for scripts/diagnostics/watch.py.

Reuses the ``pgn_quality`` fixture (a clean KQvK mate, a standard-start draw, and
a legacy-corrupted KRvK mate) and extends it with a truncated final game to
exercise the live-file tolerance. Asserts the one-shot snapshot summarizes the
run, that a game still being written is not counted, and that ``--game last``
renders SAN.

Run with: cd python && pytest tests/test_watch.py -v
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_DIAG = Path(__file__).resolve().parents[2] / "scripts" / "diagnostics"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


watch = _load("watch", _DIAG / "watch.py")
# Reuse the existing pgn_quality fixture rather than re-declaring the games.
_pgn_quality_test = _load(
    "test_pgn_quality", Path(__file__).resolve().parent / "test_pgn_quality.py"
)
_FIXTURE = _pgn_quality_test._FIXTURE

# A final game that is still being written: headers + one move, no result token.
_TRUNCATED_TAIL = """[Event "Selfplay"]
[White "w"]
[Black "b"]
[SetUp "1"]
[FEN "5k2/7Q/5K2/8/8/8/8/8 w - - 0 1"]

1. f6g6"""


def test_snapshot_summarizes_run(tmp_path):
    pgn = tmp_path / "sample.pgn"
    pgn.write_text(_FIXTURE, encoding="utf-8")

    report = watch.snapshot_report(str(pgn), baseline_pattern=str(tmp_path / "none_*"))

    assert "Games so far: 3" in report
    assert "Termination mix:" in report
    assert "checkmate" in report
    assert "repetition" in report
    # KRvK legacy token repaired -> counted as a mate conversion.
    assert "KRvK" in report and "KQvK" in report
    assert "Mate-rate trend" in report
    assert "no baseline_*.log found" in report


def test_snapshot_tolerates_truncated_final_game(tmp_path):
    pgn = tmp_path / "sample.pgn"
    pgn.write_text(_FIXTURE + "\n" + _TRUNCATED_TAIL, encoding="utf-8")

    # The still-being-written final game must not crash the report or be counted.
    report = watch.snapshot_report(str(pgn), baseline_pattern=str(tmp_path / "none_*"))
    assert "Games so far: 3" in report

    games = watch.load_all_games(str(pgn))
    assert len(games) == 3


def test_game_last_renders_san(tmp_path):
    pgn = tmp_path / "sample.pgn"
    pgn.write_text(_FIXTURE, encoding="utf-8")

    out = watch.game_report(str(pgn), "last")
    # Last game is the KRvK mate whose "a1a8q" token repairs to Ra8#.
    assert "Ra8" in out
    assert "checkmate" in out
    assert "1." in out


def test_game_out_of_range_is_reported(tmp_path):
    pgn = tmp_path / "sample.pgn"
    pgn.write_text(_FIXTURE, encoding="utf-8")
    assert "out of range" in watch.game_report(str(pgn), "99")


def test_training_tail_reads_last_step_lines(tmp_path):
    log = tmp_path / "baseline_x.log"
    log.write_text(
        "noise\n"
        "[py_training] step 1: total=3.0 policy=2.9 value=0.1\n"
        "[py_training] step 2: total=2.8 policy=2.7 value=0.1\n"
        "[other] step=99\n",
        encoding="utf-8",
    )
    tail = watch.training_tail(str(log), n=5)
    assert any("step 2" in ln for ln in tail)
    assert all(ln.startswith("[py_training] step") for ln in tail)
