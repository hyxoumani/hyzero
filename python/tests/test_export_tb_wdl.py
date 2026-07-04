"""Tests for scripts/export_tb_wdl.py normfen normalization and idempotency.

Builds a tiny snapshot cache (list[TBSample]) inline, runs the exporter against
it, and asserts the emitted CSV plus the newer-than-cache skip behavior.

Run with: cd python && pytest tests/test_export_tb_wdl.py -v
"""

from __future__ import annotations

import importlib.util
import os
import pickle
import sys
from pathlib import Path

import pytest

# Import the exporter module directly from the scripts directory.
_EXPORT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "export_tb_wdl.py"
_spec = importlib.util.spec_from_file_location("export_tb_wdl", _EXPORT_PATH)
export_tb_wdl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(export_tb_wdl)

# Make hyzero importable for the TBSample dataclass.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hyzero.data.tablebase import TBSample  # noqa: E402


def _write_cache(path: Path, samples: list[TBSample]) -> None:
    with open(path, "wb") as f:
        pickle.dump(samples, f)


def test_normfen_ignores_clock_fields():
    """normfen drops the halfmove/fullmove clocks: clock-only differences collapse."""
    a = export_tb_wdl.normfen("4k3/8/4K3/8/8/8/4R3/8 w - - 0 1")
    b = export_tb_wdl.normfen("4k3/8/4K3/8/8/8/4R3/8 w - - 37 99")
    assert a == b == "4k3/8/4K3/8/8/8/4R3/8 w - -"


def test_export_writes_normfen_wdl_lines(tmp_path, monkeypatch):
    """Each cache position becomes one `normfen,wdl` line in STM-POV WDL ints."""
    cache = tmp_path / "cache.pkl"
    out = tmp_path / "tb_wdl.csv"
    _write_cache(
        cache,
        [
            TBSample("4k3/8/8/8/8/8/Q7/4K3 w - - 0 1", 1.0, [], [], []),
            TBSample("4k3/8/8/8/8/8/Q7/4K3 b - - 5 9", -1.0, [], [], []),
        ],
    )
    monkeypatch.setenv("HYZERO_TABLEBASE_CACHE_PATH", str(cache))
    monkeypatch.setenv("HYZERO_TB_WDL_PATH", str(out))

    assert export_tb_wdl.main() == 0
    lines = out.read_text().splitlines()
    assert set(lines) == {
        "4k3/8/8/8/8/8/Q7/4K3 w - -,1",
        "4k3/8/8/8/8/8/Q7/4K3 b - -,-1",
    }


def test_graded_export_scales_by_dtz():
    """DTZ grading: a near-mate win (dtz=1) ≈ ±0.9925; a distant win (dtz=100) = ±0.25."""
    near_win, near_fb = export_tb_wdl.graded_value(1, 1)
    assert near_win == pytest.approx(0.9925)
    assert near_fb is False

    far_win, _ = export_tb_wdl.graded_value(1, 100)
    assert far_win == pytest.approx(0.25)

    far_loss, _ = export_tb_wdl.graded_value(-1, 100)
    assert far_loss == pytest.approx(-0.25)


def test_graded_zero_wdl_stays_zero():
    """A drawn position (wdl=0) grades to exactly 0.0 regardless of dtz, no fallback."""
    value, fell_back = export_tb_wdl.graded_value(0, 42)
    assert value == 0.0
    assert fell_back is False


def test_export_is_idempotent_and_rebuilds_on_stale(tmp_path, monkeypatch):
    """A second run skips when the CSV is newer; touching the cache forces a rebuild."""
    cache = tmp_path / "cache.pkl"
    out = tmp_path / "tb_wdl.csv"
    _write_cache(cache, [TBSample("8/8/8/8/8/8/8/K1k5 w - - 0 1", 0.0, [], [], [])])
    monkeypatch.setenv("HYZERO_TABLEBASE_CACHE_PATH", str(cache))
    monkeypatch.setenv("HYZERO_TB_WDL_PATH", str(out))

    assert export_tb_wdl.main() == 0
    first_mtime = out.stat().st_mtime_ns

    # Second run: CSV already newer than cache ⇒ skipped, mtime unchanged.
    assert export_tb_wdl.main() == 0
    assert out.stat().st_mtime_ns == first_mtime

    # Make the cache newer than the CSV ⇒ rebuild happens (mtime advances).
    newer = os.path.getmtime(out) + 10
    os.utime(cache, (newer, newer))
    assert export_tb_wdl.main() == 0
    assert out.stat().st_mtime_ns != first_mtime
