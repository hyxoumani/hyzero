"""Tests for the Stockfish demonstration-game generator (scripts/generate_sf_demos.py).

Pure-function tests (FEN reading, mirroring, result mapping) run everywhere. The
tests that actually search require a ``stockfish`` binary on PATH and are skipped
when it is absent, so CI without Stockfish stays green. The engine-backed smoke
test asserts that Stockfish converts won KQvK / KRvK starts to mate at a high
rate, and that the emitted PGN round-trips through the pgn_ingest reader.

Run with: cd python && pytest tests/test_generate_sf_demos.py -v
"""

from __future__ import annotations

import importlib.util
import io
import os
import random
import shutil

import chess
import pytest


_SCRIPTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "scripts"
)
_SPEC = importlib.util.spec_from_file_location(
    "generate_sf_demos", os.path.join(_SCRIPTS_DIR, "generate_sf_demos.py")
)
sfd = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sfd)

FIXTURE_STARTS = os.path.join(
    os.path.dirname(__file__), "fixtures", "sf_demo_starts.txt"
)

_HAS_STOCKFISH = shutil.which("stockfish") is not None
requires_stockfish = pytest.mark.skipif(
    not _HAS_STOCKFISH, reason="stockfish binary not on PATH"
)


def _open_engine():
    return chess.engine.SimpleEngine.popen_uci("stockfish")


def test_read_starts_skips_blanks_and_comments():
    """read_starts returns only the three FEN lines, dropping comments/blanks."""
    starts = sfd.read_starts(FIXTURE_STARTS)
    assert len(starts) == 3
    assert all(chess.Board(fen).is_valid() for fen in starts)


def test_mirror_fen_swaps_side_to_move():
    """mirror_fen hands the position to the other color to move."""
    fen = "8/8/8/4k3/8/8/3Q4/4K3 w - - 0 1"
    mirrored = sfd.mirror_fen(fen)
    assert chess.Board(mirrored).turn == chess.BLACK


def test_result_and_termination_marks_truncation_unterminated():
    """A truncated, non-over game reports Result '*' and Termination 'unterminated'."""
    board = chess.Board("8/8/8/4k3/8/8/3Q4/4K3 w - - 0 1")
    result, termination = sfd.result_and_termination(board, truncated=True)
    assert (result, termination) == ("*", "unterminated")


def test_result_and_termination_reports_checkmate_result():
    """A checkmated board maps to the decided Result with 'normal' termination."""
    board = chess.Board("6k1/6Q1/6K1/8/8/8/8/8 b - - 0 1")  # black is checkmated
    assert board.is_checkmate()
    result, termination = sfd.result_and_termination(board, truncated=False)
    assert (result, termination) == ("1-0", "normal")


@requires_stockfish
def test_stockfish_converts_won_starts_to_mate():
    """Smoke test: SF mates >90% of the won fixture starts at 50ms/move."""
    starts = sfd.read_starts(FIXTURE_STARTS)
    engine = _open_engine()
    try:
        _, stats = sfd.generate_demos(
            starts,
            engine,
            movetime_ms=50,
            depth=None,
            max_plies=200,
            mirror=False,
            games_per_start=1,
            multipv_jitter=0,
            rng=random.Random(0),
        )
    finally:
        engine.quit()

    assert stats["games"] == 3
    assert stats["mate_rate"] > 0.90, f"low SF conversion: {stats}"


@requires_stockfish
def test_generated_pgn_roundtrips_through_ingest():
    """Generated demos are ingestible: pgn_ingest accepts them and builds a batch."""
    from hyzero.data.pgn_ingest import ingest_pgn_stream, build_pgn_batch

    starts = sfd.read_starts(FIXTURE_STARTS)
    engine = _open_engine()
    try:
        games, _ = sfd.generate_demos(
            starts,
            engine,
            movetime_ms=50,
            depth=None,
            max_plies=200,
            mirror=False,
            games_per_start=1,
            multipv_jitter=0,
            rng=random.Random(0),
        )
    finally:
        engine.quit()

    pgn_text = "".join(f"{game}\n\n" for game in games)
    k_steps = 3
    trajectories, stats = ingest_pgn_stream(io.StringIO(pgn_text), k_steps=k_steps)

    assert stats["games_read"] == 3
    assert stats["games_accepted"] == 3
    assert stats["positions"] > 0

    batch = build_pgn_batch(trajectories, k_steps=k_steps)
    assert batch["observations"].shape[0] == len(trajectories)
    assert not batch["is_tablebase"].any()
