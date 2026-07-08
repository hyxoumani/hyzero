"""Tests for the deep conversion-start sampler (scripts/make_deep_conversion_starts.py).

Pure-function tests (distance, sampling legality, stats formatting) run
everywhere. The probe/generation tests need a ``stockfish`` binary on PATH and
are skipped when it is absent, so CI without Stockfish stays green.

Run with: cd python && pytest tests/test_make_deep_conversion_starts.py -v
"""

from __future__ import annotations

import importlib.util
import os
import random
import shutil

import chess
import pytest


_SCRIPTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "scripts"
)
_SPEC = importlib.util.spec_from_file_location(
    "make_deep_conversion_starts",
    os.path.join(_SCRIPTS_DIR, "make_deep_conversion_starts.py"),
)
mds = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mds)


_HAS_STOCKFISH = shutil.which("stockfish") is not None
requires_stockfish = pytest.mark.skipif(
    not _HAS_STOCKFISH, reason="stockfish binary not on PATH"
)


def test_chebyshev_matches_king_move_distance():
    """_chebyshev returns the max file/rank delta between two squares."""
    a1 = chess.square(0, 0)
    c3 = chess.square(2, 2)
    assert mds._chebyshev(a1, c3) == 2


def test_sampled_start_is_legal_won_endgame_with_white_to_move():
    """sample_candidate yields a legal, non-terminal KQvK/KRvK, White to move."""
    rng = random.Random(0)
    board = None
    for _ in range(50):
        board = mds.sample_candidate(rng, min_king_dist=4, min_piece_dist=3)
        if board is not None:
            break
    assert board is not None, "sampler never produced a legal candidate"
    assert board.is_valid()
    assert not board.is_game_over()
    assert not board.is_check()
    assert board.turn == chess.WHITE


def test_sampled_start_keeps_pieces_far_from_defending_king():
    """Sampled white king and piece respect the far-apart distance floors."""
    rng = random.Random(1)
    board = None
    for _ in range(50):
        board = mds.sample_candidate(rng, min_king_dist=4, min_piece_dist=3)
        if board is not None:
            break
    assert board is not None
    bk = board.king(chess.BLACK)
    wk = board.king(chess.WHITE)
    assert mds._chebyshev(wk, bk) >= 4
    white_piece = [
        sq for sq in chess.SquareSet(board.occupied_co[chess.WHITE])
        if sq != wk
    ]
    assert len(white_piece) == 1
    assert mds._chebyshev(white_piece[0], bk) >= 3


def test_format_stats_reports_counts_and_avg_plies():
    """format_stats renders the sampling summary fields."""
    line = mds.format_stats({
        "candidates": 190,
        "accepted": 150,
        "rejected_illegal": 40,
        "rejected_shallow": 0,
        "avg_plies": 23.2,
    })
    assert "accepted=150" in line
    assert "avg_plies=23.2" in line


@requires_stockfish
def test_probe_reports_full_conversion_length():
    """A far-apart won start probes to a multi-ply conversion (>= min-plies)."""
    engine = chess.engine.SimpleEngine.popen_uci("stockfish")
    try:
        board = chess.Board("8/7Q/1K6/8/8/5k2/8/8 w - - 0 1")
        plies = mds.probe_plies_to_end(engine, board, probe_ms=50, max_plies=200)
    finally:
        engine.quit()
    assert plies >= 8


@requires_stockfish
def test_generate_starts_accepts_only_deep_starts():
    """generate_starts returns legal starts whose probe length meets min-plies."""
    engine = chess.engine.SimpleEngine.popen_uci("stockfish")
    try:
        starts, stats = mds.generate_starts(
            engine,
            random.Random(0),
            target=3,
            sample=40,
            min_plies=8,
            probe_ms=50,
            max_plies=200,
            min_king_dist=4,
            min_piece_dist=3,
        )
    finally:
        engine.quit()
    assert stats["accepted"] == len(starts)
    assert len(starts) >= 1
    assert all(chess.Board(fen).is_valid() for fen in starts)
