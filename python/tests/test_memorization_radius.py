"""Tests for scripts/diagnostics/memorization_radius.py.

Covers the two pure, offline-checkable pieces of the study:
  1. The perturbation generator only emits LEGAL positions (relocated piece,
     side to move preserved, no board.is_valid() violations).
  2. The board encoder produces the supervised-row shape (110 planes, no
     history) that the net was trained on.

Stockfish and checkpoint loading are NOT exercised here (network/engine-bound);
the legality test stubs the SF "still won" gate to isolate move generation.

Run with: cd python && pytest tests/test_memorization_radius.py -v
"""

from __future__ import annotations

import importlib.util
import os
import random
from pathlib import Path

import chess

_MOD_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts" / "diagnostics" / "memorization_radius.py"
)
_spec = importlib.util.spec_from_file_location("memorization_radius", _MOD_PATH)
mr = importlib.util.module_from_spec(_spec)
# The module sets HYZERO_* head env defaults at import (it is normally run as a
# script). Snapshot and restore those keys so importing it here does not leak
# categorical/moves-left head config into the rest of the test suite.
_ENV_KEYS = ("HYZERO_VALUE_HEAD", "HYZERO_MOVES_LEFT_HEAD")
_saved_env = {k: os.environ.get(k) for k in _ENV_KEYS}
_spec.loader.exec_module(mr)
for _k, _v in _saved_env.items():
    if _v is None:
        os.environ.pop(_k, None)
    else:
        os.environ[_k] = _v


# A KRRvK win with Black king in the corner-ish region, White (attacker) to move.
_KRRVK = "8/7K/3R4/8/8/8/1R6/6k1 w - - 7 4"
# A KQQvK win with Black to move (attacker = Black, tests the flip path).
_KQQVK = "8/4K3/Q7/8/5Q2/8/2k5/8 w - - 8 5"


class _AlwaysWon:
    """Stub engine gate: every candidate counts as still-won."""


def _accept_all(engine, board, movetime_ms, cp_threshold=800):
    return True


def test_perturbation_generator_emits_only_legal_positions(monkeypatch):
    # Bypass the Stockfish "still won" filter so we test pure move generation.
    monkeypatch.setattr(mr, "sf_is_won", _accept_all)
    rng = random.Random(0)
    board = chess.Board(_KRRVK)

    for ring in ("d1", "d2", "d3"):
        variants = mr.gen_ring(board, ring, _AlwaysWon(), rng,
                               verify_ms=1, target=8)
        assert variants, f"ring {ring} produced no variants"
        for pos in variants:
            assert pos.is_valid(), f"ring {ring} produced an illegal position"
            assert not pos.is_game_over()
            # Side to move is preserved (attacker keeps the move).
            assert pos.turn == board.turn
            # Material is conserved (a relocation, not an add/drop).
            assert (len(pos.piece_map()) == len(board.piece_map())), ring


def test_perturbation_shifts_expected_piece(monkeypatch):
    monkeypatch.setattr(mr, "sf_is_won", _accept_all)
    rng = random.Random(1)
    board = chess.Board(_KRRVK)
    defender_king = board.king(not board.turn)

    for pos in mr.gen_ring(board, "d1", _AlwaysWon(), rng,
                           verify_ms=1, target=8):
        moved_king = pos.king(not pos.turn)
        # d1 shifts only the defender king, by exactly one square.
        assert moved_king != defender_king
        assert chess.square_distance(moved_king, defender_king) == 1


def test_encoder_shape_matches_supervised_rows():
    # 110 planes, 8x8, no history (groups 1-7 all zero).
    for fen in (_KRRVK, _KQQVK):
        obs = mr.encode_board_python(chess.Board(fen))
        assert obs.shape == (110, 8, 8)
        assert obs.dtype.name == "float32"
        # History groups (planes 12..95) must be all zero for standalone rows.
        assert not obs[12:96].any(), "history planes should be zero (no history)"


def test_flip_action_is_involution():
    # flip_action twice is identity across base and underpromo ranges.
    for a in (0, 796, 4095, mr.NUM_BASE_ACTIONS, mr.NUM_ACTIONS - 1):
        assert mr._flip_action(mr._flip_action(a)) == a
