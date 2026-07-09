"""Tests for scripts/sf_relabel_cache.py.

Covers the pure cp/mate -> value mapping (side-to-move POV, tanh scale, mate
band) and the in-place relabel that swaps a PGNTrajectory's outcome value
targets for Stockfish evals while preserving the rest of the schema (fens,
actions, policy targets, rewards, legal actions, rep flags, and lengths).

No Stockfish binary is required — the mapping and relabel are engine-free.

Run with: cd python && pytest tests/test_sf_relabel_cache.py -v
"""

from __future__ import annotations

import importlib.util
import math
import os

import chess.engine

from hyzero.data.pgn_ingest import PGNTrajectory

_SCRIPTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "scripts"
)
_SPEC = importlib.util.spec_from_file_location(
    "sf_relabel_cache", os.path.join(_SCRIPTS_DIR, "sf_relabel_cache.py")
)
srl = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(srl)


def test_cp_to_value_zero_is_neutral():
    assert srl.cp_to_value(0) == 0.0


def test_cp_to_value_matches_tanh_scale():
    assert srl.cp_to_value(200, scale=400.0) == math.tanh(0.5)


def test_cp_to_value_is_sign_antisymmetric():
    assert srl.cp_to_value(150) == -srl.cp_to_value(-150)


def test_cp_to_value_stays_in_unit_interval():
    assert -1.0 <= srl.cp_to_value(100000) <= 1.0


def test_mate_for_side_to_move_is_top_of_band():
    assert srl.mate_to_value(1) == srl.MATE_VALUE_HI


def test_mate_against_side_to_move_is_negative_band():
    assert srl.mate_to_value(-1) == -srl.MATE_VALUE_HI


def test_deeper_mate_decays_but_stays_in_band():
    v = srl.mate_to_value(4, eps=0.01)
    assert srl.MATE_VALUE_LO <= v < srl.MATE_VALUE_HI


def test_very_deep_mate_clamped_to_band_floor():
    assert srl.mate_to_value(50, eps=0.01) == srl.MATE_VALUE_LO


def test_score_to_value_uses_side_to_move_relative_cp():
    # A White-POV +300cp score is +300 for White to move (relative == pov).
    score = chess.engine.PovScore(chess.engine.Cp(300), chess.WHITE)
    assert srl.score_to_value(score, scale=400.0) == srl.cp_to_value(300)


def test_score_to_value_maps_mate():
    score = chess.engine.PovScore(chess.engine.Mate(2), chess.WHITE)
    assert srl.score_to_value(score) == srl.mate_to_value(2)


def _sample_trajectory():
    return PGNTrajectory(
        fens=["fenA", "fenB", None],
        actions=[10, 20],
        policy_actions=[10, 20, -1],
        target_values=[1.0, -1.0, 0.0],
        target_rewards=[0.0, 0.0, 0.0],
        legal_actions=[[10, 11], [20, 21], []],
        rep_flags=[False, True, False],
    )


def test_relabel_replaces_values_with_stockfish_evals():
    traj = _sample_trajectory()
    values = {"fenA": 0.25, "fenB": -0.4}
    srl.relabel_trajectories([traj], values, outcome_blend=0.0)
    assert traj.target_values == [0.25, -0.4, 0.0]


def test_relabel_blend_mixes_outcome_and_stockfish():
    traj = _sample_trajectory()
    values = {"fenA": 0.2, "fenB": -0.5}
    srl.relabel_trajectories([traj], values, outcome_blend=0.5)
    # step 0: 0.5*0.2 + 0.5*1.0 = 0.6 ; step 1: 0.5*-0.5 + 0.5*-1.0 = -0.75
    assert traj.target_values == [0.6, -0.75, 0.0]


def test_relabel_preserves_non_value_schema():
    traj = _sample_trajectory()
    ref = _sample_trajectory()
    srl.relabel_trajectories([traj], {"fenA": 0.1, "fenB": 0.2}, outcome_blend=0.0)
    assert (
        traj.fens == ref.fens
        and traj.actions == ref.actions
        and traj.policy_actions == ref.policy_actions
        and traj.target_rewards == ref.target_rewards
        and traj.legal_actions == ref.legal_actions
        and traj.rep_flags == ref.rep_flags
    )


def test_relabel_preserves_lengths():
    traj = _sample_trajectory()
    srl.relabel_trajectories([traj], {"fenA": 0.1, "fenB": 0.2}, outcome_blend=0.0)
    assert len(traj.target_values) == len(traj.fens)
