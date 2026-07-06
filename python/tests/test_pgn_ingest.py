"""Tests for the PGN → training-record ingest (external-corpus warm-start).

Ingests a small hand-crafted fixture PGN (python/tests/fixtures/sample_games.pgn)
containing three short games:

  1. A knight-shuffle game that repeats the initial position (proves the
     repetition plane 102 is set while replaying).
  2. Scholar's mate, White wins (proves value-target sign flips with STM).
  3. A low-Elo game (proves the --min-elo filter skips it).

Run with: cd python && pytest tests/test_pgn_ingest.py -v
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from hyzero.data.pgn_ingest import (
    ingest_pgn_stream,
    build_pgn_batch,
    REP_PLANE_CURRENT,
)
from hyzero.data.board_encoder import NUM_ACTIONS


FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "sample_games.pgn")


def _ingest(k_steps=3, **kwargs):
    with open(FIXTURE, "r", encoding="utf-8") as f:
        return ingest_pgn_stream(f, k_steps=k_steps, **kwargs)


def test_ingest_row_schema_matches_tb_mixer_schema():
    """build_pgn_batch emits exactly the keys/shapes/dtypes of the TB trajectory mixer."""
    from hyzero.data.tablebase import TBTrajectory, build_tb_batch_trajectories

    k_steps = 3
    trajectories, _ = _ingest(k_steps=k_steps)
    pgn_batch = build_pgn_batch(trajectories[:4], k_steps=k_steps)

    tb_traj = TBTrajectory(
        fens=["8/8/8/8/8/8/8/K6k w - - 0 1"] * (k_steps + 1),
        actions=[-1] * k_steps,
        target_values=[0.0] * (k_steps + 1),
        target_rewards=[0.0] * (k_steps + 1),
        legal_actions=[[0]] * (k_steps + 1),
        optimal_actions=[[0]] * (k_steps + 1),
        mate_step=None,
    )
    tb_batch = build_tb_batch_trajectories([tb_traj], k_steps=k_steps)

    assert set(pgn_batch.keys()) == set(tb_batch.keys())
    for key in tb_batch:
        assert pgn_batch[key].dtype == tb_batch[key].dtype, key
        assert pgn_batch[key].shape[1:] == tb_batch[key].shape[1:], key


def test_ingest_row_shapes_and_supervision_flags():
    """PGN rows are trajectory-shaped, non-tablebase, and full-weight policy."""
    k_steps = 3
    trajectories, _ = _ingest(k_steps=k_steps)
    batch = build_pgn_batch(trajectories, k_steps=k_steps)
    n = len(trajectories)

    assert batch["observations"].shape == (n, k_steps + 1, 110, 8, 8)
    assert batch["actions"].shape == (n, k_steps, 3, 8, 8)
    assert batch["target_policies"].shape == (n, k_steps + 1, NUM_ACTIONS)
    assert not batch["is_tablebase"].any()
    assert not batch["tb_policy_mask"].any()


def test_policy_target_is_one_hot_on_played_move():
    """With no smoothing the root policy target is a single unit mass."""
    k_steps = 3
    trajectories, _ = _ingest(k_steps=k_steps)
    batch = build_pgn_batch(trajectories, k_steps=k_steps)
    root_policy = batch["target_policies"][:, 0, :]
    row_sums = root_policy.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5)
    assert np.all((root_policy > 0).sum(axis=1) == 1)


def test_value_target_sign_flips_with_side_to_move():
    """In the decisive Scholar's-mate game, consecutive-step values alternate sign."""
    k_steps = 3
    # Skip the repetition game (drawn) by ingesting only accepted games and
    # selecting a trajectory from the White-wins game: its values are non-zero
    # and must alternate in sign step to step (STM alternates, discount 1.0).
    trajectories, _ = _ingest(k_steps=k_steps)
    batch = build_pgn_batch(trajectories, k_steps=k_steps)

    decisive = None
    for i in range(batch["target_values"].shape[0]):
        vals = batch["target_values"][i]
        if np.all(np.abs(vals) > 0.5):  # fully decisive window, no absorbing steps
            decisive = vals
            break
    assert decisive is not None, "expected at least one fully-decisive window"
    for k in range(k_steps):
        assert decisive[k] * decisive[k + 1] < 0, (
            f"values must alternate sign at step {k}: {decisive}"
        )


def test_value_discount_decays_toward_root(monkeypatch):
    """HYZERO_PGN_VALUE_DISCOUNT shrinks earlier-step magnitudes below the terminal."""
    monkeypatch.setenv("HYZERO_PGN_VALUE_DISCOUNT", "0.5")
    k_steps = 3
    trajectories, _ = _ingest(k_steps=k_steps)
    batch = build_pgn_batch(trajectories, k_steps=k_steps)
    for i in range(batch["target_values"].shape[0]):
        vals = batch["target_values"][i]
        if np.all(np.abs(vals) > 0.001):
            mags = np.abs(vals)
            assert mags[0] < mags[-1] + 1e-6
            return
    pytest.fail("expected a decisive window to check discount decay")


def test_repetition_plane_set_on_recurring_position():
    """The knight-shuffle game recurs the initial position → plane 102 is set."""
    k_steps = 3
    trajectories, _ = _ingest(k_steps=k_steps)
    # A repeated position must have produced at least one trajectory step whose
    # rep flag is True.
    assert any(any(t.rep_flags) for t in trajectories), "no repetition tracked"

    batch = build_pgn_batch(trajectories, k_steps=k_steps)
    obs = batch["observations"]  # [N, K+1, 110, 8, 8]
    rep_plane = obs[:, :, REP_PLANE_CURRENT, :, :]
    assert rep_plane.max() == 1.0, "repetition plane 102 was never set"


def test_min_elo_filter_skips_low_rated_game():
    """--min-elo 2000 skips the 1000/1100-rated game but keeps the two 2500+ games."""
    _, stats_all = _ingest(k_steps=3)
    _, stats_filtered = _ingest(k_steps=3, min_elo=2000)

    assert stats_all["games_accepted"] == 3
    assert stats_filtered["games_accepted"] == 2
    assert stats_filtered["games_skipped"] >= 1


def test_max_games_limit_caps_accepted_games():
    """--max-games caps the number of accepted games."""
    _, stats = _ingest(k_steps=3, max_games=1)
    assert stats["games_accepted"] == 1


def test_skip_first_n_plies_reduces_positions():
    """Skipping opening plies yields strictly fewer window roots."""
    _, stats0 = _ingest(k_steps=2, skip_first_n_plies=0)
    _, stats2 = _ingest(k_steps=2, skip_first_n_plies=2)
    assert stats2["positions"] < stats0["positions"]


def test_pgn_cache_roundtrip_and_sampling(tmp_path):
    """A pickled PGNCache reloads and samples trajectories the builder accepts."""
    import pickle
    from hyzero.data.pgn_ingest import PGNCache

    k_steps = 3
    trajectories, _ = _ingest(k_steps=k_steps)
    path = tmp_path / "pgn_cache.pkl"
    with open(path, "wb") as f:
        pickle.dump(trajectories, f)

    cache = PGNCache(str(path))
    assert len(cache) == len(trajectories)
    sampled = cache.sample(4)
    batch = build_pgn_batch(sampled, k_steps=k_steps)
    assert batch["observations"].shape[0] == 4
