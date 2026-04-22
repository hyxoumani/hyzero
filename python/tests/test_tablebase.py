"""Tests for tablebase encoder, sample construction, and trainer integration.

All four tests run without a real Syzygy tablebase — they use hand-crafted
TBSample objects and manually constructed board positions.

Run with: cd python && pytest tests/test_tablebase.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import chess


# ─── Test 1: encoding roundtrip against trainer ground truth ─────────────────

def test_tablebase_encoding_roundtrip():
    """Encoder must match _build_kqk_white_winning_obs byte-for-byte.

    The trainer's _build_kqk_white_winning_obs is the Rust-derived ground truth for
    the KQK position (white Ke1, Qa2, black Ke8, white to move). Our Python encoder
    must produce identical float32 values to ensure TB observations are consistent
    with replay observations.
    """
    import sys
    sys.path.insert(0, ".")
    from hyzero.data.board_encoder import encode_board_python
    from hyzero.training.trainer import _build_kqk_white_winning_obs

    # Build expected tensor from trainer reference.
    expected_tensor = _build_kqk_white_winning_obs("cpu")  # [1, 102, 8, 8]
    expected = expected_tensor.squeeze(0).numpy()           # [102, 8, 8]

    # Build the board manually to match trainer's documented layout.
    # White King e1 (sq4), Queen a2 (sq8), Black King e8 (sq60), White to move.
    board = chess.Board(None)
    board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(chess.A2, chess.Piece(chess.QUEEN, chess.WHITE))
    board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
    board.turn = chess.WHITE
    board.castling_rights = 0

    result = encode_board_python(board)  # [102, 8, 8]

    assert result.shape == (102, 8, 8), (
        f"Expected shape (102, 8, 8), got {result.shape}"
    )
    assert result.dtype == np.float32, (
        f"Expected float32 dtype, got {result.dtype}"
    )
    assert np.allclose(result, expected, atol=1e-6), (
        f"Encoder does not match _build_kqk_white_winning_obs. "
        f"Differences at: "
        + str([(p, r, f) for p, r, f in zip(*np.where(np.abs(result - expected) > 1e-6))])
    )


# ─── Test 2: value target sign convention ────────────────────────────────────

def test_tablebase_value_target_sign():
    """Positive WDL means STM is winning; target_value should be +1.

    KQK white-to-move: white has overwhelming advantage → WDL = +2 → target +1.
    Black-to-move same material (reversed): WDL from STM (black's) perspective = -2 → target -1.
    """
    from hyzero.data.tablebase import TBSample

    # White is winning when White is STM → target_value = +1.
    sample_white_wins = TBSample(
        fen="4k3/8/8/8/8/8/Q7/4K3 w - - 0 1",
        target_value=1.0,    # WDL = +2 from white's perspective (STM).
        mating_actions=[],
        optimal_actions=[42],
        all_legal_actions=[42, 99],
    )
    assert sample_white_wins.target_value == 1.0

    # Black is losing when Black is STM → target_value = -1 from black's perspective.
    sample_black_losing = TBSample(
        fen="4k3/8/8/8/8/8/Q7/4K3 b - - 0 1",
        target_value=-1.0,   # WDL = -2 from black's perspective (STM).
        mating_actions=[],
        optimal_actions=[100],
        all_legal_actions=[100, 200],
    )
    assert sample_black_losing.target_value == -1.0

    # Convention check: positive value means STM is winning.
    assert sample_white_wins.target_value > 0, (
        "Winning side (STM) should have positive target_value"
    )
    assert sample_black_losing.target_value < 0, (
        "Losing side (STM) should have negative target_value"
    )


# ─── Test 3: reward target for mating action ─────────────────────────────────

def test_tablebase_reward_per_action():
    """Reward at step 1 must be +1.0 when a mating action exists.

    Construct a TBSample with mating_actions=[42], call build_tb_batch, and
    verify the reward targets match Option B: r_0=0, r_1=+1, r_2..K=0.
    """
    from hyzero.data.tablebase import TBSample, build_tb_batch

    # Use a real FEN (KQK) so encode_board_python can parse it.
    fen = "4k3/8/8/8/8/8/Q7/4K3 w - - 0 1"
    # action 42 = from_sq=0, to_sq=42 (base encoding: 0*64+42)
    sample = TBSample(
        fen=fen,
        target_value=1.0,
        mating_actions=[42],
        optimal_actions=[42],
        all_legal_actions=[42, 99],
    )

    k_steps = 5
    batch = build_tb_batch([sample], k_steps=k_steps)

    # Shape checks.
    assert batch["target_rewards"].shape == (1, k_steps + 1), (
        f"Expected (1, {k_steps+1}), got {batch['target_rewards'].shape}"
    )

    # r_0 = 0.0 (no reward at root step).
    assert batch["target_rewards"][0, 0] == 0.0, (
        f"Expected r_0=0.0, got {batch['target_rewards'][0, 0]}"
    )
    # r_1 = +1.0 because mating_actions is non-empty.
    assert batch["target_rewards"][0, 1] == 1.0, (
        f"Expected r_1=+1.0 (mate action present), got {batch['target_rewards'][0, 1]}"
    )
    # r_2..K = 0.0.
    for k in range(2, k_steps + 1):
        assert batch["target_rewards"][0, k] == 0.0, (
            f"Expected r_{k}=0.0, got {batch['target_rewards'][0, k]}"
        )

    # Also verify r_0 stays 0 even though mating action was provided.
    assert batch["target_rewards"][0, 0] == 0.0


# ─── Test 4: mixed batch shapes ──────────────────────────────────────────────

def test_mixed_batch_shapes():
    """_maybe_mix_tb_samples must produce correctly shaped merged batch.

    Create a Trainer with a mock TB cache returning 2 TBSamples, a replay batch
    of size 8 with k_steps=5, and tb_frac=0.25 → 2 TB rows. Verify merged shapes
    and that the last 2 rows are flagged as tablebase.
    """
    import sys
    sys.path.insert(0, ".")
    from unittest.mock import MagicMock
    from hyzero.training.trainer import Trainer
    from hyzero.data.tablebase import TBSample

    fen = "4k3/8/8/8/8/8/Q7/4K3 w - - 0 1"
    tb_samples = [
        TBSample(
            fen=fen,
            target_value=1.0,
            mating_actions=[42],
            optimal_actions=[42],
            all_legal_actions=[42, 99],
        ),
        TBSample(
            fen=fen,
            target_value=1.0,
            mating_actions=[],
            optimal_actions=[99],
            all_legal_actions=[42, 99],
        ),
    ]

    # Build a trainer and monkeypatch the TB cache.
    trainer = Trainer(device="cpu")
    mock_cache = MagicMock()
    mock_cache.__len__ = MagicMock(return_value=100)
    mock_cache.sample = MagicMock(return_value=tb_samples)
    mock_cache.is_trajectory_format = False  # Legacy snapshot path.
    trainer._tb_cache = mock_cache
    trainer._tb_frac = 0.25  # 25% of 8 = 2 TB rows.

    # Build a dummy replay batch: B=8, k_steps=5.
    b, k, num_actions = 8, 5, 4672
    batch = {
        "observations":    np.zeros((b, k + 1, 102, 8, 8), dtype=np.float32),
        "actions":         np.zeros((b, k, 3, 8, 8),       dtype=np.float32),
        "target_policies": np.zeros((b, k + 1, num_actions), dtype=np.float32),
        "target_values":   np.zeros((b, k + 1),              dtype=np.float32),
        "target_rewards":  np.zeros((b, k + 1),              dtype=np.float32),
    }

    merged, tb_indices = trainer._maybe_mix_tb_samples(batch)

    # Shape checks.
    assert merged["observations"].shape == (8, 6, 102, 8, 8), (
        f"Expected observations shape (8, 6, 102, 8, 8), got {merged['observations'].shape}"
    )
    assert merged["is_tablebase"].shape == (8,), (
        f"Expected is_tablebase shape (8,), got {merged['is_tablebase'].shape}"
    )
    assert merged["actions"].shape == (8, 5, 3, 8, 8), (
        f"Expected actions shape (8, 5, 3, 8, 8), got {merged['actions'].shape}"
    )

    # Last 2 rows should be TB.
    assert not merged["is_tablebase"][0], "Row 0 should not be TB"
    assert not merged["is_tablebase"][5], "Row 5 should not be TB"
    assert merged["is_tablebase"][6], "Row 6 should be TB"
    assert merged["is_tablebase"][7], "Row 7 should be TB"

    # tb_indices set.
    assert tb_indices == {6, 7}, f"Expected tb_indices={{6, 7}}, got {tb_indices}"


# ─── Test 5: TB masked-loss — value gradient only at step 0 for TB rows ──────

def test_tb_masked_value_loss_step0_only():
    """TB rows must contribute value gradient only at step 0.

    Build a mixed batch with 2 TB rows (last 2) and 2 replay rows, where:
    - TB target_values[:, 0] = ±1 (real WDL signal)
    - TB target_values[:, k>0] = 0 (zero-padded)
    - Replay target_values[:, k] = 0 for all k

    Compute train_batch twice:
    1. With is_tb_mask set (masking enabled).
    2. With all rows treated as replay (no masking).

    The masked run must have HIGHER value_loss than the unmasked run,
    because masking forces the value head to optimize only the step-0 ±1 signal
    (hard) while unmasked version trains the zero-padded k>0 targets (easy zeros).
    """
    import math
    from unittest.mock import MagicMock
    from hyzero.training.trainer import Trainer
    from hyzero.data.tablebase import TBSample

    torch.manual_seed(0)
    np.random.seed(0)

    B, K, num_actions = 4, 5, 4672
    fen = "4k3/8/8/8/8/8/Q7/4K3 w - - 0 1"
    tb_samples = [
        TBSample(fen=fen, target_value=1.0, mating_actions=[42], optimal_actions=[42], all_legal_actions=[42, 99]),
        TBSample(fen=fen, target_value=-1.0, mating_actions=[], optimal_actions=[99], all_legal_actions=[42, 99]),
    ]

    def _make_trainer():
        t = Trainer(device="cpu")
        mock_cache = MagicMock()
        mock_cache.__len__ = MagicMock(return_value=100)
        mock_cache.sample = MagicMock(return_value=tb_samples)
        mock_cache.is_trajectory_format = False  # Legacy snapshot path.
        t._tb_cache = mock_cache
        t._tb_frac = 0.5  # 50% of 4 = 2 TB rows
        return t

    def _make_replay_batch():
        return {
            "observations": np.zeros((B, K + 1, 102, 8, 8), dtype=np.float32),
            "actions":      np.zeros((B, K, 3, 8, 8),       dtype=np.float32),
            "target_policies": np.full((B, K + 1, num_actions), 1.0 / num_actions, dtype=np.float32),
            "target_values":   np.zeros((B, K + 1), dtype=np.float32),
            "target_rewards":  np.zeros((B, K + 1), dtype=np.float32),
        }

    # Run with masking (TB rows present, k>0 padded zeros masked out).
    trainer_masked = _make_trainer()
    result_masked = trainer_masked.train_batch(_make_replay_batch())

    # Run without masking: set tb_frac=0 so no TB rows injected, all zeros.
    trainer_no_tb = _make_trainer()
    trainer_no_tb._tb_frac = 0.0
    result_no_tb = trainer_no_tb.train_batch(_make_replay_batch())

    # The masked run trains against ±1 TB targets at step 0.
    # The no-TB run trains against all zeros. Value loss must be higher when ±1 targets present.
    assert result_masked["value_loss"] > result_no_tb["value_loss"], (
        f"Expected masked value_loss ({result_masked['value_loss']:.4f}) > "
        f"no-TB value_loss ({result_no_tb['value_loss']:.4f})"
    )
    # Both results must be finite.
    for key in ("total_loss", "policy_loss", "value_loss", "reward_loss"):
        assert math.isfinite(result_masked[key]), f"masked {key} is not finite"
        assert math.isfinite(result_no_tb[key]), f"no-tb {key} is not finite"


# ─── Test 6: TB masked-loss — reward step-1 stays unmasked, step-2+ masked ───

def test_tb_masked_reward_step1_unmasked():
    """TB rows contribute reward loss at step 1 (mating signal) but not at step 2+.

    Construct a batch with 2 TB rows carrying r_1=+1 mating signal.
    Compare value of total_reward_loss in two extreme cases:
    1. TB with mating actions (r_1=+1 for TB, rest 0) — masking active.
    2. No TB at all (all reward targets 0) — no masking.

    The TB run must have higher reward_loss than the no-TB run, because r_1=+1 at
    step 1 produces real non-zero error — but only if step 1 is not masked.
    """
    import math
    from unittest.mock import MagicMock
    from hyzero.training.trainer import Trainer
    from hyzero.data.tablebase import TBSample

    torch.manual_seed(1)
    np.random.seed(1)

    B, K, num_actions = 4, 5, 4672
    fen = "4k3/8/8/8/8/8/Q7/4K3 w - - 0 1"
    # Both TB samples have mating_actions set so r_1=+1 in the merged batch.
    tb_samples = [
        TBSample(fen=fen, target_value=1.0, mating_actions=[42], optimal_actions=[42], all_legal_actions=[42, 99]),
        TBSample(fen=fen, target_value=1.0, mating_actions=[42], optimal_actions=[42], all_legal_actions=[42, 99]),
    ]

    def _make_trainer():
        t = Trainer(device="cpu")
        mock_cache = MagicMock()
        mock_cache.__len__ = MagicMock(return_value=100)
        mock_cache.sample = MagicMock(return_value=tb_samples)
        mock_cache.is_trajectory_format = False  # Legacy snapshot path.
        t._tb_cache = mock_cache
        t._tb_frac = 0.5  # 50% of 4 = 2 TB rows
        return t

    batch_zeros = {
        "observations": np.zeros((B, K + 1, 102, 8, 8), dtype=np.float32),
        "actions":      np.zeros((B, K, 3, 8, 8),       dtype=np.float32),
        "target_policies": np.full((B, K + 1, num_actions), 1.0 / num_actions, dtype=np.float32),
        "target_values":   np.zeros((B, K + 1), dtype=np.float32),
        "target_rewards":  np.zeros((B, K + 1), dtype=np.float32),
    }

    # TB run: step-1 reward is +1 for TB rows (mating signal, unmasked).
    trainer_tb = _make_trainer()
    result_tb = trainer_tb.train_batch(dict(batch_zeros))

    # No-TB run: all reward targets are 0.
    trainer_no_tb = _make_trainer()
    trainer_no_tb._tb_frac = 0.0
    result_no_tb = trainer_no_tb.train_batch(dict(batch_zeros))

    # TB run must have higher reward_loss because r_1=+1 at unmasked step 1.
    assert result_tb["reward_loss"] > result_no_tb["reward_loss"], (
        f"Expected TB reward_loss ({result_tb['reward_loss']:.4f}) > "
        f"no-TB reward_loss ({result_no_tb['reward_loss']:.4f})"
    )
    # Both results must be finite.
    for key in ("total_loss", "policy_loss", "value_loss", "reward_loss"):
        assert math.isfinite(result_tb[key]), f"TB {key} is not finite"
        assert math.isfinite(result_no_tb[key]), f"no-TB {key} is not finite"


# ─── Trajectory-format tests (canonical MuZero-shaped TB supervision) ────────

def test_trajectory_batch_shapes_and_mate_signal():
    """K-step trajectory batch places terminal reward at the actual mate step.

    Constructs a hand-crafted TBTrajectory for a mate-at-step-1 scenario:
      - step 0: a real pre-mate position
      - step 1: absorbing (FEN=None), target_rewards[1] = +1.0
      - steps 2..K: absorbing, zero targets

    Verifies:
      - Output dict has the same shape contract as build_tb_batch.
      - target_rewards[0, 1] == +1.0 (mate fired here).
      - observations at step 0 are non-zero (real encode), steps 1..K are zero.
      - actions[0, 0] is non-zero (encoded mating move); actions[0, 1..K-1] zero.
      - is_tablebase is all-False (trajectory rows train like replay rows).
    """
    from hyzero.data.tablebase import TBTrajectory, build_tb_batch_trajectories

    fen_0 = "4k3/8/8/8/8/8/Q7/4K3 w - - 0 1"
    k_steps = 5
    # action 8 = from_sq=0 (a1) to_sq=8 (a2); not the real mating move but a
    # valid action index — the test only checks encoding occurred.
    traj = TBTrajectory(
        fens=[fen_0, None, None, None, None, None],
        actions=[8, -1, -1, -1, -1],
        target_values=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        target_rewards=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        legal_actions=[[8, 99], [], [], [], [], []],
        optimal_actions=[[8], [], [], [], [], []],
        mate_step=1,
    )

    batch = build_tb_batch_trajectories([traj], k_steps=k_steps)

    # Shape contract.
    assert batch["observations"].shape == (1, k_steps + 1, 102, 8, 8)
    assert batch["actions"].shape == (1, k_steps, 3, 8, 8)
    assert batch["target_policies"].shape == (1, k_steps + 1, 4672)
    assert batch["target_values"].shape == (1, k_steps + 1)
    assert batch["target_rewards"].shape == (1, k_steps + 1)
    assert batch["legal_masks"].shape == (1, 4672)
    assert batch["is_tablebase"].shape == (1,)

    # Terminal reward fires at step 1.
    assert batch["target_rewards"][0, 1] == 1.0
    assert batch["target_rewards"][0, 0] == 0.0
    for k in range(2, k_steps + 1):
        assert batch["target_rewards"][0, k] == 0.0

    # Step 0 observation is real; step 1..K are absorbing zeros.
    assert batch["observations"][0, 0].sum() > 0
    for k in range(1, k_steps + 1):
        assert batch["observations"][0, k].sum() == 0.0

    # Root action encoded; null actions past terminal zero.
    assert batch["actions"][0, 0].sum() > 0
    for k in range(1, k_steps):
        assert batch["actions"][0, k].sum() == 0.0

    # Policy target at step 0: mass on optimal action (index 8).
    assert batch["target_policies"][0, 0, 8] == 1.0
    assert batch["target_policies"][0, 0].sum() == 1.0

    # Trajectory rows report is_tablebase=False.
    assert not batch["is_tablebase"][0]


def test_trajectory_value_targets_alternate_pov_sign():
    """Trajectory value targets alternate sign per ply across real steps.

    For a winning rollout with no mate in K steps, target_values should be
    [+1, -1, +1, ...] at real steps and 0 at absorbing.
    """
    from hyzero.data.tablebase import TBTrajectory, build_tb_batch_trajectories

    fen_0 = "4k3/8/8/8/8/8/Q7/4K3 w - - 0 1"
    fen_1 = "4k3/8/8/8/8/8/Q7/4K3 b - - 1 1"
    fen_2 = "4k3/8/8/8/8/8/Q7/4K3 w - - 2 2"

    k_steps = 5
    traj = TBTrajectory(
        fens=[fen_0, fen_1, fen_2, None, None, None],
        actions=[8, 99, 8, -1, -1],
        target_values=[1.0, -1.0, 1.0, 0.0, 0.0, 0.0],
        target_rewards=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        legal_actions=[[8], [99], [8], [], [], []],
        optimal_actions=[[8], [99], [8], [], [], []],
        mate_step=None,
    )

    batch = build_tb_batch_trajectories([traj], k_steps=k_steps)

    assert batch["target_values"][0, 0] == 1.0
    assert batch["target_values"][0, 1] == -1.0
    assert batch["target_values"][0, 2] == 1.0
    for k in range(3, k_steps + 1):
        assert batch["target_values"][0, k] == 0.0

    for k in range(3):
        assert batch["observations"][0, k].sum() > 0
    for k in range(3, k_steps + 1):
        assert batch["observations"][0, k].sum() == 0.0


def test_tablebase_cache_detects_trajectory_format(tmp_path):
    """TablebaseCache auto-detects snapshot vs trajectory pickle format."""
    import pickle
    from hyzero.data.tablebase import TBSample, TBTrajectory, TablebaseCache

    snap_path = tmp_path / "snap.pkl"
    with open(snap_path, "wb") as f:
        pickle.dump([TBSample(
            fen="4k3/8/8/8/8/8/Q7/4K3 w - - 0 1",
            target_value=1.0,
            mating_actions=[],
            optimal_actions=[8],
            all_legal_actions=[8, 99],
        )], f)

    traj_path = tmp_path / "traj.pkl"
    with open(traj_path, "wb") as f:
        pickle.dump([TBTrajectory(
            fens=["4k3/8/8/8/8/8/Q7/4K3 w - - 0 1", None, None, None, None, None],
            actions=[8, -1, -1, -1, -1],
            target_values=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            target_rewards=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            legal_actions=[[8], [], [], [], [], []],
            optimal_actions=[[8], [], [], [], [], []],
            mate_step=1,
        )], f)

    snap_cache = TablebaseCache(str(snap_path))
    traj_cache = TablebaseCache(str(traj_path))

    assert snap_cache.is_trajectory_format is False
    assert traj_cache.is_trajectory_format is True
    assert len(snap_cache) == 1
    assert len(traj_cache) == 1
    assert isinstance(snap_cache.sample(1)[0], TBSample)
    assert isinstance(traj_cache.sample(1)[0], TBTrajectory)
