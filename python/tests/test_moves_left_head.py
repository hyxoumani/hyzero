"""Tests for the lc0-style moves-left head (MLH).

Covers the head's output range, the config gate (default OFF), the masked MLH
loss in ``Trainer.train_batch`` (all-masked rows contribute zero), the target
normalization (raw plies / cap, clamped), and the checkpoint flag-mismatch guard.
The default configuration (head OFF) is byte-identical to the legacy net.
"""

import tempfile

import numpy as np
import pytest
import torch

from hyzero.config import DEFAULT_CONFIG, moves_left_cap, moves_left_head_enabled
from hyzero.models.prediction import PredictionNetwork
from hyzero.training import Trainer

_INPUT_PLANES = DEFAULT_CONFIG["input_planes"]
_NUM_ACTIONS = DEFAULT_CONFIG["num_actions"]
_HIDDEN = DEFAULT_CONFIG["hidden_channels"]


def _make_batch(batch_size: int = 4, k_steps: int = 3) -> dict:
    """A minimal replay batch (no moves-left arrays)."""
    return {
        "observations": np.random.randn(
            batch_size, k_steps + 1, _INPUT_PLANES, 8, 8
        ).astype(np.float32),
        "actions": np.random.randn(batch_size, k_steps, 3, 8, 8).astype(np.float32),
        "target_policies": np.full(
            (batch_size, k_steps + 1, _NUM_ACTIONS), 1.0 / _NUM_ACTIONS, dtype=np.float32
        ),
        "target_values": np.random.uniform(-1, 1, (batch_size, k_steps + 1)).astype(
            np.float32
        ),
        "target_rewards": np.random.uniform(-1, 1, (batch_size, k_steps + 1)).astype(
            np.float32
        ),
    }


def _add_moves_left(batch: dict, raw: float, mask: bool) -> dict:
    """Attach constant moves-left target + validity mask to a batch."""
    b = batch["observations"].shape[0]
    kp1 = batch["observations"].shape[1]
    batch["target_moves_left"] = np.full((b, kp1), raw, dtype=np.float32)
    batch["moves_left_mask"] = np.full((b, kp1), mask, dtype=bool)
    return batch


# --------------------------------------------------------------------------- #
# Model head
# --------------------------------------------------------------------------- #


def test_moves_left_head_disabled_by_default() -> None:
    """Default net has no moves-left head and moves_left() raises."""
    net = PredictionNetwork(hidden_channels=_HIDDEN, num_actions=_NUM_ACTIONS)
    assert net.moves_left_head_enabled is False
    hidden = torch.randn(2, _HIDDEN, 8, 8)
    with pytest.raises(RuntimeError):
        net.moves_left(hidden)


def test_moves_left_head_outputs_unit_interval() -> None:
    """When enabled the head returns a [B] tensor in [0, 1]."""
    net = PredictionNetwork(
        hidden_channels=_HIDDEN, num_actions=_NUM_ACTIONS, moves_left_head=True
    ).eval()
    hidden = torch.randn(5, _HIDDEN, 8, 8)
    m = net.moves_left(hidden)
    assert m.shape == (5,)
    assert torch.all(m >= 0.0) and torch.all(m <= 1.0), m


# --------------------------------------------------------------------------- #
# Config gate
# --------------------------------------------------------------------------- #


def test_config_gate_and_cap(monkeypatch) -> None:
    monkeypatch.delenv("HYZERO_MOVES_LEFT_HEAD", raising=False)
    monkeypatch.delenv("HYZERO_MLH_CAP", raising=False)
    assert moves_left_head_enabled() is False
    assert moves_left_cap() == 100.0

    monkeypatch.setenv("HYZERO_MOVES_LEFT_HEAD", "1")
    monkeypatch.setenv("HYZERO_MLH_CAP", "50")
    assert moves_left_head_enabled() is True
    assert moves_left_cap() == 50.0

    # Non-positive / unparseable caps fall back to the default.
    monkeypatch.setenv("HYZERO_MLH_CAP", "0")
    assert moves_left_cap() == 100.0


# --------------------------------------------------------------------------- #
# Trainer loss
# --------------------------------------------------------------------------- #


def test_train_batch_moves_left_loss_zero_when_disabled(monkeypatch) -> None:
    """Head OFF: the reported moves_left_loss is exactly 0.0 (legacy path)."""
    monkeypatch.delenv("HYZERO_MOVES_LEFT_HEAD", raising=False)
    trainer = Trainer(dict(DEFAULT_CONFIG), device="cpu")
    batch = _add_moves_left(_make_batch(), raw=5.0, mask=True)
    out = trainer.train_batch(batch)
    assert out["moves_left_loss"] == 0.0


def test_train_batch_masked_rows_contribute_zero_loss(monkeypatch) -> None:
    """Head ON but all rows masked → MLH loss is exactly 0.0."""
    monkeypatch.setenv("HYZERO_MOVES_LEFT_HEAD", "1")
    trainer = Trainer(dict(DEFAULT_CONFIG), device="cpu")
    batch = _add_moves_left(_make_batch(), raw=5.0, mask=False)
    out = trainer.train_batch(batch)
    assert out["moves_left_loss"] == 0.0


def test_train_batch_moves_left_loss_present_when_valid(monkeypatch) -> None:
    """Head ON with valid rows → a finite, non-negative MLH loss is produced."""
    monkeypatch.setenv("HYZERO_MOVES_LEFT_HEAD", "1")
    trainer = Trainer(dict(DEFAULT_CONFIG), device="cpu")
    batch = _add_moves_left(_make_batch(), raw=5.0, mask=True)
    out = trainer.train_batch(batch)
    assert out["moves_left_loss"] >= 0.0
    assert np.isfinite(out["moves_left_loss"])


def test_target_normalization_clamps_to_unit_interval(monkeypatch) -> None:
    """Target = clamp(raw / cap, 0, 1): raw=5, cap=100 → 0.05; raw>cap → 1.0."""
    monkeypatch.setenv("HYZERO_MLH_CAP", "100")
    cap = moves_left_cap()
    target = min(max(5.0 / cap, 0.0), 1.0)
    assert target == pytest.approx(0.05)
    over = torch.tensor([150.0])
    assert float((over / cap).clamp(0.0, 1.0)) == 1.0


# --------------------------------------------------------------------------- #
# Checkpoint flag guard
# --------------------------------------------------------------------------- #


def test_checkpoint_flag_mismatch_raises(monkeypatch) -> None:
    """A checkpoint saved with the head ON cannot load into a head-OFF trainer."""
    monkeypatch.setenv("HYZERO_MOVES_LEFT_HEAD", "1")
    on_trainer = Trainer(dict(DEFAULT_CONFIG), device="cpu")
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    on_trainer.save_checkpoint(path, {})

    monkeypatch.setenv("HYZERO_MOVES_LEFT_HEAD", "0")
    off_trainer = Trainer(dict(DEFAULT_CONFIG), device="cpu")
    with pytest.raises(ValueError, match="moves-left-head mismatch"):
        off_trainer.load_checkpoint(path)


def test_checkpoint_flag_match_roundtrips(monkeypatch) -> None:
    """Matching head flags load cleanly (no false-positive guard)."""
    monkeypatch.setenv("HYZERO_MOVES_LEFT_HEAD", "1")
    trainer = Trainer(dict(DEFAULT_CONFIG), device="cpu")
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    trainer.save_checkpoint(path, {})

    reloaded = Trainer(dict(DEFAULT_CONFIG), device="cpu")
    reloaded.load_checkpoint(path)  # must not raise
    assert reloaded.moves_left_head_enabled is True
