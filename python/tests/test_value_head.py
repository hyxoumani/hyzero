"""Tests for the distributional (HL-Gauss / categorical) value head.

Covers the HL-Gauss target construction, the categorical expectation round-trip,
byte-identity of the scalar-mode value loss, and the checkpoint mode-mismatch
guard. The default (scalar) mode remains the legacy scalar+tanh head.
"""

import os
import tempfile

import pytest
import torch
import torch.nn.functional as F

from hyzero.config import VALUE_SUPPORT_SIZE
from hyzero.models.prediction import (
    PredictionNetwork,
    build_value_support,
    hl_gauss_target,
)
import numpy as np

from hyzero.config import DEFAULT_CONFIG
from hyzero.training import Trainer

_INPUT_PLANES = DEFAULT_CONFIG["input_planes"]
_NUM_ACTIONS = DEFAULT_CONFIG["num_actions"]


def _make_batch(batch_size: int = 4, k_steps: int = 3) -> dict:
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


def _sigma(support: torch.Tensor) -> float:
    bin_width = float((support[-1] - support[0]) / (support.shape[0] - 1))
    return 0.75 * bin_width


def test_hl_gauss_target_sums_to_one() -> None:
    """Every HL-Gauss target row is a normalized probability distribution."""
    support = build_value_support(VALUE_SUPPORT_SIZE)
    targets = torch.tensor([-1.0, -0.4, 0.0, 0.37, 1.0])
    dist = hl_gauss_target(targets, support, _sigma(support))
    sums = dist.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), sums


def test_hl_gauss_target_peaks_at_correct_bin() -> None:
    """The HL-Gauss distribution peaks at the support atom nearest the target."""
    support = build_value_support(VALUE_SUPPORT_SIZE)
    # Targets chosen to coincide with support atoms (bin_width=0.04) so the
    # nearest bin is unambiguous — a target exactly on a bin boundary is a tie.
    for scalar in (-1.0, -0.6, 0.0, 0.6, 1.0):
        target = torch.tensor([scalar])
        dist = hl_gauss_target(target, support, _sigma(support))
        peak_bin = int(dist.argmax(dim=-1).item())
        nearest_bin = int((support - scalar).abs().argmin().item())
        assert peak_bin == nearest_bin, (
            f"scalar={scalar}: peak bin {peak_bin} != nearest bin {nearest_bin}"
        )


def test_categorical_expectation_round_trips_scalar() -> None:
    """Expectation over the HL-Gauss dist recovers the scalar within bin resolution."""
    support = build_value_support(VALUE_SUPPORT_SIZE)
    bin_width = float((support[-1] - support[0]) / (support.shape[0] - 1))
    net = PredictionNetwork(
        hidden_channels=8, num_actions=16, value_head="categorical",
        value_support_size=VALUE_SUPPORT_SIZE,
    )
    for scalar in (-0.8, -0.3, 0.0, 0.42, 0.9):
        target = torch.tensor([scalar])
        dist = hl_gauss_target(target, support, _sigma(support))
        # Feed log(dist) as logits so softmax recovers dist exactly, then the
        # network's own value_expectation must map back to the scalar.
        logits = dist.clamp(min=1e-12).log()
        recovered = net.value_expectation(logits).item()
        assert abs(recovered - scalar) <= bin_width, (
            f"scalar={scalar} recovered={recovered} (bin_width={bin_width})"
        )


def test_scalar_value_loss_is_byte_identical_to_mse() -> None:
    """Scalar-mode value loss equals plain MSE on a fixed batch (byte-identical)."""
    trainer = Trainer(device="cpu")  # default scalar mode
    assert trainer.value_head_mode == "scalar"
    torch.manual_seed(1234)
    value_out = torch.randn(7, 1)  # [B, 1] raw scalar head output
    targets = torch.randn(7)       # [B]
    got = trainer._value_loss(value_out, targets)
    expected = F.mse_loss(value_out.squeeze(-1), targets)
    assert torch.equal(got, expected), (got, expected)
    # Per-sample path is likewise identical to the squared error.
    got_ps = trainer._value_loss_per_sample(value_out, targets)
    expected_ps = (value_out.squeeze(-1) - targets) ** 2
    assert torch.equal(got_ps, expected_ps)


def test_categorical_value_loss_is_finite_and_positive() -> None:
    """Categorical value loss is a finite cross-entropy over the batch."""
    os.environ["HYZERO_VALUE_HEAD"] = "categorical"
    try:
        trainer = Trainer(device="cpu")
    finally:
        del os.environ["HYZERO_VALUE_HEAD"]
    assert trainer.value_head_mode == "categorical"
    value_out = torch.randn(5, VALUE_SUPPORT_SIZE)
    targets = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    loss = trainer._value_loss(value_out, targets)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_categorical_train_batch_end_to_end() -> None:
    """A full train_batch in categorical mode returns finite losses (diagnostics run)."""
    os.environ["HYZERO_VALUE_HEAD"] = "categorical"
    try:
        trainer = Trainer(device="cpu")
    finally:
        del os.environ["HYZERO_VALUE_HEAD"]
    result = trainer.train_batch(_make_batch())
    for key in ("total_loss", "policy_loss", "value_loss", "reward_loss"):
        assert isinstance(result[key], float)
        assert result[key] == result[key], f"{key} is NaN"  # finite check
    assert result["value_loss"] > 0.0


def test_checkpoint_mode_mismatch_raises() -> None:
    """Loading a scalar checkpoint into a categorical trainer errors clearly."""
    scalar_trainer = Trainer(device="cpu")  # scalar
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tf:
        ckpt_path = tf.name
    try:
        scalar_trainer.save_checkpoint(ckpt_path)

        os.environ["HYZERO_VALUE_HEAD"] = "categorical"
        try:
            cat_trainer = Trainer(device="cpu")
        finally:
            del os.environ["HYZERO_VALUE_HEAD"]

        with pytest.raises(ValueError, match="value-head mode mismatch"):
            cat_trainer.load_checkpoint(ckpt_path)
    finally:
        os.remove(ckpt_path)


def test_checkpoint_same_mode_loads() -> None:
    """A scalar checkpoint loads back into a scalar trainer without error."""
    trainer = Trainer(device="cpu")
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tf:
        ckpt_path = tf.name
    try:
        trainer.save_checkpoint(ckpt_path)
        other = Trainer(device="cpu")
        other.load_checkpoint(ckpt_path)  # must not raise
    finally:
        os.remove(ckpt_path)
