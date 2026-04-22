"""Tests for the MuZero training loop (Task 25)."""

import math
import os
import tempfile

import numpy as np
import pytest
import torch

from hyzero.config import DEFAULT_CONFIG
from hyzero.training import Trainer
from hyzero.training.trainer import _parse_loss_weight_env, _parse_lr_schedule_env, _reinit_value_head

INPUT_PLANES = DEFAULT_CONFIG["input_planes"]   # 102
NUM_ACTIONS = DEFAULT_CONFIG["num_actions"]     # 4672


def make_random_batch(batch_size: int = 4, k_steps: int = 3) -> dict:
    """Create a random batch compatible with Trainer.train_batch.

    observations shape: [B, K+1, 102, 8, 8] — all K+1 steps for consistency loss.
    """
    return {
        "observations": np.random.randn(batch_size, k_steps + 1, INPUT_PLANES, 8, 8).astype(np.float32),
        "actions": np.random.randn(batch_size, k_steps, 3, 8, 8).astype(np.float32),
        "target_policies": np.full(
            (batch_size, k_steps + 1, NUM_ACTIONS), 1.0 / NUM_ACTIONS, dtype=np.float32
        ),
        "target_values": np.random.uniform(-1, 1, (batch_size, k_steps + 1)).astype(
            np.float32
        ),
        "target_rewards": np.random.uniform(-1, 1, (batch_size, k_steps + 1)).astype(
            np.float32
        ),
    }


def test_train_batch_returns_losses() -> None:
    """train_batch must return a dict with all required keys, all finite."""
    trainer = Trainer(device="cpu")
    batch = make_random_batch(batch_size=4, k_steps=3)
    result = trainer.train_batch(batch)

    required_keys = {"total_loss", "policy_loss", "value_loss", "reward_loss", "consistency_loss", "model_version", "lr"}
    assert required_keys.issubset(set(result.keys())), f"Missing keys: {required_keys - set(result.keys())}"

    for key in ("total_loss", "policy_loss", "value_loss", "reward_loss", "consistency_loss"):
        assert isinstance(result[key], float), f"{key} must be a float, got {type(result[key])}"
        assert math.isfinite(result[key]), f"{key} is not finite: {result[key]}"

    assert result["model_version"] == 1


def test_loss_decreases() -> None:
    """Training on the same batch for 10 steps should reduce total_loss."""
    np.random.seed(42)
    torch.manual_seed(42)

    trainer = Trainer(device="cpu")
    batch = make_random_batch(batch_size=4, k_steps=3)

    initial_loss = trainer.train_batch(batch)["total_loss"]
    for _ in range(9):
        final_loss = trainer.train_batch(batch)["total_loss"]

    assert final_loss < initial_loss, (
        f"Loss did not decrease: initial={initial_loss:.4f}, final={final_loss:.4f}"
    )


def test_checkpoint_roundtrip() -> None:
    """save_checkpoint + load_checkpoint must restore weights so outputs match."""
    np.random.seed(0)
    torch.manual_seed(0)

    trainer_a = Trainer(device="cpu")
    batch = make_random_batch(batch_size=4, k_steps=3)
    trainer_a.train_batch(batch)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt_path = f.name

    try:
        trainer_a.save_checkpoint(ckpt_path, eval_metrics={"test_metric": 1.23})

        trainer_b = Trainer(device="cpu")
        eval_metrics = trainer_b.load_checkpoint(ckpt_path)

        assert eval_metrics == {"test_metric": 1.23}
        assert trainer_b.model_version == trainer_a.model_version

        # Both trainers should produce identical outputs on the same batch.
        torch.manual_seed(99)
        result_a = trainer_a.train_batch(batch)
        torch.manual_seed(99)
        result_b = trainer_b.train_batch(batch)

        assert abs(result_a["total_loss"] - result_b["total_loss"]) < 1e-5, (
            f"total_loss mismatch: {result_a['total_loss']} vs {result_b['total_loss']}"
        )
    finally:
        os.unlink(ckpt_path)


def test_get_weights_nonempty() -> None:
    """get_weights() must return a non-empty bytes object."""
    trainer = Trainer(device="cpu")
    weights = trainer.get_weights()

    assert isinstance(weights, bytes), f"Expected bytes, got {type(weights)}"
    assert len(weights) > 0, "get_weights() returned empty bytes"


def test_kstep_unroll() -> None:
    """K=5 unroll must complete and return finite losses."""
    trainer = Trainer(device="cpu")
    batch = make_random_batch(batch_size=4, k_steps=5)
    result = trainer.train_batch(batch)

    for key in ("total_loss", "policy_loss", "value_loss", "reward_loss", "consistency_loss"):
        assert math.isfinite(result[key]), f"{key} is not finite with K=5: {result[key]}"


def test_gradient_flows() -> None:
    """After train_batch, all three networks must have at least one non-None gradient."""
    trainer = Trainer(device="cpu")
    batch = make_random_batch(batch_size=4, k_steps=3)
    trainer.train_batch(batch)

    for name, network in (("h", trainer.h), ("g", trainer.g), ("f", trainer.f)):
        has_grad = any(
            p.grad is not None for p in network.parameters() if p.requires_grad
        )
        assert has_grad, f"Network '{name}' has no gradients after train_batch"


# --- Loss weight env-var helper tests ---


def test_loss_weight_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset env var returns default of 1.0."""
    monkeypatch.delenv("HYZERO_POLICY_LOSS_WEIGHT", raising=False)
    assert _parse_loss_weight_env("HYZERO_POLICY_LOSS_WEIGHT") == 1.0


def test_loss_weight_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Valid float is parsed; negative is clamped to 0.0; >100 is clamped to 100.0."""
    monkeypatch.setenv("HYZERO_POLICY_LOSS_WEIGHT", "2.5")
    assert _parse_loss_weight_env("HYZERO_POLICY_LOSS_WEIGHT") == 2.5

    monkeypatch.setenv("HYZERO_POLICY_LOSS_WEIGHT", "-1")
    assert _parse_loss_weight_env("HYZERO_POLICY_LOSS_WEIGHT") == 0.0

    monkeypatch.setenv("HYZERO_POLICY_LOSS_WEIGHT", "200")
    assert _parse_loss_weight_env("HYZERO_POLICY_LOSS_WEIGHT") == 100.0


def test_loss_weight_invalid(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Non-numeric env var returns default 1.0 and prints a warning to stderr."""
    monkeypatch.setenv("HYZERO_POLICY_LOSS_WEIGHT", "abc")
    result = _parse_loss_weight_env("HYZERO_POLICY_LOSS_WEIGHT")
    assert result == 1.0
    captured = capsys.readouterr()
    assert "WARNING" in captured.err


# --- LR schedule env-var tests ---


def test_lr_schedule_none_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """No env var set → lr_scheduler is None; train step does not change LR."""
    monkeypatch.delenv("HYZERO_LR_SCHEDULE", raising=False)
    monkeypatch.delenv("HYZERO_LR_COSINE_T_MAX", raising=False)
    monkeypatch.delenv("HYZERO_LR_COSINE_ETA_MIN", raising=False)

    trainer = Trainer(device="cpu")
    assert trainer.lr_scheduler is None

    batch = make_random_batch(batch_size=2, k_steps=2)
    lr_before = trainer.optimizer.param_groups[0]["lr"]
    result = trainer.train_batch(batch)
    lr_after = trainer.optimizer.param_groups[0]["lr"]

    assert lr_before == lr_after, "LR must not change when schedule is none"
    assert "lr" in result
    assert math.isfinite(result["lr"])


def test_lr_schedule_cosine_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    """HYZERO_LR_SCHEDULE=cosine → scheduler is not None; LR decreases monotonically over 3 steps."""
    monkeypatch.setenv("HYZERO_LR_SCHEDULE", "cosine")
    monkeypatch.setenv("HYZERO_LR_COSINE_T_MAX", "100")
    monkeypatch.setenv("HYZERO_LR_COSINE_ETA_MIN", "1e-6")

    trainer = Trainer(device="cpu")
    assert trainer.lr_scheduler is not None

    batch = make_random_batch(batch_size=2, k_steps=2)
    lrs = []
    for _ in range(3):
        result = trainer.train_batch(batch)
        lrs.append(result["lr"])

    assert lrs[0] > lrs[1] > lrs[2], (
        f"LR should decrease monotonically under cosine schedule, got: {lrs}"
    )


def test_lr_schedule_invalid_value(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Unknown HYZERO_LR_SCHEDULE value → warning printed, scheduler is None."""
    monkeypatch.setenv("HYZERO_LR_SCHEDULE", "foo")

    trainer = Trainer(device="cpu")
    assert trainer.lr_scheduler is None

    captured = capsys.readouterr()
    assert "WARNING" in captured.err


# --- Consistency loss tests ---


def test_train_batch_with_consistency_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    """HYZERO_CONSISTENCY_LOSS_WEIGHT=0.5 enables consistency loss: must be finite and >= 0."""
    monkeypatch.setenv("HYZERO_CONSISTENCY_LOSS_WEIGHT", "0.5")

    trainer = Trainer(device="cpu")
    batch = make_random_batch(batch_size=4, k_steps=2)
    result = trainer.train_batch(batch)

    assert "consistency_loss" in result
    assert isinstance(result["consistency_loss"], float)
    assert math.isfinite(result["consistency_loss"])
    assert result["consistency_loss"] >= 0.0


def test_train_batch_consistency_loss_zero_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """HYZERO_CONSISTENCY_LOSS_WEIGHT=0.0 disables consistency loss: must be exactly 0.0."""
    monkeypatch.setenv("HYZERO_CONSISTENCY_LOSS_WEIGHT", "0.0")

    trainer = Trainer(device="cpu")
    batch = make_random_batch(batch_size=4, k_steps=2)
    result = trainer.train_batch(batch)

    assert result["consistency_loss"] == 0.0


# --- Biased value-head reinit tests ---


def test_reinit_value_head_bias_offset(monkeypatch: pytest.MonkeyPatch) -> None:
    """HYZERO_REINIT_VALUE_BIAS=0.3 shifts initial value output to positive half-plane.

    After reinit with bias_offset=+0.3, a forward pass on a random hidden state
    should yield mean output > 0.1 (expected ~tanh(0.3) ≈ 0.29).  This verifies
    the final Linear bias is initialised to the constant rather than zero,
    guaranteeing the value head starts in the positive attractor for TB supervision.
    """
    monkeypatch.setenv("HYZERO_REINIT_VALUE_BIAS", "0.3")

    torch.manual_seed(0)
    trainer = Trainer(device="cpu")

    _reinit_value_head(trainer.f)

    # Forward pass on random hidden states using the default hidden_channels.
    hidden_channels = trainer.config["hidden_channels"]
    dummy_hidden = torch.randn(16, hidden_channels, 8, 8)
    with torch.no_grad():
        _, value = trainer.f(dummy_hidden)  # [16, 1]

    mean_output = value.mean().item()
    assert mean_output > 0.1, (
        f"Expected mean value output > 0.1 after bias_offset=+0.3 reinit, got {mean_output:.4f}"
    )


def test_reinit_value_head_no_bias_offset(monkeypatch: pytest.MonkeyPatch) -> None:
    """HYZERO_REINIT_VALUE_BIAS unset (default 0.0) keeps zero-bias initialisation.

    After reinit without a bias offset, the mean output over many random inputs
    should be close to zero (Kaiming normal weights → zero-mean distribution).
    We allow a generous tolerance of ±0.3 to account for random-seed variance.
    """
    monkeypatch.delenv("HYZERO_REINIT_VALUE_BIAS", raising=False)

    torch.manual_seed(1)
    trainer = Trainer(device="cpu")

    _reinit_value_head(trainer.f)

    hidden_channels = trainer.config["hidden_channels"]
    dummy_hidden = torch.randn(64, hidden_channels, 8, 8)
    with torch.no_grad():
        _, value = trainer.f(dummy_hidden)  # [64, 1]

    mean_output = value.mean().item()
    # With zero bias and symmetric kaiming init the distribution is approximately
    # zero-centred; tanh squashes further toward zero.  Allow ±0.3 tolerance.
    assert abs(mean_output) < 0.3, (
        f"Expected mean value output near 0 without bias offset, got {mean_output:.4f}"
    )
