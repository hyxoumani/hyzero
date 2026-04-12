"""Tests for the MuZero training loop (Task 25)."""

import math
import os
import tempfile

import numpy as np
import pytest
import torch

from hyzero.training import Trainer


def make_random_batch(batch_size: int = 4, k_steps: int = 3) -> dict:
    """Create a random batch compatible with Trainer.train_batch."""
    return {
        "observations": np.random.randn(batch_size, 19, 8, 8).astype(np.float32),
        "actions": np.random.randn(batch_size, k_steps, 3, 8, 8).astype(np.float32),
        "target_policies": np.full(
            (batch_size, k_steps + 1, 4096), 1.0 / 4096, dtype=np.float32
        ),
        "target_values": np.random.uniform(-1, 1, (batch_size, k_steps + 1)).astype(
            np.float32
        ),
        "target_rewards": np.random.uniform(-1, 1, (batch_size, k_steps + 1)).astype(
            np.float32
        ),
    }


def test_train_batch_returns_losses() -> None:
    """train_batch must return a dict with all 5 required keys, all finite."""
    trainer = Trainer(device="cpu")
    batch = make_random_batch(batch_size=4, k_steps=3)
    result = trainer.train_batch(batch)

    required_keys = {"total_loss", "policy_loss", "value_loss", "reward_loss", "model_version"}
    assert required_keys == set(result.keys()), f"Missing keys: {required_keys - set(result.keys())}"

    for key in ("total_loss", "policy_loss", "value_loss", "reward_loss"):
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

    for key in ("total_loss", "policy_loss", "value_loss", "reward_loss"):
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
