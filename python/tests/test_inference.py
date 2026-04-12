"""Tests for the MuZero inference server (Task 26)."""

import numpy as np
import pytest
import torch

from hyzero.config import DEFAULT_CONFIG
from hyzero.inference import InferenceServer
from hyzero.training import Trainer

C = DEFAULT_CONFIG["hidden_channels"]           # 64
INPUT_PLANES = DEFAULT_CONFIG["input_planes"]   # 19
NUM_ACTIONS = DEFAULT_CONFIG["num_actions"]     # 4096
ACTION_PLANES = DEFAULT_CONFIG["action_planes"] # 3


@pytest.mark.parametrize("batch_size", [1, 8, 32])
def test_root_setup_batch_shapes(batch_size: int) -> None:
    """root_setup_batch returns correct shapes for various batch sizes."""
    server = InferenceServer(device="cpu")
    obs = np.random.randn(batch_size, INPUT_PLANES, 8, 8).astype(np.float32)

    hidden, policies, values = server.root_setup_batch(obs)

    assert hidden.shape == (batch_size, C, 8, 8), f"hidden: {hidden.shape}"
    assert policies.shape == (batch_size, NUM_ACTIONS), f"policies: {policies.shape}"
    assert values.shape == (batch_size,), f"values: {values.shape}"


@pytest.mark.parametrize("batch_size", [1, 8, 32])
def test_expand_leaf_batch_shapes(batch_size: int) -> None:
    """expand_leaf_batch returns correct shapes for various batch sizes."""
    server = InferenceServer(device="cpu")
    hidden_in = np.random.randn(batch_size, C, 8, 8).astype(np.float32)
    actions = np.random.randn(batch_size, ACTION_PLANES, 8, 8).astype(np.float32)

    new_hidden, rewards, policies, values = server.expand_leaf_batch(hidden_in, actions)

    assert new_hidden.shape == (batch_size, C, 8, 8), f"new_hidden: {new_hidden.shape}"
    assert rewards.shape == (batch_size,), f"rewards: {rewards.shape}"
    assert policies.shape == (batch_size, NUM_ACTIONS), f"policies: {policies.shape}"
    assert values.shape == (batch_size,), f"values: {values.shape}"


def test_policies_sum_to_one() -> None:
    """Softmax policies must sum to approximately 1.0."""
    server = InferenceServer(device="cpu")
    obs = np.random.randn(4, INPUT_PLANES, 8, 8).astype(np.float32)

    _, policies, _ = server.root_setup_batch(obs)
    sums = policies.sum(axis=-1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-5, err_msg="root policies don't sum to 1")

    hidden = np.random.randn(4, C, 8, 8).astype(np.float32)
    actions = np.random.randn(4, ACTION_PLANES, 8, 8).astype(np.float32)
    _, _, expand_policies, _ = server.expand_leaf_batch(hidden, actions)
    sums = expand_policies.sum(axis=-1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-5, err_msg="expand policies don't sum to 1")


def test_values_bounded() -> None:
    """Values must be in [-1, 1] (tanh output)."""
    server = InferenceServer(device="cpu")
    obs = np.random.randn(8, INPUT_PLANES, 8, 8).astype(np.float32)

    _, _, values = server.root_setup_batch(obs)
    assert np.all(np.abs(values) <= 1.0), f"Values out of bounds: min={values.min()}, max={values.max()}"


def test_rewards_bounded() -> None:
    """Rewards must be in [-1, 1] (tanh output)."""
    server = InferenceServer(device="cpu")
    hidden = np.random.randn(8, C, 8, 8).astype(np.float32)
    actions = np.random.randn(8, ACTION_PLANES, 8, 8).astype(np.float32)

    _, rewards, _, _ = server.expand_leaf_batch(hidden, actions)
    assert np.all(np.abs(rewards) <= 1.0), f"Rewards out of bounds: min={rewards.min()}, max={rewards.max()}"


def test_outputs_are_numpy() -> None:
    """All outputs must be numpy arrays, not torch tensors."""
    server = InferenceServer(device="cpu")
    obs = np.random.randn(2, INPUT_PLANES, 8, 8).astype(np.float32)

    hidden, policies, values = server.root_setup_batch(obs)
    for name, arr in [("hidden", hidden), ("policies", policies), ("values", values)]:
        assert isinstance(arr, np.ndarray), f"root {name}: expected np.ndarray, got {type(arr)}"
        assert arr.dtype == np.float32, f"root {name}: expected float32, got {arr.dtype}"

    actions = np.random.randn(2, ACTION_PLANES, 8, 8).astype(np.float32)
    new_hidden, rewards, exp_policies, exp_values = server.expand_leaf_batch(hidden, actions)
    for name, arr in [
        ("new_hidden", new_hidden),
        ("rewards", rewards),
        ("policies", exp_policies),
        ("values", exp_values),
    ]:
        assert isinstance(arr, np.ndarray), f"expand {name}: expected np.ndarray, got {type(arr)}"
        assert arr.dtype == np.float32, f"expand {name}: expected float32, got {arr.dtype}"


def test_load_weights_from_trainer() -> None:
    """load_weights with Trainer.get_weights() bytes must change inference output."""
    torch.manual_seed(0)
    np.random.seed(0)

    server = InferenceServer(device="cpu")
    obs = np.random.randn(2, INPUT_PLANES, 8, 8).astype(np.float32)

    # Capture output before weight sync.
    _, policies_before, values_before = server.root_setup_batch(obs)

    # Train a few steps so weights diverge from init.
    trainer = Trainer(device="cpu")
    batch = {
        "observations": np.random.randn(4, 19, 8, 8).astype(np.float32),
        "actions": np.random.randn(4, 3, 3, 8, 8).astype(np.float32),
        "target_policies": np.full((4, 4, 4096), 1.0 / 4096, dtype=np.float32),
        "target_values": np.zeros((4, 4), dtype=np.float32),
        "target_rewards": np.zeros((4, 4), dtype=np.float32),
    }
    for _ in range(5):
        trainer.train_batch(batch)

    # Sync weights from trainer to server.
    weight_bytes = trainer.get_weights()
    server.load_weights(weight_bytes)

    # Output should now differ from before (trained weights != initial weights).
    _, policies_after, values_after = server.root_setup_batch(obs)

    # At least one element must differ (extremely unlikely to match after 5 training steps).
    assert not np.allclose(policies_before, policies_after, atol=1e-6), (
        "Policies unchanged after load_weights — weights may not have been loaded"
    )


def test_load_weights_produces_consistent_output() -> None:
    """After load_weights, server must produce same output as trainer networks in eval mode."""
    torch.manual_seed(42)
    np.random.seed(42)

    trainer = Trainer(device="cpu")
    batch = {
        "observations": np.random.randn(4, 19, 8, 8).astype(np.float32),
        "actions": np.random.randn(4, 3, 3, 8, 8).astype(np.float32),
        "target_policies": np.full((4, 4, 4096), 1.0 / 4096, dtype=np.float32),
        "target_values": np.zeros((4, 4), dtype=np.float32),
        "target_rewards": np.zeros((4, 4), dtype=np.float32),
    }
    for _ in range(3):
        trainer.train_batch(batch)

    weight_bytes = trainer.get_weights()

    server = InferenceServer(device="cpu")
    server.load_weights(weight_bytes)

    # Compare outputs on the same input.
    obs = np.random.randn(2, INPUT_PLANES, 8, 8).astype(np.float32)
    server_hidden, server_policies, server_values = server.root_setup_batch(obs)

    # Run the same path through trainer networks in eval mode.
    trainer.h.eval()
    trainer.g.eval()
    trainer.f.eval()
    with torch.no_grad():
        obs_t = torch.from_numpy(obs)
        trainer_hidden = trainer.h(obs_t)
        trainer_logits, trainer_value = trainer.f(trainer_hidden)
        trainer_policies = torch.nn.functional.softmax(trainer_logits, dim=-1).numpy()
        trainer_values = trainer_value.squeeze(-1).numpy()
        trainer_hidden_np = trainer_hidden.numpy()

    np.testing.assert_allclose(server_hidden, trainer_hidden_np, atol=1e-5,
                               err_msg="hidden states mismatch after load_weights")
    np.testing.assert_allclose(server_policies, trainer_policies, atol=1e-5,
                               err_msg="policies mismatch after load_weights")
    np.testing.assert_allclose(server_values, trainer_values, atol=1e-5,
                               err_msg="values mismatch after load_weights")
