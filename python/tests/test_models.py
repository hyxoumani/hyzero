"""Forward-pass shape and correctness tests for hyzero MuZero models."""

import torch
import pytest

from hyzero.models import (
    ResidualBlock,
    RepresentationNetwork,
    DynamicsNetwork,
    PredictionNetwork,
)
from hyzero.config import DEFAULT_CONFIG

# Fixed seeds for reproducibility; all on CPU.
torch.manual_seed(42)

C = DEFAULT_CONFIG["hidden_channels"]       # 64
INPUT_PLANES = DEFAULT_CONFIG["input_planes"]  # 19
NUM_ACTIONS = DEFAULT_CONFIG["num_actions"]    # 4096
ACTION_PLANES = DEFAULT_CONFIG["action_planes"]  # 3
NUM_RES = DEFAULT_CONFIG["num_res_blocks"]       # 4


def _make_networks():
    rep = RepresentationNetwork(INPUT_PLANES, C, NUM_RES).eval()
    dyn = DynamicsNetwork(C, ACTION_PLANES, NUM_RES).eval()
    pred = PredictionNetwork(C, NUM_ACTIONS).eval()
    return rep, dyn, pred


def test_residual_block_preserves_shape():
    block = ResidualBlock(C).eval()
    x = torch.randn(2, C, 8, 8)
    with torch.no_grad():
        out = block(x)
    assert out.shape == (2, C, 8, 8), f"Expected (2,{C},8,8), got {out.shape}"


def test_representation_network_shape():
    rep = RepresentationNetwork(INPUT_PLANES, C, NUM_RES).eval()
    x = torch.randn(2, INPUT_PLANES, 8, 8)
    with torch.no_grad():
        hidden = rep(x)
    assert hidden.shape == (2, C, 8, 8), f"Expected (2,{C},8,8), got {hidden.shape}"


def test_dynamics_network_shapes():
    dyn = DynamicsNetwork(C, ACTION_PLANES, NUM_RES).eval()
    hidden = torch.randn(2, C, 8, 8)
    action = torch.randn(2, ACTION_PLANES, 8, 8)
    with torch.no_grad():
        next_hidden, reward = dyn(hidden, action)
    assert next_hidden.shape == (2, C, 8, 8), f"Expected (2,{C},8,8), got {next_hidden.shape}"
    assert reward.shape == (2, 1), f"Expected (2,1), got {reward.shape}"


def test_dynamics_reward_bounded():
    dyn = DynamicsNetwork(C, ACTION_PLANES, NUM_RES).eval()
    hidden = torch.randn(2, C, 8, 8)
    action = torch.randn(2, ACTION_PLANES, 8, 8)
    with torch.no_grad():
        _, reward = dyn(hidden, action)
    assert (reward.abs() <= 1.0).all(), "Reward must be in [-1, 1] (tanh output)"


def test_prediction_network_shapes():
    pred = PredictionNetwork(C, NUM_ACTIONS).eval()
    hidden = torch.randn(2, C, 8, 8)
    with torch.no_grad():
        policy_logits, value = pred(hidden)
    assert policy_logits.shape == (2, NUM_ACTIONS), (
        f"Expected (2,{NUM_ACTIONS}), got {policy_logits.shape}"
    )
    assert value.shape == (2, 1), f"Expected (2,1), got {value.shape}"


def test_prediction_value_bounded():
    pred = PredictionNetwork(C, NUM_ACTIONS).eval()
    hidden = torch.randn(2, C, 8, 8)
    with torch.no_grad():
        _, value = pred(hidden)
    assert (value.abs() <= 1.0).all(), "Value must be in [-1, 1] (tanh output)"


def test_prediction_policy_raw_logits():
    """Policy head should return raw logits (not softmax) with correct shape and no NaNs."""
    pred = PredictionNetwork(C, NUM_ACTIONS).eval()
    hidden = torch.randn(2, C, 8, 8)
    with torch.no_grad():
        policy_logits, _ = pred(hidden)
    assert policy_logits.shape == (2, NUM_ACTIONS)
    assert not torch.isnan(policy_logits).any(), "Policy logits contain NaN"


def test_batch_size_1():
    """All three networks must work with batch size 1."""
    rep, dyn, pred = _make_networks()
    obs = torch.randn(1, INPUT_PLANES, 8, 8)
    action = torch.randn(1, ACTION_PLANES, 8, 8)
    with torch.no_grad():
        hidden = rep(obs)
        assert hidden.shape == (1, C, 8, 8)
        next_hidden, reward = dyn(hidden, action)
        assert next_hidden.shape == (1, C, 8, 8)
        assert reward.shape == (1, 1)
        policy_logits, value = pred(hidden)
        assert policy_logits.shape == (1, NUM_ACTIONS)
        assert value.shape == (1, 1)


def test_batch_size_32():
    """All three networks must work with batch size 32."""
    rep, dyn, pred = _make_networks()
    obs = torch.randn(32, INPUT_PLANES, 8, 8)
    action = torch.randn(32, ACTION_PLANES, 8, 8)
    with torch.no_grad():
        hidden = rep(obs)
        assert hidden.shape == (32, C, 8, 8)
        next_hidden, reward = dyn(hidden, action)
        assert next_hidden.shape == (32, C, 8, 8)
        assert reward.shape == (32, 1)
        policy_logits, value = pred(hidden)
        assert policy_logits.shape == (32, NUM_ACTIONS)
        assert value.shape == (32, 1)
