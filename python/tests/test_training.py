"""Tests for the MuZero training loop (Task 25)."""

import math
import os
import tempfile

import numpy as np
import pytest
import torch

from hyzero.config import DEFAULT_CONFIG
from hyzero.training import Trainer
from hyzero.training.trainer import (
    _antisym_probe_n,
    _parse_loss_weight_env,
    _parse_lr_schedule_env,
    _reinit_value_head,
)

INPUT_PLANES = DEFAULT_CONFIG["input_planes"]   # 110
NUM_ACTIONS = DEFAULT_CONFIG["num_actions"]     # 4672


def make_random_batch(batch_size: int = 4, k_steps: int = 3) -> dict:
    """Create a random batch compatible with Trainer.train_batch.

    observations shape: [B, K+1, 110, 8, 8] — all K+1 steps for consistency loss.
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


# --- Value-antisymmetry regularizer tests ---


def _seeded_trainer_loss(batch: dict, seed: int = 7) -> float:
    """Build a deterministically-seeded trainer and return one train_batch total_loss."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    trainer = Trainer(device="cpu")
    return trainer.train_batch(batch)["total_loss"]


def test_antisym_loss_zero_when_weight_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """HYZERO_ANTISYM_LOSS_WEIGHT=0 (and unset) is a true no-op on total_loss.

    Two identically-seeded trainers — one with the weight explicitly 0.0, one with
    it unset — must produce a byte-identical total_loss, proving the regularizer adds
    nothing (no extra forward passes, no loss change) for existing default runs.
    """
    np.random.seed(123)
    batch = make_random_batch(batch_size=4, k_steps=3)

    monkeypatch.delenv("HYZERO_ANTISYM_LOSS_WEIGHT", raising=False)
    loss_unset = _seeded_trainer_loss(batch)

    monkeypatch.setenv("HYZERO_ANTISYM_LOSS_WEIGHT", "0.0")
    loss_zero = _seeded_trainer_loss(batch)

    assert loss_zero == loss_unset, (
        f"weight=0 must match unset exactly: {loss_zero} vs {loss_unset}"
    )


def test_antisym_loss_penalizes_nonantisymmetric_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With weight>0 and a value head that ignores POV, total_loss gains the penalty.

    Stub the prediction head so v(obs)==v(flip(obs))==const != 0, the maximally
    non-antisymmetric case where v+v_flip = 2*const. The squared-sum penalty is then
    strictly positive, so the weighted total_loss must exceed the weight-0 baseline.
    FAILS without the regularizer term (the two losses would be equal).
    """
    np.random.seed(321)
    batch = make_random_batch(batch_size=4, k_steps=3)

    def constant_value_f(self_f, hidden):
        logits = torch.zeros(hidden.shape[0], NUM_ACTIONS, device=hidden.device)
        value = torch.full((hidden.shape[0], 1), 0.5, device=hidden.device)
        return logits, value

    np.random.seed(7)
    torch.manual_seed(7)
    trainer_zero = Trainer(device="cpu")
    trainer_zero.f.forward = constant_value_f.__get__(trainer_zero.f)
    monkeypatch.setenv("HYZERO_ANTISYM_LOSS_WEIGHT", "0.0")
    loss_zero = trainer_zero.train_batch(batch)["total_loss"]

    np.random.seed(7)
    torch.manual_seed(7)
    trainer_pos = Trainer(device="cpu")
    trainer_pos.f.forward = constant_value_f.__get__(trainer_pos.f)
    monkeypatch.setenv("HYZERO_ANTISYM_LOSS_WEIGHT", "1.0")
    loss_pos = trainer_pos.train_batch(batch)["total_loss"]

    assert loss_pos > loss_zero, (
        f"Antisym penalty must raise total_loss: pos={loss_pos} vs zero={loss_zero}"
    )


def test_antisym_probe_n_default_and_clamp(monkeypatch: pytest.MonkeyPatch) -> None:
    """HYZERO_ANTISYM_PROBE_N defaults to 8, parses ints, and clamps to [1, 64]."""
    monkeypatch.delenv("HYZERO_ANTISYM_PROBE_N", raising=False)
    assert _antisym_probe_n() == 8

    monkeypatch.setenv("HYZERO_ANTISYM_PROBE_N", "16")
    assert _antisym_probe_n() == 16

    monkeypatch.setenv("HYZERO_ANTISYM_PROBE_N", "0")
    assert _antisym_probe_n() == 1

    monkeypatch.setenv("HYZERO_ANTISYM_PROBE_N", "999")
    assert _antisym_probe_n() == 64


def _masked_k0_entropy(logits: torch.Tensor, legal_mask: torch.Tensor) -> float:
    """Replicate the trainer's k0 pred_entropy_legal computation.

    Mirrors the masked-softmax + nan_to_num(neginf=0.0) entropy emitted in the
    [policy_stats] diagnostic at trainer.py (k0, guarded by legal_mask is not
    None). Kept in sync with that block.
    """
    masked_logits = logits.masked_fill(~legal_mask, float("-inf"))
    masked_probs = torch.softmax(masked_logits, dim=-1)
    masked_log_probs = torch.log_softmax(masked_logits, dim=-1)
    masked_log_probs = masked_log_probs.nan_to_num(nan=0.0, neginf=0.0)
    return (-masked_probs * masked_log_probs).sum(dim=-1).mean().item()


def test_pred_entropy_legal_peaked_distribution_is_low() -> None:
    """A peaked masked distribution yields near-zero legal entropy."""
    legal_mask = torch.zeros(1, NUM_ACTIONS, dtype=torch.bool)
    legal_mask[0, :4] = True  # 4 legal moves
    logits = torch.full((1, NUM_ACTIONS), -1e9)
    logits[0, 0] = 50.0  # essentially all mass on one legal move
    assert _masked_k0_entropy(logits, legal_mask) == pytest.approx(0.0, abs=1e-4)


def test_pred_entropy_legal_uniform_over_legal_is_log_n() -> None:
    """A uniform-over-legal distribution yields log(n_legal) legal entropy."""
    n_legal = 5
    legal_mask = torch.zeros(1, NUM_ACTIONS, dtype=torch.bool)
    legal_mask[0, :n_legal] = True
    logits = torch.zeros(1, NUM_ACTIONS)  # equal logits → uniform after masking
    assert _masked_k0_entropy(logits, legal_mask) == pytest.approx(
        math.log(n_legal), abs=1e-5
    )


# --- HYZERO_TB_POLICY_WEIGHT (tablebase policy-CE gating) tests ---


def _make_tb_policy_batch(
    n_replay: int = 2,
    n_tb: int = 2,
    k_steps: int = 2,
    seed: int = 0,
    tb_policy: np.ndarray | None = None,
    tb_value_fill: float | None = None,
) -> dict:
    """Build a mixed replay+TB-trajectory batch with a tb_policy_mask.

    Replay rows come first, TB-trajectory rows last (matching _maybe_mix_tb_samples
    ordering). is_tablebase is all-False (trajectory regime: real k>=1 targets);
    tb_policy_mask is True only on the trailing TB rows.

    obs/actions/values/rewards/legal_masks are shared (drawn from one seed) so that
    BatchNorm statistics are identical across batches that differ only in the TB
    rows' policy or value targets — letting tests isolate the gated policy path.
    """
    rng = np.random.RandomState(seed)
    b = n_replay + n_tb
    obs = rng.randn(b, k_steps + 1, INPUT_PLANES, 8, 8).astype(np.float32)
    actions = rng.randn(b, k_steps, 3, 8, 8).astype(np.float32)
    target_values = rng.uniform(-1, 1, (b, k_steps + 1)).astype(np.float32)
    target_rewards = rng.uniform(-1, 1, (b, k_steps + 1)).astype(np.float32)

    # Replay policy targets: peaked on a single (distinct) action per row.
    target_policies = np.zeros((b, k_steps + 1, NUM_ACTIONS), dtype=np.float32)
    for i in range(b):
        for k in range(k_steps + 1):
            target_policies[i, k, (i + k) % NUM_ACTIONS] = 1.0
    # Override TB-row policy targets if requested (e.g. uniform-over-optimal).
    if tb_policy is not None:
        target_policies[n_replay:] = tb_policy

    # Legal masks: all-True so the k0 legal-mask path engages without dropping mass.
    legal_masks = np.ones((b, NUM_ACTIONS), dtype=bool)

    is_tablebase = np.zeros(b, dtype=bool)            # trajectory regime
    tb_policy_mask = np.zeros(b, dtype=bool)
    tb_policy_mask[n_replay:] = True                  # only TB rows gated

    if tb_value_fill is not None:
        target_values[n_replay:] = tb_value_fill

    return {
        "observations": obs,
        "actions": actions,
        "target_policies": target_policies,
        "target_values": target_values,
        "target_rewards": target_rewards,
        "legal_masks": legal_masks,
        "is_tablebase": is_tablebase,
        "tb_policy_mask": tb_policy_mask,
    }


def test_tb_policy_weight_zero_excludes_tb_policy_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """w_tb=0.0: TB-row policy targets do not affect policy/total loss at any k."""
    monkeypatch.setenv("HYZERO_TB_POLICY_WEIGHT", "0.0")

    uniform_tb = np.full(
        (2, 3, NUM_ACTIONS), 1.0 / NUM_ACTIONS, dtype=np.float32
    )  # n_tb=2, k_steps+1=3
    peaked_tb = np.zeros((2, 3, NUM_ACTIONS), dtype=np.float32)
    peaked_tb[:, :, 123] = 1.0

    batch_a = _make_tb_policy_batch(k_steps=2, seed=1, tb_policy=uniform_tb)
    batch_b = _make_tb_policy_batch(k_steps=2, seed=1, tb_policy=peaked_tb)

    res_a = _seeded_trainer_loss_full(batch_a)
    res_b = _seeded_trainer_loss_full(batch_b)

    # TB policy targets are zero-weighted → policy CE (k0 and k>=1 fold into the
    # reported avg policy_loss) is identical regardless of those targets.
    assert res_a["policy_loss"] == pytest.approx(res_b["policy_loss"], abs=1e-6)
    assert res_a["total_loss"] == pytest.approx(res_b["total_loss"], abs=1e-6)


def test_tb_policy_weight_zero_preserves_tb_value_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """w_tb=0.0: TB-row VALUE targets still move total loss (value path intact)."""
    monkeypatch.setenv("HYZERO_TB_POLICY_WEIGHT", "0.0")

    uniform_tb = np.full((2, 3, NUM_ACTIONS), 1.0 / NUM_ACTIONS, dtype=np.float32)
    batch_lo = _make_tb_policy_batch(
        k_steps=2, seed=2, tb_policy=uniform_tb, tb_value_fill=-1.0
    )
    batch_hi = _make_tb_policy_batch(
        k_steps=2, seed=2, tb_policy=uniform_tb, tb_value_fill=1.0
    )

    res_lo = _seeded_trainer_loss_full(batch_lo)
    res_hi = _seeded_trainer_loss_full(batch_hi)

    # Even with TB policy gradient disabled, TB value supervision (k0 and k>=1)
    # still flows for trajectory rows, so changing the TB value target must move
    # both the value loss and the total loss.
    assert res_lo["value_loss"] != pytest.approx(res_hi["value_loss"], abs=1e-6)
    assert res_lo["total_loss"] != pytest.approx(res_hi["total_loss"], abs=1e-6)


def test_tb_policy_weight_one_matches_unmasked_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """w_tb=1.0 (default): masked batch total loss equals the unmasked-batch loss."""
    monkeypatch.setenv("HYZERO_TB_POLICY_WEIGHT", "1.0")

    uniform_tb = np.full((2, 3, NUM_ACTIONS), 1.0 / NUM_ACTIONS, dtype=np.float32)
    batch_masked = _make_tb_policy_batch(k_steps=2, seed=3, tb_policy=uniform_tb)
    # Same inputs, but strip the TB bookkeeping so the trainer takes the plain
    # (unmasked) replay path. With weight 1.0 the two must be bit-for-bit equal.
    batch_plain = dict(batch_masked)
    del batch_plain["is_tablebase"]
    del batch_plain["tb_policy_mask"]

    res_masked = _seeded_trainer_loss_full(batch_masked)
    res_plain = _seeded_trainer_loss_full(batch_plain)

    assert res_masked["total_loss"] == pytest.approx(res_plain["total_loss"], abs=1e-6)
    assert res_masked["policy_loss"] == pytest.approx(
        res_plain["policy_loss"], abs=1e-6
    )


def test_policy_loss_weight_zero_equals_replay_only_ce() -> None:
    """Weighted policy CE with TB rows at weight 0 equals the replay-only mean CE.

    Unit-level proof (no BatchNorm coupling): a per-row weight of [1,1,0,0] makes
    the weighted mean collapse to the plain mean over the two replay rows, with
    the k0 legal-mask + nan_to_num semantics preserved.
    """
    trainer = Trainer(device="cpu")
    torch.manual_seed(11)
    b = 4
    logits = torch.randn(b, NUM_ACTIONS)
    targets = torch.zeros(b, NUM_ACTIONS)
    for i in range(b):
        targets[i, i] = 1.0  # peaked target per row
    legal_mask = torch.zeros(b, NUM_ACTIONS, dtype=torch.bool)
    legal_mask[:, :8] = True  # 8 legal moves per row

    row_weight = torch.tensor([1.0, 1.0, 0.0, 0.0])  # TB rows (2,3) zero-weighted
    weighted = trainer._policy_loss(logits, targets, legal_mask, row_weight=row_weight)
    replay_only = trainer._policy_loss(
        logits[:2], targets[:2], legal_mask[:2]
    )  # plain mean over replay rows
    assert weighted.item() == pytest.approx(replay_only.item(), abs=1e-6)


def test_pred_entropy_legal_replay_excludes_tb_rows(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """pred_entropy_legal_replay measures only replay rows (TB rows excluded).

    Replay rows are forced to a single legal move → their legal-masked entropy is
    exactly 0 regardless of logits. TB rows get many legal moves → positive
    entropy. The blended pred_entropy_legal therefore exceeds the replay-only
    pred_entropy_legal_replay, which must read 0.
    """
    monkeypatch.setenv("HYZERO_TB_POLICY_WEIGHT", "0.0")

    n_replay, n_tb, k_steps = 2, 2, 1
    uniform_tb = np.full(
        (n_tb, k_steps + 1, NUM_ACTIONS), 1.0 / NUM_ACTIONS, dtype=np.float32
    )
    batch = _make_tb_policy_batch(
        n_replay=n_replay, n_tb=n_tb, k_steps=k_steps, seed=4, tb_policy=uniform_tb
    )
    # Replay rows: exactly ONE legal move each → 0 legal entropy (point mass).
    # TB rows: many legal moves → positive legal entropy.
    legal = np.zeros((n_replay + n_tb, NUM_ACTIONS), dtype=bool)
    for i in range(n_replay):
        legal[i, i] = True
    legal[n_replay:, :64] = True
    batch["legal_masks"] = legal

    np.random.seed(4)
    torch.manual_seed(4)
    trainer = Trainer(device="cpu")
    trainer.train_batch(batch)

    out = capsys.readouterr().out
    pol_line = next(
        (ln for ln in out.splitlines() if ln.startswith("[policy_stats]")), None
    )
    assert pol_line is not None, "expected a [policy_stats] diagnostic line"
    replay_ent = _parse_metric(pol_line, "pred_entropy_legal_replay")
    blended_ent = _parse_metric(pol_line, "pred_entropy_legal")
    assert replay_ent is not None, f"missing pred_entropy_legal_replay in: {pol_line}"
    assert "pred_top1_replay" in pol_line
    assert replay_ent == pytest.approx(0.0, abs=1e-4)
    assert blended_ent > replay_ent + 1e-3


def _seeded_trainer_loss_full(batch: dict, seed: int = 7) -> dict:
    """Like _seeded_trainer_loss but returns the full train_batch result dict."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    trainer = Trainer(device="cpu")
    return trainer.train_batch(batch)


def _parse_metric(line: str, key: str) -> float | None:
    """Extract a ``key=value`` float from a whitespace-delimited diagnostic line.

    Matches the exact token ``{key}=`` so that prefix keys (pred_entropy_legal)
    do not accidentally capture suffixed ones (pred_entropy_legal_replay).
    """
    for tok in line.split():
        if tok.startswith(key + "="):
            return float(tok.split("=", 1)[1])
    return None
