"""MuZero training loop with K-step unroll, loss computation, and checkpointing."""

import io
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from hyzero.config import DEFAULT_CONFIG
from hyzero.models.representation import RepresentationNetwork
from hyzero.models.dynamics import DynamicsNetwork
from hyzero.models.prediction import PredictionNetwork


def _parse_lr_schedule_env() -> tuple[str, int, float]:
    """Parse LR schedule env vars, returning (schedule, t_max, eta_min).

    Env vars:
        HYZERO_LR_SCHEDULE:      "none" (default) or "cosine". Unknown value warns and uses "none".
        HYZERO_LR_COSINE_T_MAX:  int, total annealing steps. Default 5000. Clamped to [100, 1_000_000].
        HYZERO_LR_COSINE_ETA_MIN: float, minimum LR. Default 1e-5. Clamped to [0.0, 1e-2].

    Returns:
        Tuple of (schedule_name, t_max, eta_min).
    """
    # --- schedule name ---
    raw_sched = os.environ.get("HYZERO_LR_SCHEDULE", "none").strip().lower()
    if raw_sched not in ("none", "cosine"):
        print(
            f"[trainer] WARNING: HYZERO_LR_SCHEDULE={raw_sched!r} is not valid"
            " (expected 'none' or 'cosine'); using 'none'",
            file=sys.stderr,
        )
        raw_sched = "none"

    # --- T_max ---
    raw_tmax = os.environ.get("HYZERO_LR_COSINE_T_MAX")
    t_max = 5000
    if raw_tmax is not None:
        try:
            t_max = int(raw_tmax)
        except (ValueError, TypeError):
            print(
                f"[trainer] WARNING: HYZERO_LR_COSINE_T_MAX={raw_tmax!r} is not a valid int;"
                " using default 5000",
                file=sys.stderr,
            )
            t_max = 5000
        else:
            clamped = max(100, min(1_000_000, t_max))
            if clamped != t_max:
                print(
                    f"[trainer] WARNING: HYZERO_LR_COSINE_T_MAX={t_max} clamped to {clamped}",
                    file=sys.stderr,
                )
            t_max = clamped

    # --- eta_min ---
    raw_eta = os.environ.get("HYZERO_LR_COSINE_ETA_MIN")
    eta_min = 1e-5
    if raw_eta is not None:
        try:
            eta_min = float(raw_eta)
        except (ValueError, TypeError):
            print(
                f"[trainer] WARNING: HYZERO_LR_COSINE_ETA_MIN={raw_eta!r} is not a valid float;"
                " using default 1e-5",
                file=sys.stderr,
            )
            eta_min = 1e-5
        else:
            clamped = max(0.0, min(1e-2, eta_min))
            if clamped != eta_min:
                print(
                    f"[trainer] WARNING: HYZERO_LR_COSINE_ETA_MIN={eta_min} clamped to {clamped}",
                    file=sys.stderr,
                )
            eta_min = clamped

    return raw_sched, t_max, eta_min


def _parse_loss_weight_env(name: str, default: float = 1.0) -> float:
    """Parse a loss weight env var, clamping to [0.0, 100.0].

    Args:
        name:    Environment variable name.
        default: Value to return on missing or unparseable input.

    Returns:
        Parsed and clamped float weight.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except (ValueError, TypeError):
        print(
            f"[trainer] WARNING: {name}={raw!r} is not a valid float; using default {default}",
            file=sys.stderr,
        )
        return default
    clamped = max(0.0, min(100.0, value))
    if clamped != value:
        print(
            f"[trainer] WARNING: {name}={value} clamped to {clamped}",
            file=sys.stderr,
        )
    return clamped


class Trainer:
    """Manages MuZero training: K-step unroll, loss computation, and weight checkpointing.

    Attributes:
        config: Hyperparameter dictionary.
        device: Torch device string ("cpu" or "cuda").
        model_version: Number of train_batch calls completed.
        h: RepresentationNetwork.
        g: DynamicsNetwork.
        f: PredictionNetwork.
        optimizer: Shared AdamW optimizer over all three networks.
    """

    def __init__(self, config: dict = None, device: str = "cpu") -> None:
        self.config = config if config is not None else dict(DEFAULT_CONFIG)
        self.device = device

        cfg = self.config
        self.h = RepresentationNetwork(
            input_planes=cfg["input_planes"],
            hidden_channels=cfg["hidden_channels"],
            num_res_blocks=cfg["num_res_blocks"],
        ).to(device).train()

        self.g = DynamicsNetwork(
            hidden_channels=cfg["hidden_channels"],
            action_planes=cfg["action_planes"],
            num_res_blocks=cfg["num_res_blocks"],
        ).to(device).train()

        self.f = PredictionNetwork(
            hidden_channels=cfg["hidden_channels"],
            num_actions=cfg["num_actions"],
        ).to(device).train()

        all_params = (
            list(self.h.parameters())
            + list(self.g.parameters())
            + list(self.f.parameters())
        )
        self.optimizer = torch.optim.AdamW(
            all_params,
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
        )

        lr_schedule, lr_t_max, lr_eta_min = _parse_lr_schedule_env()
        if lr_schedule == "cosine":
            self.lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = (
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=lr_t_max,
                    eta_min=lr_eta_min,
                )
            )
            print(f"[trainer] LR schedule: cosine T_max={lr_t_max} eta_min={lr_eta_min}")
        else:
            self.lr_scheduler = None
            print(f"[trainer] LR schedule: none (fixed lr={cfg['lr']})")

        self.model_version: int = 0

        self.policy_loss_weight = _parse_loss_weight_env("HYZERO_POLICY_LOSS_WEIGHT")
        self.value_loss_weight = _parse_loss_weight_env("HYZERO_VALUE_LOSS_WEIGHT")
        self.reward_loss_weight = _parse_loss_weight_env("HYZERO_REWARD_LOSS_WEIGHT")
        print(
            f"[trainer] loss weights: policy={self.policy_loss_weight:.2f}"
            f" value={self.value_loss_weight:.2f}"
            f" reward={self.reward_loss_weight:.2f}"
        )

    def train_batch(self, batch: dict) -> dict:
        """Run one K-step unroll training step.

        Args:
            batch: Dictionary of numpy arrays:
                "observations":    [B, K+1, 102, 8, 8]  — all K+1 steps for consistency loss
                "actions":         [B, K, 3, 8, 8]
                "target_policies": [B, K+1, 4672]
                "target_values":   [B, K+1]
                "target_rewards":  [B, K+1]
                "legal_masks":     [B, 4672] bool (optional) — if present, illegal
                                   actions are masked to -inf in the policy loss so
                                   gradients do not push logits at illegal positions.

        Returns:
            dict with keys: total_loss, policy_loss, value_loss, reward_loss,
            consistency_loss, model_version, lr (all loss values are Python floats).
        """
        # Convert numpy arrays to tensors on the target device.
        # observations: [B, K+1, 102, 8, 8] — step 0 is root, steps 1..K for consistency
        obs_all = torch.from_numpy(batch["observations"]).to(self.device)      # [B, K+1, 102, 8, 8]
        obs = obs_all[:, 0]                                                     # [B, 102, 8, 8]
        actions = torch.from_numpy(batch["actions"]).to(self.device)           # [B, K, 3, 8, 8]
        tgt_policies = torch.from_numpy(batch["target_policies"]).to(self.device)  # [B, K+1, 4672]
        tgt_values = torch.from_numpy(batch["target_values"]).to(self.device)  # [B, K+1]
        tgt_rewards = torch.from_numpy(batch["target_rewards"]).to(self.device)  # [B, K+1]
        # Legal mask (root step only): [B, 4672] bool, or None if not provided.
        legal_mask_np = batch.get("legal_masks")
        legal_mask = (
            torch.from_numpy(legal_mask_np).to(self.device)
            if legal_mask_np is not None
            else None
        )

        k_steps = actions.shape[1]  # K

        self.optimizer.zero_grad()

        total_policy_loss = torch.tensor(0.0, device=self.device)
        total_value_loss = torch.tensor(0.0, device=self.device)
        total_reward_loss = torch.tensor(0.0, device=self.device)

        # Collect hidden states at each step for consistency loss computation.
        # hidden_states[k] = latent output of g after k dynamics steps
        # (hidden_states[0] = h(obs_root), hidden_states[k] = g(hidden_{k-1}, a_{k-1}))
        hidden_states: list[torch.Tensor] = []

        # Step 0: encode observation, predict (policy, value).
        hidden = self.h(obs)  # [B, hidden_channels, 8, 8]
        hidden_states.append(hidden)
        policy_logits, value = self.f(hidden)  # [B, 4672], [B, 1]

        # Policy loss at step 0 — apply legal mask if provided.
        total_policy_loss = total_policy_loss + self._policy_loss(
            policy_logits, tgt_policies[:, 0], legal_mask
        )
        # Value loss at step 0.
        total_value_loss = total_value_loss + F.mse_loss(value.squeeze(-1), tgt_values[:, 0])

        # Steps 1..K: unroll dynamics.
        for k in range(1, k_steps + 1):
            action_plane = actions[:, k - 1]  # [B, 3, 8, 8]
            hidden, reward = self.g(hidden, action_plane)  # [B, hidden_channels, 8, 8], [B, 1]
            hidden_states.append(hidden)

            # MuZero: scale gradient at dynamics boundary (Appendix G) to stabilize K-step unroll
            hidden.register_hook(lambda grad: grad * 0.5)

            policy_logits, value = self.f(hidden)  # [B, 4672], [B, 1]

            # No mask for latent steps (operating in learned latent space, not real board)
            total_policy_loss = total_policy_loss + self._policy_loss(policy_logits, tgt_policies[:, k])
            total_value_loss = total_value_loss + F.mse_loss(value.squeeze(-1), tgt_values[:, k])
            total_reward_loss = total_reward_loss + F.mse_loss(reward.squeeze(-1), tgt_rewards[:, k])

        # Average losses over K+1 steps for policy/value; rewards only over K steps (none at step 0).
        n_steps = k_steps + 1
        avg_policy_loss = total_policy_loss / n_steps
        avg_value_loss = total_value_loss / n_steps
        avg_reward_loss = total_reward_loss / k_steps

        # EfficientZero self-supervised consistency loss (Ye et al., NeurIPS 2021).
        # For each dynamics step k in 1..K, force:
        #   g(h(obs_{k-1}), a_{k-1}) ≈ h(obs_k)
        # via cosine similarity, with stop-gradient on the target (h(obs_k)) side.
        # This gives the dynamics network `g` a DIRECT training signal independent of `f`.
        consistency_weight = _parse_loss_weight_env("HYZERO_CONSISTENCY_LOSS_WEIGHT", default=0.5)
        consistency_loss = torch.tensor(0.0, device=self.device)
        if consistency_weight > 0 and k_steps > 0:
            for k_idx in range(1, k_steps + 1):
                # Dynamics branch: project -> predict (online branch, receives gradients)
                dyn_latent_k = hidden_states[k_idx]  # [B, C, 8, 8]
                p1 = self.h.predict(self.h.project(dyn_latent_k))  # [B, proj_dim]
                # Target branch: h(obs_k) projected with stop-gradient (no gradient flows here)
                obs_k = obs_all[:, k_idx]  # [B, 102, 8, 8]
                target_latent = self.h(obs_k)  # [B, C, 8, 8]
                p2 = self.h.project(target_latent).detach()  # [B, proj_dim], stop-grad
                # Cosine similarity loss: 1 - cos_sim (maximize similarity → minimize loss)
                consistency_loss = consistency_loss + (
                    1 - F.cosine_similarity(p1, p2, dim=-1).mean()
                )
            consistency_loss = consistency_loss / k_steps

        total_loss = (
            self.policy_loss_weight * avg_policy_loss
            + self.value_loss_weight * avg_value_loss
            + self.reward_loss_weight * avg_reward_loss
            + consistency_weight * consistency_loss
        )

        total_loss.backward()
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        self.model_version += 1

        return {
            "total_loss": total_loss.item(),
            "policy_loss": avg_policy_loss.item(),
            "value_loss": avg_value_loss.item(),
            "reward_loss": avg_reward_loss.item(),
            "consistency_loss": consistency_loss.item(),
            "model_version": self.model_version,
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    def _policy_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        legal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Cross-entropy between logits and a target probability distribution.

        Args:
            logits:      [B, num_actions]
            targets:     [B, num_actions]  (soft probability distribution summing to 1)
            legal_mask:  [B, num_actions] bool or None.
                         If provided, illegal positions are masked to -inf before
                         log_softmax so gradients do not push logits at illegal positions.

        Returns:
            Scalar tensor (mean over batch).
        """
        if legal_mask is not None:
            logits = logits.masked_fill(~legal_mask, float('-inf'))
        log_probs = F.log_softmax(logits, dim=-1)
        # Replace -inf (from masked illegal actions) with 0.0 before multiplying by targets.
        # Since targets are always 0.0 at illegal positions, 0.0 * 0.0 = 0.0 is correct.
        # This avoids 0.0 * (-inf) = NaN in IEEE 754 arithmetic.
        log_probs = log_probs.nan_to_num(nan=0.0, neginf=0.0)
        return -torch.sum(targets * log_probs, dim=-1).mean()

    def get_weights(self) -> bytes:
        """Serialize network weights to bytes for inference server transfer.

        Does not include optimizer state or model_version — use save_checkpoint
        for full training state persistence.

        Returns:
            bytes object containing the serialized weights.
        """
        buf = io.BytesIO()
        torch.save(
            {
                "h": self.h.state_dict(),
                "g": self.g.state_dict(),
                "f": self.f.state_dict(),
            },
            buf,
        )
        return buf.getvalue()

    def save_checkpoint(self, path: str, eval_metrics: dict = None) -> None:
        """Save network weights, optimizer state, and metadata to disk.

        Args:
            path:         File path to write the checkpoint to.
            eval_metrics: Optional dict of evaluation metrics to persist.
        """
        checkpoint = {
            "h": self.h.state_dict(),
            "g": self.g.state_dict(),
            "f": self.f.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "model_version": self.model_version,
            "eval_metrics": eval_metrics,
        }
        if self.lr_scheduler is not None:
            checkpoint["lr_scheduler"] = self.lr_scheduler.state_dict()
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> dict:
        """Restore network weights, optimizer state, and model_version from disk.

        Args:
            path: File path to load the checkpoint from.

        Returns:
            eval_metrics dict that was stored in the checkpoint (may be None).
        """
        # weights_only=False: checkpoint contains eval_metrics dict alongside tensors
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.h.load_state_dict(checkpoint["h"])
        self.g.load_state_dict(checkpoint["g"])
        self.f.load_state_dict(checkpoint["f"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.model_version = checkpoint["model_version"]
        if self.lr_scheduler is not None and "lr_scheduler" in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        return checkpoint.get("eval_metrics")
