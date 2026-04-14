"""MuZero training loop with K-step unroll, loss computation, and checkpointing."""

import io

import numpy as np
import torch
import torch.nn.functional as F

from hyzero.config import DEFAULT_CONFIG
from hyzero.models.representation import RepresentationNetwork
from hyzero.models.dynamics import DynamicsNetwork
from hyzero.models.prediction import PredictionNetwork


class Trainer:
    """Manages MuZero training: K-step unroll, loss computation, and weight checkpointing.

    Attributes:
        config: Hyperparameter dictionary.
        device: Torch device string ("cpu" or "cuda").
        model_version: Number of train_batch calls completed.
        h: RepresentationNetwork.
        g: DynamicsNetwork.
        f: PredictionNetwork.
        optimizer: Shared Adam optimizer over all three networks.
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
        self.optimizer = torch.optim.Adam(
            all_params,
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
        )

        self.model_version: int = 0

    def train_batch(self, batch: dict) -> dict:
        """Run one K-step unroll training step.

        Args:
            batch: Dictionary of numpy arrays:
                "observations":    [B, 103, 8, 8]
                "actions":         [B, K, 3, 8, 8]
                "target_policies": [B, K+1, 4672]
                "target_values":   [B, K+1]
                "target_rewards":  [B, K+1]
                "legal_masks":     [B, 4672] bool (optional) — if present, illegal
                                   actions are masked to -inf in the policy loss so
                                   gradients do not push logits at illegal positions.

        Returns:
            dict with keys: total_loss, policy_loss, value_loss, reward_loss, model_version
            (all loss values are Python floats).
        """
        # Convert numpy arrays to tensors on the target device.
        obs = torch.from_numpy(batch["observations"]).to(self.device)          # [B, 103, 8, 8]
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

        # Step 0: encode observation, predict (policy, value).
        hidden = self.h(obs)  # [B, 64, 8, 8]
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
            hidden, reward = self.g(hidden, action_plane)  # [B, 64, 8, 8], [B, 1]

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

        total_loss = avg_policy_loss + avg_value_loss + avg_reward_loss

        total_loss.backward()
        self.optimizer.step()

        self.model_version += 1

        return {
            "total_loss": total_loss.item(),
            "policy_loss": avg_policy_loss.item(),
            "value_loss": avg_value_loss.item(),
            "reward_loss": avg_reward_loss.item(),
            "model_version": self.model_version,
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
        return -torch.sum(targets * F.log_softmax(logits, dim=-1), dim=-1).mean()

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
        torch.save(
            {
                "h": self.h.state_dict(),
                "g": self.g.state_dict(),
                "f": self.f.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "model_version": self.model_version,
                "eval_metrics": eval_metrics,
            },
            path,
        )

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
        return checkpoint.get("eval_metrics")
