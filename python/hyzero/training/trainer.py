"""MuZero training loop with K-step unroll, loss computation, and checkpointing."""

import io
import math

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

        # Cosine LR schedule with linear warmup
        warmup = cfg.get("lr_warmup_steps", 100)
        decay = cfg.get("lr_decay_steps", 10000)
        min_f = cfg.get("lr_min_factor", 0.1)

        def _lr_lambda(step: int) -> float:
            if step < warmup:
                return max(step, 1) / warmup  # linear warmup
            progress = min((step - warmup) / max(decay - warmup, 1), 1.0)
            return min_f + 0.5 * (1.0 - min_f) * (1.0 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, _lr_lambda)

        self.model_version: int = 0

    def train_batch(self, batch: dict) -> dict:
        """Run one K-step unroll training step.

        Args:
            batch: Dictionary of numpy arrays:
                "observations":    [B, 19, 8, 8]
                "actions":         [B, K, 3, 8, 8]
                "target_policies": [B, K+1, 4096]
                "target_values":   [B, K+1]
                "target_rewards":  [B, K+1]

        Returns:
            dict with keys: total_loss, policy_loss, value_loss, reward_loss, model_version
            (all loss values are Python floats).
        """
        # Convert numpy arrays to tensors on the target device.
        obs = torch.from_numpy(batch["observations"]).to(self.device)          # [B, 19, 8, 8]
        actions = torch.from_numpy(batch["actions"]).to(self.device)           # [B, K, 3, 8, 8]
        tgt_policies = torch.from_numpy(batch["target_policies"]).to(self.device)  # [B, K+1, 4096]
        tgt_values = torch.from_numpy(batch["target_values"]).to(self.device)  # [B, K+1]
        tgt_rewards = torch.from_numpy(batch["target_rewards"]).to(self.device)  # [B, K+1]

        k_steps = actions.shape[1]  # K

        self.optimizer.zero_grad()

        total_policy_loss = torch.tensor(0.0, device=self.device)
        total_value_loss = torch.tensor(0.0, device=self.device)
        total_reward_loss = torch.tensor(0.0, device=self.device)

        # Step 0: encode observation, predict (policy, value).
        hidden = self.h(obs)  # [B, 64, 8, 8]
        policy_logits, value = self.f(hidden)  # [B, 4096], [B, 1]

        # Policy loss at step 0.
        total_policy_loss = total_policy_loss + self._policy_loss(policy_logits, tgt_policies[:, 0])
        # Value loss at step 0.
        total_value_loss = total_value_loss + F.mse_loss(value.squeeze(-1), tgt_values[:, 0])

        # Steps 1..K: unroll dynamics.
        for k in range(1, k_steps + 1):
            action_plane = actions[:, k - 1]  # [B, 3, 8, 8]
            hidden, reward = self.g(hidden, action_plane)  # [B, 64, 8, 8], [B, 1]

            # MuZero: scale gradient at dynamics boundary (Appendix G) to stabilize K-step unroll
            hidden.register_hook(lambda grad: grad * 0.5)

            policy_logits, value = self.f(hidden)  # [B, 4096], [B, 1]

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
        self.scheduler.step()

        self.model_version += 1

        return {
            "total_loss": total_loss.item(),
            "policy_loss": avg_policy_loss.item(),
            "value_loss": avg_value_loss.item(),
            "reward_loss": avg_reward_loss.item(),
            "model_version": self.model_version,
        }

    def _policy_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Cross-entropy between logits and a target probability distribution.

        Args:
            logits:  [B, num_actions]
            targets: [B, num_actions]  (soft probability distribution summing to 1)

        Returns:
            Scalar tensor (mean over batch).
        """
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
                "scheduler": self.scheduler.state_dict(),
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
        if "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.model_version = checkpoint["model_version"]
        return checkpoint.get("eval_metrics")
