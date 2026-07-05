"""Inference server: batch inference with MuZero networks under torch.no_grad()."""

import io

import numpy as np
import torch
import torch.nn.functional as F

from hyzero.config import DEFAULT_CONFIG, value_head_mode
from hyzero.models.representation import RepresentationNetwork
from hyzero.models.dynamics import DynamicsNetwork
from hyzero.models.prediction import PredictionNetwork


class InferenceServer:
    """Wraps all three MuZero networks for batch inference.

    All methods run under torch.no_grad() and return numpy float32 arrays.
    Networks are always in eval mode.

    Attributes:
        config: Hyperparameter dictionary.
        device: Torch device string ("cpu" or "cuda").
        h: RepresentationNetwork (eval mode).
        g: DynamicsNetwork (eval mode).
        f: PredictionNetwork (eval mode).
    """

    def __init__(self, config: dict = None, device: str = "cpu") -> None:
        self.config = config if config is not None else dict(DEFAULT_CONFIG)
        self.device = device

        cfg = self.config
        self.h = RepresentationNetwork(
            input_planes=cfg["input_planes"],
            hidden_channels=cfg["hidden_channels"],
            num_res_blocks=cfg["num_res_blocks"],
        ).to(device).eval()

        self.g = DynamicsNetwork(
            hidden_channels=cfg["hidden_channels"],
            action_planes=cfg["action_planes"],
            num_res_blocks=cfg["num_res_blocks"],
        ).to(device).eval()

        self.f = PredictionNetwork(
            hidden_channels=cfg["hidden_channels"],
            num_actions=cfg["num_actions"],
            value_head=value_head_mode(),
        ).to(device).eval()

    @torch.no_grad()
    def root_setup_batch(
        self,
        observations: np.ndarray,
        legal_masks: np.ndarray | None = None,
    ) -> tuple:
        """Encode observations and predict policy + value for root nodes.

        Args:
            observations: [B, 103, 8, 8] float32 numpy array.
            legal_masks:  [B, 4672] bool numpy array or None.
                          If provided, illegal actions are masked to -inf before softmax
                          so the returned policy is non-zero only on legal moves.

        Returns:
            Tuple of numpy float32 arrays:
                hidden_states: [B, 64, 8, 8]
                policies:      [B, 4672]  (softmax-normalized, masked if legal_masks provided)
                values:        [B]
        """
        # observations: [B, 103, 8, 8] -> tensor on device
        obs_t = torch.from_numpy(observations).to(self.device)

        hidden = self.h(obs_t)                    # [B, 64, 8, 8]
        policy_logits, value = self.f(hidden)     # [B, 4672], [B, 1]

        if legal_masks is not None:
            mask_t = torch.from_numpy(legal_masks).to(self.device)  # [B, 4672] bool
            policy_logits = policy_logits.masked_fill(~mask_t, float('-inf'))

        policies = F.softmax(policy_logits, dim=-1)  # [B, 4672]

        # Contract: return a scalar value per position. In scalar mode this is
        # value.squeeze(-1); in categorical mode it is the support expectation.
        values = self.f.value_expectation(value)  # [B]

        return (
            hidden.cpu().numpy().astype(np.float32),
            policies.cpu().numpy().astype(np.float32),
            values.cpu().numpy().astype(np.float32),  # [B]
        )

    @torch.no_grad()
    def expand_leaf_batch(
        self,
        hidden_states: np.ndarray,
        actions: np.ndarray,
    ) -> tuple:
        """Expand leaf nodes: dynamics step then prediction.

        Note: no legal-move mask is applied here. At depth > 0 the engine operates
        in the learned latent space where there is no real board to derive legal moves from.

        Args:
            hidden_states: [B, 64, 8, 8] float32 numpy array.
            actions:       [B, 3, 8, 8]  float32 numpy action planes.

        Returns:
            Tuple of numpy float32 arrays:
                new_hidden: [B, 64, 8, 8]
                rewards:    [B]
                policies:   [B, 4672]  (softmax-normalized, unmasked)
                values:     [B]
        """
        # hidden_states: [B, 64, 8, 8], actions: [B, 3, 8, 8] -> tensors
        hidden_t = torch.from_numpy(hidden_states).to(self.device)
        action_t = torch.from_numpy(actions).to(self.device)

        new_hidden, reward = self.g(hidden_t, action_t)  # [B, 64, 8, 8], [B, 1]
        policy_logits, value = self.f(new_hidden)        # [B, 4096], [B, 1]

        policies = F.softmax(policy_logits, dim=-1)      # [B, 4096]

        # Scalar value contract (see root_setup_batch): expectation in categorical mode.
        values = self.f.value_expectation(value)  # [B]

        return (
            new_hidden.cpu().numpy().astype(np.float32),
            reward.squeeze(-1).cpu().numpy().astype(np.float32),    # [B]
            policies.cpu().numpy().astype(np.float32),
            values.cpu().numpy().astype(np.float32),     # [B]
        )

    def load_weights(self, state_dict_bytes: bytes) -> None:
        """Load network weights serialized by Trainer.get_weights().

        Args:
            state_dict_bytes: Bytes produced by Trainer.get_weights().
        """
        # weights_only=False: payload is a dict of state_dicts (no arbitrary code)
        buf = io.BytesIO(state_dict_bytes)
        checkpoint = torch.load(buf, map_location=self.device, weights_only=False)

        self.h.load_state_dict(checkpoint["h"])
        self.g.load_state_dict(checkpoint["g"])
        self.f.load_state_dict(checkpoint["f"])

        # Ensure networks stay in eval mode after loading.
        self.h.eval()
        self.g.eval()
        self.f.eval()
