"""Inference server: batch inference with MuZero networks under torch.no_grad()."""

import io

import numpy as np
import torch
import torch.nn.functional as F

from hyzero.config import DEFAULT_CONFIG, moves_left_head_enabled, value_head_mode
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

        # Moves-left head (MLH) must match the trainer's config (same process, same
        # env) so a synced state_dict loads cleanly and, when enabled, root/leaf
        # inference can return the extra normalized plies-remaining estimate.
        self.moves_left_head_enabled = moves_left_head_enabled()
        self.f = PredictionNetwork(
            hidden_channels=cfg["hidden_channels"],
            num_actions=cfg["num_actions"],
            value_head=value_head_mode(),
            moves_left_head=self.moves_left_head_enabled,
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

        out = (
            hidden.cpu().numpy().astype(np.float32),
            policies.cpu().numpy().astype(np.float32),
            values.cpu().numpy().astype(np.float32),  # [B]
        )
        # Backward-compatible extension: only when the moves-left head is enabled,
        # append the normalized plies-remaining estimate m ∈ [0, 1] as a trailing
        # tuple element. The Rust side reads it conditionally on the same env flag;
        # legacy consumers read indices 0..2 and ignore any extra element.
        if self.moves_left_head_enabled:
            moves_left = self.f.moves_left(hidden)  # [B]
            out = out + (moves_left.cpu().numpy().astype(np.float32),)
        return out

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

        out = (
            new_hidden.cpu().numpy().astype(np.float32),
            reward.squeeze(-1).cpu().numpy().astype(np.float32),    # [B]
            policies.cpu().numpy().astype(np.float32),
            values.cpu().numpy().astype(np.float32),     # [B]
        )
        # Backward-compatible extension (see root_setup_batch): append m ∈ [0, 1]
        # only when the moves-left head is enabled.
        if self.moves_left_head_enabled:
            moves_left = self.f.moves_left(new_hidden)  # [B]
            out = out + (moves_left.cpu().numpy().astype(np.float32),)
        return out

    def load_weights(self, state_dict_bytes: bytes) -> None:
        """Load network weights serialized by Trainer.get_weights().

        Args:
            state_dict_bytes: Bytes produced by Trainer.get_weights().
        """
        # weights_only=True: payload is a dict of state_dicts / plain values;
        # verified to load cleanly, and rejects arbitrary-code deserialization.
        buf = io.BytesIO(state_dict_bytes)
        checkpoint = torch.load(buf, map_location=self.device, weights_only=True)

        self.h.load_state_dict(checkpoint["h"])
        self.g.load_state_dict(checkpoint["g"])
        # The moves-left head is additive: tolerate loading legacy (head-less)
        # weights into a server configured with the MLH by leaving the freshly
        # initialized head as-is. The reverse (incoming weights carry a trained
        # MLH the server lacks) surfaces as an unexpected-key error via strict
        # loading, matching Trainer.load_checkpoint.
        incoming_f = checkpoint["f"]
        incoming_has_mlh = any(k.startswith("moves_left_head") for k in incoming_f)
        if self.moves_left_head_enabled and not incoming_has_mlh:
            missing, unexpected = self.f.load_state_dict(incoming_f, strict=False)
            if unexpected or any(
                not k.startswith("moves_left_head") for k in missing
            ):
                raise ValueError(
                    "moves-left-head mismatch: f weights are incompatible beyond"
                    f" the additive MLH head (missing={missing},"
                    f" unexpected={unexpected})."
                )
            print(
                "[inference] MLH head fresh-initialized on legacy ckpt", flush=True
            )
        else:
            self.f.load_state_dict(incoming_f)

        # Ensure networks stay in eval mode after loading.
        self.h.eval()
        self.g.eval()
        self.f.eval()
