"""Prediction network f: hidden state -> (policy logits, value)."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from hyzero.config import VALUE_SUPPORT_SIZE


def build_value_support(support_size: int) -> torch.Tensor:
    """Return the categorical value support: ``support_size`` atoms over [-1, 1]."""
    return torch.linspace(-1.0, 1.0, support_size)


def hl_gauss_target(
    targets: torch.Tensor, support: torch.Tensor, sigma: float
) -> torch.Tensor:
    """Build HL-Gauss smoothed categorical targets from scalar value targets.

    Follows Farebrother et al. 2024 ("Stop Regressing"): each scalar target ``y``
    is turned into a discretized Gaussian over the value support by integrating a
    Normal(y, sigma) density across the bin owned by each support atom. Bin edges
    are the midpoints between adjacent atoms; the outer half-bins extend to the
    support boundary. The result is renormalized so each row sums to 1.

    Args:
        targets: [B] scalar targets (any real value; typically in [-1, 1]).
        support: [N] value support atoms (assumed uniformly spaced, ascending).
        sigma:   Smoothing standard deviation (bin_width * 0.75 by convention).

    Returns:
        [B, N] float tensor; each row is a probability distribution summing to 1.
    """
    targets = targets.reshape(-1, 1)  # [B, 1]
    n = support.shape[0]
    bin_width = (support[-1] - support[0]) / (n - 1)
    # N+1 bin edges: outer edges at the support boundary +/- half a bin, inner
    # edges at the midpoints between adjacent atoms.
    edges = torch.cat(
        [
            (support[:1] - bin_width / 2.0),
            (support[:-1] + support[1:]) / 2.0,
            (support[-1:] + bin_width / 2.0),
        ]
    ).to(targets.dtype)  # [N+1]
    cdf = 0.5 * (1.0 + torch.erf((edges - targets) / (sigma * math.sqrt(2.0))))  # [B, N+1]
    probs = cdf[:, 1:] - cdf[:, :-1]  # [B, N]
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return probs


class PredictionNetwork(nn.Module):
    """Predicts policy logits and state value from a hidden state.

    Input:  [B, hidden_channels, 8, 8]
    Outputs:
      policy_logits: [B, num_actions]  (raw logits, no softmax)
      value:         mode-dependent raw value-head output:
                       - "scalar":      [B, 1] bounded in [-1, 1] via Tanh
                       - "categorical": [B, value_support_size] raw logits over
                                        a fixed support spanning [-1, 1]

    Use :meth:`value_expectation` to map the raw value output to a scalar [B]
    value in [-1, 1] — this is the contract the Rust MCTS side consumes and is
    identical to ``value.squeeze(-1)`` in scalar mode.
    """

    def __init__(
        self,
        hidden_channels: int = 64,
        num_actions: int = 4672,
        value_head: str = "scalar",
        value_support_size: int = VALUE_SUPPORT_SIZE,
    ) -> None:
        super().__init__()
        board_size = 8 * 8  # 64 spatial positions

        if value_head not in ("scalar", "categorical"):
            raise ValueError(
                f"value_head must be 'scalar' or 'categorical', got {value_head!r}"
            )
        self.value_head_mode = value_head
        self.value_support_size = value_support_size

        # Policy head: 1x1 conv (64->2) -> BN -> ReLU -> flatten -> linear (128->4672)
        self.policy_head = nn.Sequential(
            nn.Conv2d(hidden_channels, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * board_size, num_actions),
        )

        if value_head == "scalar":
            # Value head: 1x1 conv (64->1) -> BN -> ReLU -> flatten -> linear (64->64)
            #             -> ReLU -> linear (64->1) -> tanh
            self.value_head = nn.Sequential(
                nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=False),
                nn.BatchNorm2d(1),
                nn.ReLU(inplace=True),
                nn.Flatten(),
                nn.Linear(board_size, hidden_channels),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channels, 1),
                nn.Tanh(),
            )
        else:
            # Distributional (HL-Gauss) head: same trunk, final linear outputs
            # value_support_size logits over the fixed support (no tanh).
            self.value_head = nn.Sequential(
                nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=False),
                nn.BatchNorm2d(1),
                nn.ReLU(inplace=True),
                nn.Flatten(),
                nn.Linear(board_size, hidden_channels),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channels, value_support_size),
            )
            # Fixed, non-learned support in [-1, 1]. Registered as a buffer so it
            # moves with .to(device) and is saved/restored with the state_dict.
            self.register_buffer(
                "value_support", build_value_support(value_support_size)
            )

    def forward(
        self, hidden_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        hidden_state: [B, hidden_channels, 8, 8]
        Returns: (policy_logits [B, num_actions], value_out)
          value_out is [B, 1] (scalar mode) or [B, value_support_size] (categorical).
        """
        policy_logits = self.policy_head(hidden_state)
        value = self.value_head(hidden_state)
        return policy_logits, value

    def value_expectation(self, value_out: torch.Tensor) -> torch.Tensor:
        """Map a raw value-head output to a scalar [B] value in [-1, 1].

        Scalar mode: returns ``value_out.squeeze(-1)`` (bit-identical to legacy).
        Categorical mode: returns the expectation of the value support under
        softmax(logits), i.e. the mean of the predicted value distribution.
        """
        if self.value_head_mode == "categorical":
            probs = F.softmax(value_out, dim=-1)  # [B, N]
            return (probs * self.value_support).sum(dim=-1)  # [B]
        return value_out.squeeze(-1)
