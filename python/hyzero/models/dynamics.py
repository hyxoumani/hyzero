"""Dynamics network g: (hidden state, action) -> (next hidden state, reward)."""

import torch
import torch.nn as nn

from hyzero.models.common import ResidualBlock


class DynamicsNetwork(nn.Module):
    """Predicts the next hidden state and immediate reward from a hidden state and action.

    Inputs:
      hidden_state:  [B, hidden_channels, 8, 8]
      action_planes: [B, action_planes, 8, 8]
    Outputs:
      next_hidden:   [B, hidden_channels, 8, 8]
      reward:        [B, 1]  bounded in [-1, 1] via Tanh
    """

    def __init__(
        self,
        hidden_channels: int = 64,
        action_planes: int = 3,
        num_res_blocks: int = 4,
    ) -> None:
        super().__init__()
        in_channels = hidden_channels + action_planes
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_channels) for _ in range(num_res_blocks)]
        )
        # Reward head: 1x1 conv -> flatten -> linear -> tanh
        board_size = 8 * 8  # 64 spatial positions
        self.reward_head = nn.Sequential(
            nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=False),
            nn.Flatten(),
            nn.Linear(board_size, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        hidden_state: torch.Tensor,
        action_planes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        hidden_state:  [B, hidden_channels, 8, 8]
        action_planes: [B, action_planes, 8, 8]
        Returns: (next_hidden [B, hidden_channels, 8, 8], reward [B, 1])
        """
        x = torch.cat([hidden_state, action_planes], dim=1)
        x = self.stem(x)
        next_hidden = self.res_blocks(x)
        reward = self.reward_head(next_hidden)
        return next_hidden, reward
