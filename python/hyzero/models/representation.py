"""Representation network h: observation -> hidden state."""

import torch
import torch.nn as nn

from hyzero.models.common import ResidualBlock


class RepresentationNetwork(nn.Module):
    """Encodes a raw board observation into a latent hidden state.

    Input:  [B, input_planes, 8, 8]  (default input_planes=19)
    Output: [B, hidden_channels, 8, 8]  (default hidden_channels=64)
    """

    def __init__(self, input_planes: int = 19, hidden_channels: int = 64, num_res_blocks: int = 4) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_planes, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_channels) for _ in range(num_res_blocks)]
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """observation: [B, input_planes, 8, 8] -> [B, hidden_channels, 8, 8]"""
        x = self.stem(observation)
        x = self.res_blocks(x)
        return x
