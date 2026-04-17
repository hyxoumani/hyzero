"""Representation network h: observation -> hidden state."""

import torch
import torch.nn as nn

from hyzero.models.common import ResidualBlock


class RepresentationNetwork(nn.Module):
    """Encodes a raw board observation into a latent hidden state.

    Input:  [B, input_planes, 8, 8]  (default input_planes=102)
    Output: [B, hidden_channels, 8, 8]  (default hidden_channels=128)

    Also exposes projector and predictor heads for EfficientZero-style
    self-supervised consistency loss (SimSiam, Ye et al. NeurIPS 2021).
    These are only used during training, not inference.
    """

    def __init__(
        self,
        input_planes: int = 102,
        hidden_channels: int = 128,
        num_res_blocks: int = 4,
        proj_dim: int = 256,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_planes, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_channels) for _ in range(num_res_blocks)]
        )

        # SimSiam-style projector: shared between representation & dynamics branches.
        # Projects [B, C*64] -> [B, proj_dim] with BN to prevent collapse.
        board_size = 8 * 8  # 64
        self.projector = nn.Sequential(
            nn.Linear(hidden_channels * board_size, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
        )
        # Predictor: applied only on the dynamics-output branch (with stop-grad on target).
        # 128-dim bottleneck following SimSiam design.
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, proj_dim // 2),
            nn.BatchNorm1d(proj_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim // 2, proj_dim),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """observation: [B, input_planes, 8, 8] -> [B, hidden_channels, 8, 8]"""
        x = self.stem(observation)
        x = self.res_blocks(x)
        return x

    def project(self, hidden: torch.Tensor) -> torch.Tensor:
        """Apply projector to a hidden state.

        Args:
            hidden: [B, C, 8, 8]

        Returns:
            [B, proj_dim]
        """
        return self.projector(hidden.flatten(1))

    def predict(self, projected: torch.Tensor) -> torch.Tensor:
        """Apply predictor to a projection (used on dynamics-output branch only).

        Args:
            projected: [B, proj_dim]

        Returns:
            [B, proj_dim]
        """
        return self.predictor(projected)
