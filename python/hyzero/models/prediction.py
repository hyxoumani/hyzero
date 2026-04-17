"""Prediction network f: hidden state -> (policy logits, value)."""

import torch
import torch.nn as nn

class PredictionNetwork(nn.Module):
    """Predicts policy logits and state value from a hidden state.

    Input:  [B, hidden_channels, 8, 8]
    Outputs:
      policy_logits: [B, num_actions]  (raw logits, no softmax)
      value:         [B, 1]  bounded in [-1, 1] via Tanh
    """

    def __init__(
        self,
        hidden_channels: int = 64,
        num_actions: int = 4672,
    ) -> None:
        super().__init__()
        board_size = 8 * 8  # 64 spatial positions

        # Policy head: 1x1 conv (64->2) -> BN -> ReLU -> flatten -> linear (128->4672)
        self.policy_head = nn.Sequential(
            nn.Conv2d(hidden_channels, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * board_size, num_actions),
        )

        # Value head: 1x1 conv (64->1) -> BN -> ReLU -> flatten -> linear (64->64) -> ReLU -> linear (64->1) -> tanh
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

    def forward(
        self, hidden_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        hidden_state: [B, hidden_channels, 8, 8]
        Returns: (policy_logits [B, num_actions], value [B, 1])
        """
        policy_logits = self.policy_head(hidden_state)
        value = self.value_head(hidden_state)
        return policy_logits, value
