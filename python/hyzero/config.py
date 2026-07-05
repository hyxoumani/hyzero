"""Default hyperparameters for the hyzero MuZero model."""

import os
import sys

DEFAULT_CONFIG = {
    "hidden_channels": 128,   # C — channels in all hidden layers
    "num_res_blocks": 4,      # residual blocks per network
    "input_planes": 102,      # 6 game-state planes + 96 piece planes (8 positions × 12 piece planes); side-to-move plane removed (see Phase 3b)
    "num_actions": 4672,      # 4096 base (from×to) + 576 underpromotion slots
    "action_planes": 3,       # spatial action encoding planes
    "lr": 1e-3,
    "weight_decay": 1e-4,
}

# Distributional (categorical / HL-Gauss) value head support size: N atoms
# uniformly spaced over [-1, 1]. Only used when the value head mode is
# "categorical"; the legacy scalar+tanh head ignores it.
VALUE_SUPPORT_SIZE = 51


def value_head_mode() -> str:
    """Return the configured value-head mode from ``HYZERO_VALUE_HEAD``.

    "scalar" (default) selects the legacy scalar+tanh regression head — byte
    identical to pre-distributional behavior. "categorical" selects the
    HL-Gauss distributional head (``VALUE_SUPPORT_SIZE`` logits). An unknown
    value warns to stderr and falls back to "scalar".
    """
    mode = os.environ.get("HYZERO_VALUE_HEAD", "scalar").strip().lower()
    if mode not in ("scalar", "categorical"):
        print(
            f"[config] WARNING: HYZERO_VALUE_HEAD={mode!r} is not valid"
            " (expected 'scalar' or 'categorical'); using 'scalar'",
            file=sys.stderr,
        )
        mode = "scalar"
    return mode
