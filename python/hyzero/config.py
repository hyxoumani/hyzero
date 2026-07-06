"""Default hyperparameters for the hyzero MuZero model."""

import os
import sys

DEFAULT_CONFIG = {
    "hidden_channels": 128,   # C — channels in all hidden layers
    "num_res_blocks": 4,      # residual blocks per network
    "input_planes": 110,      # 6 game-state + 96 piece (8 positions × 12) + 8 lc0-style repetition planes; side-to-move plane removed (see Phase 3b)
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


def moves_left_head_enabled() -> bool:
    """Return whether the lc0-style moves-left head (MLH) is enabled.

    Read from ``HYZERO_MOVES_LEFT_HEAD``; default OFF (legacy). Any value that is
    not "0" / "false" / "no" / empty (case-insensitive, trimmed) enables it. When
    enabled, the prediction network grows a small scalar head predicting the
    normalized plies-remaining to the trajectory terminal in [0, 1].
    """
    raw = os.environ.get("HYZERO_MOVES_LEFT_HEAD", "").strip().lower()
    return raw not in ("", "0", "false", "no")


def moves_left_cap() -> float:
    """Return the moves-left normalization cap (plies) from ``HYZERO_MLH_CAP``.

    Plies-remaining targets are normalized as ``plies / cap`` and clamped to
    [0, 1]. Default 100.0. Non-positive or unparseable values fall back to 100.0.
    """
    raw = os.environ.get("HYZERO_MLH_CAP")
    if raw is None:
        return 100.0
    try:
        value = float(raw)
    except (ValueError, TypeError):
        print(
            f"[config] WARNING: HYZERO_MLH_CAP={raw!r} is not a valid float;"
            " using default 100.0",
            file=sys.stderr,
        )
        return 100.0
    return value if value > 0.0 else 100.0
