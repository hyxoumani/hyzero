"""Default hyperparameters for the hyzero MuZero model."""

DEFAULT_CONFIG = {
    "hidden_channels": 128,   # C — channels in all hidden layers
    "num_res_blocks": 4,      # residual blocks per network
    "input_planes": 102,      # 6 game-state planes + 96 piece planes (8 positions × 12 piece planes); side-to-move plane removed (see Phase 3b)
    "num_actions": 4672,      # 4096 base (from×to) + 576 underpromotion slots
    "action_planes": 3,       # spatial action encoding planes
    "lr": 1e-3,
    "weight_decay": 1e-4,
}
