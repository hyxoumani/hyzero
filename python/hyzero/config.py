"""Default hyperparameters for the hyzero MuZero model."""

DEFAULT_CONFIG = {
    "hidden_channels": 64,    # C — channels in all hidden layers
    "num_res_blocks": 6,      # residual blocks per network
    "input_planes": 103,      # observation planes: 8 history × 12 piece planes + 7 game-state planes
    "num_actions": 4672,      # 4096 base (from×to) + 576 underpromotion slots
    "action_planes": 3,       # spatial action encoding planes
    "lr": 1e-3,
    "weight_decay": 1e-4,
}
