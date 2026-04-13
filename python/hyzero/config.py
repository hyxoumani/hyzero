"""Default hyperparameters for the hyzero MuZero model."""

DEFAULT_CONFIG = {
    "hidden_channels": 64,    # C — channels in all hidden layers
    "num_res_blocks": 4,      # residual blocks per network
    "input_planes": 19,       # observation planes (board encoding)
    "num_actions": 4096,      # 64 × 64 action space
    "action_planes": 3,       # spatial action encoding planes
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "value_loss_weight": 10.0,
    "reward_loss_weight": 10.0,
}
