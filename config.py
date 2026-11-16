"""
Default configuration dictionary for DOCOF (Decomposed Object-Centric Occupancy Forecasting).
"""

CONFIG = {
    # Dataset Parameters
    "dataset_root": "data/nuScenes",
    "preprocessed_dir": "preprocessed_docof_v2",
    "split": "train",
    "past_frames": 3,
    "future_frames": 4,
    "max_objects": 20,

    # Kinematics & State-Space Parameters
    "state_dim": 7,          # [x, y, vx, vy, yaw, width, length]
    "hidden_dim": 128,       # GRU hidden dimension
    "dt": 0.5,               # Time step between horizons (seconds)
    "max_speed": 40.0,       # v_max constraint (m/s) -> 144 km/h
    "max_acc": 10.0,         # a_max acceleration bound (m/s^2)
    "max_omega": 3.14159,    # omega_max yaw rate bound (rad/s)

    # Bird's-Eye View (BEV) Grid Parameters
    "grid_size": 200,        # 200 x 200 grid
    "bev_range": 50.0,       # +/- 50 meters (0.5m / cell resolution)
    "renderer_kappa": 10.0,  # Edge sharpness parameter for soft-box rendering

    # Training & Optimization
    "batch_size": 1,
    "epochs": 30,
    "lr": 1e-4,
    "grad_clip": 5.0,
    "weights_dir": "checkpoints"
}

# Alias for backward compatibility with v2 scripts
CONFIG_V2 = CONFIG
