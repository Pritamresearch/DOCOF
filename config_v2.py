CONFIG_V2 = {

    # Dataset
    "dataset_root": r"D:\Pritam\TS BEV\NuScenes",

    "past_frames": 3,
    "future_frames": 4,
    "max_objects": 20,

    # Model
    "state_dim": 7,
    "hidden_dim": 128,
    "dt": 0.5,
    "max_speed": 40.0,
    "max_acc": 10.0,
    "max_omega": 3.14,

    # BEV
    "grid_size": 200,
    "bev_range": 50,

    # Training
    "batch_size": 1,
    "epochs": 30,
    "lr": 1e-4,
    "grad_clip": 5.0
}