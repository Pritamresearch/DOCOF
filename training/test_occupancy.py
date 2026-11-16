import os
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes

from data.occupancy_generator import OccupancyGenerator

# =====================
# CONFIG
# =====================
DATAROOT = "dataset/nuScenes mini"   # change if needed
VERSION = "v1.0-mini"

# =====================
# LOAD NUSCENES
# =====================
nusc = NuScenes(
    version=VERSION,
    dataroot=DATAROOT,
    verbose=True
)

# =====================
# GENERATOR
# =====================
occ_gen = OccupancyGenerator(
    x_range=(-50, 50),
    y_range=(-50, 50),
    grid_size=128
)

# =====================
# TEST ONE SAMPLE
# =====================
sample = nusc.sample[0]
sample_token = sample["token"]

grid = occ_gen.generate_future_occupancy(
    nusc,
    sample_token,
    horizon=1
)

print("Grid shape:", grid.shape)
print("Occupied cells:", grid.sum())

# =====================
# VISUALIZE
# =====================
plt.imshow(grid, cmap="gray")
plt.title("Future Occupancy (t+1)")
plt.show()
