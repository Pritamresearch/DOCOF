import numpy as np
from collections import Counter
from torch.utils.data import DataLoader

from nuscenes.nuscenes import NuScenes

from data.loader import NuScenesIntentDataset

# --------------------------------------------------
# 1. Initialize nuScenes
# --------------------------------------------------

DATAROOT = "D:/Pritam/TS BEV/NuScenes"   # <-- change to your path

nusc = NuScenes(
    version="v1.0-trainval",
    dataroot=DATAROOT,
    verbose=True
)

# --------------------------------------------------
# 2. Collect sample tokens (limit for testing)
# --------------------------------------------------

sample_tokens = [s["token"] for s in nusc.sample][:500]  # first 500 samples

# --------------------------------------------------
# 3. Create Dataset
# --------------------------------------------------

dataset = NuScenesIntentDataset(
    nusc=nusc,
    sample_tokens=sample_tokens,
    past_frames=5,
    future_horizon=1.0
)

# --------------------------------------------------
# 4. Create DataLoader
# --------------------------------------------------

dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)

# --------------------------------------------------
# 5. Compute Distribution
# --------------------------------------------------

all_intents = []

for batch in dataloader:

    intents = batch["intent_label"]  # shape: [1, N_objects]

    intents = intents.squeeze(0).numpy()

    all_intents.extend(intents.tolist())

counter = Counter(all_intents)

print("\nIntent Distribution:")
for k in sorted(counter.keys()):
    print(f"Class {k}: {counter[k]}")