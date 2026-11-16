# data/nuscenes_dataset.py

import torch
import numpy as np
from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes


class NuScenesDOCOFDataset(Dataset):

    def __init__(self,
                 root,
                 split,
                 past_frames=3,
                 future_frames=4,
                 interval=0.5,
                 bev_range=50,
                 resolution=0.5,
                 ego_aligned=True):

        self.nusc = NuScenes(
            version='v1.0-trainval',
            dataroot=root,
            verbose=False
        )

        self.split = split
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.interval = interval
        self.bev_range = bev_range
        self.resolution = resolution
        self.grid_size = int((bev_range * 2) / resolution)
        self.ego_aligned = ego_aligned

        splits = create_splits_scenes()
        scene_names = splits['train'] if split == 'train' else splits['val']

        self.samples = []

        for scene in self.nusc.scene:
            if scene['name'] not in scene_names:
                continue

            token = scene['first_sample_token']

            while token != "":
                self.samples.append(token)
                sample = self.nusc.get('sample', token)
                token = sample['next']

        print(f"[INFO] Loaded {len(self.samples)} samples from {split} split")

    def __len__(self):
        return len(self.samples)

    # ----------------------------------------------------
    # Coordinate Conversion
    # ----------------------------------------------------
    def world_to_grid(self, x, y):

        gx = int((x + self.bev_range) / self.resolution)
        gy = int((y + self.bev_range) / self.resolution)

        if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
            return gx, gy
        return None

    # ----------------------------------------------------
    # Build BEV From Sample
    # ----------------------------------------------------
    def build_bev(self, sample):

        bev = torch.zeros(self.grid_size, self.grid_size)

        sd_token = sample["data"]["LIDAR_TOP"]
        sd = self.nusc.get("sample_data", sd_token)
        ego_pose = self.nusc.get("ego_pose", sd["ego_pose_token"])
        ego_translation = np.array(ego_pose["translation"][:2])

        for ann_token in sample["anns"]:
            ann = self.nusc.get("sample_annotation", ann_token)

            x_global, y_global = ann["translation"][:2]
            w, l = ann["size"][:2]

            x = x_global - ego_translation[0]
            y = y_global - ego_translation[1]

            coord = self.world_to_grid(x, y)
            if coord is None:
                continue

            gx, gy = coord

            half_w = int((w / self.resolution) / 2)
            half_l = int((l / self.resolution) / 2)

            x1 = max(0, gx - half_w)
            x2 = min(self.grid_size - 1, gx + half_w)
            y1 = max(0, gy - half_l)
            y2 = min(self.grid_size - 1, gy + half_l)

            bev[y1:y2, x1:x2] = 1.0

        return bev

    # ----------------------------------------------------
    # Get Item
    # ----------------------------------------------------
    def __getitem__(self, idx):

        token = self.samples[idx]
        sample = self.nusc.get('sample', token)

        # -------------------------
        # Past BEV stack
        # -------------------------
        past_bev = torch.zeros(
            self.past_frames,
            self.grid_size,
            self.grid_size
        )

        current = sample

        for t in range(self.past_frames):

            past_bev[self.past_frames - t - 1] = self.build_bev(current)

            if current["prev"] == "":
                break

            current = self.nusc.get("sample", current["prev"])

        # -------------------------
        # Future BEV stack (t+1..t+4)
        # -------------------------
        future_bev = torch.zeros(
            self.future_frames,
            self.grid_size,
            self.grid_size
        )

        future = sample

        for k in range(self.future_frames):

            if future["next"] == "":
                break

            future = self.nusc.get("sample", future["next"])

            future_bev[k] = self.build_bev(future)

        return {
            "past_bev": past_bev.unsqueeze(1).float(),
            "future_bev": future_bev.long()
        }