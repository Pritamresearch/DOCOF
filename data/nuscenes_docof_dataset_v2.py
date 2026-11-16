import torch
import numpy as np
from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from pyquaternion import Quaternion


class NuScenesDOCOFDatasetV2(Dataset):

    def __init__(self,
                 root,
                 split="train",
                 past_frames=3,
                 future_frames=4,
                 max_objects=20,
                 dt=0.5):

        # Store root for safe reinitialization
        self.root = root
        self.split = split
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.max_objects = max_objects
        self.dt = dt

        # Load nuScenes once
        self.nusc = NuScenes(
            version="v1.0-trainval",
            dataroot=root,
            verbose=False
        )

        splits = create_splits_scenes()
        scene_names = splits["train"] if split == "train" else splits["val"]

        # Preload all sample tokens + sample objects
        self.samples = []
        self.sample_data = []

        for scene in self.nusc.scene:
            if scene["name"] not in scene_names:
                continue

            token = scene["first_sample_token"]

            while token != "":
                sample = self.nusc.get("sample", token)
                self.samples.append(token)
                self.sample_data.append(sample)
                token = sample["next"]

    def __len__(self):
        return len(self.samples)

    # --------------------------------------------------
    # Windows-safe pickle handling
    # --------------------------------------------------

    def __getstate__(self):
        state = self.__dict__.copy()
        state["nusc"] = None  # remove heavy object before pickling
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.nusc is None:
            self.nusc = NuScenes(
                version="v1.0-trainval",
                dataroot=self.root,
                verbose=False
            )

    # --------------------------------------------------
    # Extract objects for a single frame
    # --------------------------------------------------

    def extract_objects(self, sample):

        objects = {}

        for ann_token in sample["anns"]:
            ann = self.nusc.get("sample_annotation", ann_token)

            inst = ann["instance_token"]

            x, y = ann["translation"][:2]
            w, l, _ = ann["size"]
            yaw = Quaternion(ann["rotation"]).yaw_pitch_roll[0]

            objects[inst] = np.array(
                [x, y, 0.0, 0.0, yaw, w, l],
                dtype=np.float32
            )

        return objects

    # --------------------------------------------------
    # Main getter
    # --------------------------------------------------

    def __getitem__(self, idx):

        sample = self.sample_data[idx]

        # -----------------------------
        # Collect past frames
        # -----------------------------

        history_dicts = []
        instance_set = set()

        current = sample

        for _ in range(self.past_frames):

            obj_dict = self.extract_objects(current)
            history_dicts.append(obj_dict)
            instance_set.update(obj_dict.keys())

            if current["prev"] == "":
                break

            current = self.nusc.get("sample", current["prev"])

        instance_list = list(instance_set)[:self.max_objects]
        N = len(instance_list)

        history = np.zeros(
            (self.past_frames, N, 7),
            dtype=np.float32
        )

        for t, obj_dict in enumerate(history_dicts):
            for i, inst in enumerate(instance_list):
                if inst in obj_dict:
                    history[t, i] = obj_dict[inst]

        # -----------------------------
        # Compute velocity
        # -----------------------------

        for t in range(1, self.past_frames):
            pos_t = history[t, :, :2]
            pos_prev = history[t-1, :, :2]
            velocity = (pos_t - pos_prev) / self.dt
            history[t, :, 2:4] = velocity

        state_t = history[-1]

        # -----------------------------
        # Collect future frames
        # -----------------------------

        future_states = []
        validity_masks = []

        future = sample

        for _ in range(self.future_frames):

            if future["next"] == "":
                break

            future = self.nusc.get("sample", future["next"])
            obj_dict = self.extract_objects(future)

            state_k = np.zeros((N, 7), dtype=np.float32)
            mask_k = np.zeros((N,), dtype=np.float32)

            for i, inst in enumerate(instance_list):
                if inst in obj_dict:
                    state_k[i] = obj_dict[inst]
                    mask_k[i] = 1.0

            future_states.append(
                torch.tensor(state_k, dtype=torch.float32)
            )

            validity_masks.append(
                torch.tensor(mask_k, dtype=torch.float32)
            )

        return {
            "history": torch.tensor(history, dtype=torch.float32),
            "state_t": torch.tensor(state_t, dtype=torch.float32),
            "future_states": future_states,
            "validity_masks": validity_masks
        }