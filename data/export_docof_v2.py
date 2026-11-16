import os
import torch
import numpy as np
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from pyquaternion import Quaternion


# -------------------------------------------------------
# Ego-centric transform (translation + rotation)
# -------------------------------------------------------

def world_to_ego(nusc, sample, point_world):

    sd_token = sample["data"]["LIDAR_TOP"]
    sd = nusc.get("sample_data", sd_token)
    ego_pose = nusc.get("ego_pose", sd["ego_pose_token"])

    ego_translation = np.array(ego_pose["translation"][:2])
    ego_rotation = Quaternion(ego_pose["rotation"])

    # Translate
    p = np.array(point_world[:2]) - ego_translation

    # Rotate into ego frame
    p = ego_rotation.inverse.rotate([p[0], p[1], 0])[:2]

    return p


# -------------------------------------------------------
# Extract objects (ego-aligned)
# -------------------------------------------------------
def extract_objects(nusc, sample):

    objects = {}

    sd_token = sample["data"]["LIDAR_TOP"]
    sd = nusc.get("sample_data", sd_token)
    ego_pose = nusc.get("ego_pose", sd["ego_pose_token"])

    ego_translation = np.array(ego_pose["translation"][:2])
    ego_rotation = Quaternion(ego_pose["rotation"])
    ego_yaw = ego_rotation.yaw_pitch_roll[0]

    for ann_token in sample["anns"]:

        ann = nusc.get("sample_annotation", ann_token)

        inst = ann["instance_token"]

        # Global position
        obj_translation = ann["translation"]
        dx = obj_translation[0] - ego_translation[0]
        dy = obj_translation[1] - ego_translation[1]

        # Rotate into ego frame
        cos_y = np.cos(-ego_yaw)
        sin_y = np.sin(-ego_yaw)

        x_ego = cos_y * dx - sin_y * dy
        y_ego = sin_y * dx + cos_y * dy

        # Object yaw in ego frame
        obj_yaw_global = Quaternion(ann["rotation"]).yaw_pitch_roll[0]
        yaw_ego = obj_yaw_global - ego_yaw

        # 🔥 Normalize yaw to [-pi, pi]
        yaw_ego = (yaw_ego + np.pi) % (2 * np.pi) - np.pi

        w, l, _ = ann["size"]

        objects[inst] = np.array(
            [x_ego, y_ego, 0.0, 0.0, yaw_ego, w, l],
            dtype=np.float32
        )

    return objects


# -------------------------------------------------------
# Build occupancy grid
# -------------------------------------------------------

def build_occupancy(state, mask,
                    grid_size=200,
                    bev_range=50):

    occ = torch.zeros(grid_size, grid_size)

    coords = torch.linspace(-bev_range, bev_range, grid_size)
    grid_x, grid_y = torch.meshgrid(coords, coords, indexing="xy")

    N = state.shape[0]

    for i in range(N):

        if mask[i] == 0:
            continue

        x, y, _, _, yaw, w, l = state[i]

        dx = grid_x - x
        dy = grid_y - y

        cos_y = torch.cos(yaw)
        sin_y = torch.sin(yaw)

        x_rot = cos_y * dx + sin_y * dy
        y_rot = -sin_y * dx + cos_y * dy

        box = (torch.abs(x_rot) <= w / 2) & \
              (torch.abs(y_rot) <= l / 2)

        occ[box] = 1.0

    return occ


# -------------------------------------------------------
# Export Split
# -------------------------------------------------------

def export_split(root,
                 save_root,
                 split="train",
                 past_frames=3,
                 future_frames=4,
                 max_objects=20,
                 dt=0.5,
                 grid_size=200,
                 bev_range=50):

    nusc = NuScenes(
        version="v1.0-trainval",
        dataroot=root,
        verbose=False
    )

    splits = create_splits_scenes()
    scene_names = splits["train"] if split == "train" else splits["val"]

    os.makedirs(os.path.join(save_root, split), exist_ok=True)

    index = 0

    for scene in tqdm(nusc.scene, desc=f"Export {split}"):

        if scene["name"] not in scene_names:
            continue

        token = scene["first_sample_token"]

        while token != "":

            sample = nusc.get("sample", token)

            # -------------------------
            # Past
            # -------------------------

            history_dicts = []
            instance_set = set()
            current = sample

            for _ in range(past_frames):

                obj_dict = extract_objects(nusc, current)
                history_dicts.append(obj_dict)
                instance_set.update(obj_dict.keys())

                if current["prev"] == "":
                    break

                current = nusc.get("sample", current["prev"])

            # 🔥 Reverse once AFTER loop
            history_dicts = history_dicts[::-1]

            instance_list = list(instance_set)[:max_objects]
            N = len(instance_list)

            if N == 0:
                token = sample["next"]
                continue

            history = np.zeros((past_frames, N, 7),
                               dtype=np.float32)

            for t, obj_dict in enumerate(history_dicts):
                for i, inst in enumerate(instance_list):
                    if inst in obj_dict:
                        history[t, i] = obj_dict[inst]

            # velocity
            for t in range(1, past_frames):
                pos_t = history[t, :, :2]
                pos_prev = history[t-1, :, :2]
                history[t, :, 2:4] = (pos_t - pos_prev) / dt

            state_t = history[-1]

            # -------------------------
            # Future
            # -------------------------

            future_states = []
            future_occupancy = []
            validity_masks = []

            future = sample

            for _ in range(future_frames):

                if future["next"] == "":
                    break

                future = nusc.get("sample", future["next"])
                obj_dict = extract_objects(nusc, future)

                state_k = np.zeros((N, 7), dtype=np.float32)
                mask_k = np.zeros((N,), dtype=np.float32)

                for i, inst in enumerate(instance_list):
                    if inst in obj_dict:
                        state_k[i] = obj_dict[inst]
                        mask_k[i] = 1.0

                state_tensor = torch.tensor(state_k)
                mask_tensor = torch.tensor(mask_k, dtype=torch.float32)

                occ_k = build_occupancy(
                    state_tensor,
                    mask_tensor,
                    grid_size,
                    bev_range
                )

                future_states.append(state_tensor)
                future_occupancy.append(occ_k)
                validity_masks.append(mask_tensor)

            # Pad
            if len(future_states) > 0:
                zero_state = torch.zeros_like(future_states[0])
                zero_mask = torch.zeros_like(validity_masks[0])
                zero_occ = torch.zeros_like(future_occupancy[0])
            else:
                zero_state = torch.zeros((N, 7))
                zero_mask = torch.zeros((N,))
                zero_occ = torch.zeros((grid_size, grid_size))

            while len(future_states) < future_frames:
                future_states.append(zero_state.clone())
                future_occupancy.append(zero_occ.clone())
                validity_masks.append(zero_mask.clone())

            save_path = os.path.join(
                save_root,
                split,
                f"{index}.pt"
            )

            torch.save({
                "history": torch.tensor(history),
                "state_t": torch.tensor(state_t),
                "future_states": future_states,
                "future_occupancy": future_occupancy,
                "validity_masks": validity_masks
            }, save_path)

            index += 1
            token = sample["next"]

    print(f"{split} export complete. Total samples: {index}")


# -------------------------------------------------------
# Main
# -------------------------------------------------------

if __name__ == "__main__":

    ROOT = r"D:\Pritam\TS BEV\NuScenes"
    SAVE_ROOT = r"D:\Pritam\TS BEV\preprocessed_docof_v2"

    export_split(ROOT, SAVE_ROOT, "train")
    export_split(ROOT, SAVE_ROOT, "val")