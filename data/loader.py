import torch
from torch.utils.data import Dataset
import numpy as np
from utils.compute_intent import compute_intent


class NuScenesIntentDataset(Dataset):

    def __init__(self, nusc, sample_tokens, past_frames=5, future_frames=3):

        self.nusc = nusc
        self.sample_tokens = sample_tokens
        self.past_frames = past_frames
        self.future_frames = future_frames

        self.grid_size = 200
        self.resolution = 0.5
        self.offset = 50

    def __len__(self):
        return len(self.sample_tokens)

    def world_to_grid(self, x, y):
        gx = int((x + self.offset) / self.resolution)
        gy = int((y + self.offset) / self.resolution)

        if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
            return gx, gy
        return None

    def load_future_bev(self, sample):

        K = self.future_frames
        future_seq = torch.zeros(K, 1, self.grid_size, self.grid_size)

        # 🔥 Ego pose of CURRENT frame (fixed reference)
        sd_token = sample["data"]["LIDAR_TOP"]
        sd = self.nusc.get("sample_data", sd_token)
        ego_pose = self.nusc.get("ego_pose", sd["ego_pose_token"])
        ego_translation_ref = np.array(ego_pose["translation"][:2])

        current_sample = sample

        for k in range(K):

            if current_sample["next"] == "":
                break

            current_sample = self.nusc.get("sample", current_sample["next"])

            bev = torch.zeros(1, self.grid_size, self.grid_size)

            for ann_token in current_sample["anns"]:
                ann = self.nusc.get("sample_annotation", ann_token)

                x_global, y_global = ann["translation"][:2]
                w, l = ann["size"][:2]

                # 🔥 Convert future object to CURRENT ego frame
                x = x_global - ego_translation_ref[0]
                y = y_global - ego_translation_ref[1]

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

                bev[0, y1:y2, x1:x2] = 1.0

            future_seq[k] = bev

        return future_seq

    def load_past_bev(self, sample):

        T = self.past_frames
        bev_seq = torch.zeros(T, 1, self.grid_size, self.grid_size)

        current_sample = sample

        for t in range(T):

            bev = torch.zeros(1, self.grid_size, self.grid_size)

            sd_token = current_sample["data"]["LIDAR_TOP"]
            sd = self.nusc.get("sample_data", sd_token)
            ego_pose = self.nusc.get("ego_pose", sd["ego_pose_token"])
            ego_translation = np.array(ego_pose["translation"][:2])

            for ann_token in current_sample["anns"]:

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

                bev[0, y1:y2, x1:x2] = 1.0

            bev_seq[T - t - 1] = bev

            if current_sample["prev"] == "":
                break

            current_sample = self.nusc.get("sample", current_sample["prev"])

        return bev_seq

    def __getitem__(self, idx):

        sample_token = self.sample_tokens[idx]
        sample = self.nusc.get("sample", sample_token)

        bev_past = self.load_past_bev(sample)
        bev_future = self.load_future_bev(sample)

        intents = []
        boxes = []
        motion_targets = []

        # Ego pose of CURRENT frame
        sd_token = sample["data"]["LIDAR_TOP"]
        sd = self.nusc.get("sample_data", sd_token)
        ego_pose = self.nusc.get("ego_pose", sd["ego_pose_token"])
        ego_translation = np.array(ego_pose["translation"][:2])

        for ann_token in sample["anns"]:

            ann = self.nusc.get("sample_annotation", ann_token)

            # -------------------------
            # 1. Convert CURRENT center to ego frame
            # -------------------------
            x_global, y_global = ann["translation"][:2]
            x0 = x_global - ego_translation[0]
            y0 = y_global - ego_translation[1]

            center0 = self.world_to_grid(x0, y0)
            if center0 is None:
                continue

            gx0, gy0 = center0

            # -------------------------
            # 2. Build future motion (always fixed length)
            # -------------------------
            # -------------------------
            # 2. Build future motion (always fixed length)
            # -------------------------
            obj_motion = []
            current_ann = ann

            # -------------------------
            # 2. Build future motion in METERS (ego frame)
            # -------------------------
            obj_motion = []
            current_ann = ann

            # current center in ego frame (meters)
            x0_global, y0_global = ann["translation"][:2]
            x0 = x0_global - ego_translation[0]
            y0 = y0_global - ego_translation[1]

            for k in range(self.future_frames):

                if current_ann["next"] == "":
                    obj_motion.append([0.0, 0.0])
                    continue

                current_ann = self.nusc.get("sample_annotation", current_ann["next"])

                x_future_global, y_future_global = current_ann["translation"][:2]

                # convert future center to ego frame
                x_future = x_future_global - ego_translation[0]
                y_future = y_future_global - ego_translation[1]

                # displacement in meters
                dx_meter = x_future - x0
                dy_meter = y_future - y0

                obj_motion.append([dx_meter, dy_meter])

            # Ensure fixed length
            if len(obj_motion) < self.future_frames:
                obj_motion += [[0.0, 0.0]] * (self.future_frames - len(obj_motion))

            motion_targets.append(obj_motion)

            # -------------------------
            # 3. Box for ROI
            # -------------------------
            w, l = ann["size"][:2]

            half_w = int((w / self.resolution) / 2)
            half_l = int((l / self.resolution) / 2)

            x1 = gx0 - half_w
            x2 = gx0 + half_w
            y1 = gy0 - half_l
            y2 = gy0 + half_l

            boxes.append([x1, y1, x2, y2])

            # -------------------------
            # 4. Intent
            # -------------------------
            v_t = np.array(self.nusc.box_velocity(ann_token)[:2])
            v_f = np.array(self.nusc.box_velocity(current_ann["token"])[:2])

            p_t = np.array(ann["translation"][:2])
            p_f = np.array(current_ann["translation"][:2])

            intent_label = compute_intent(p_t, p_f, 0.0, v_t, v_f)
            intents.append(intent_label)

        # -------------------------
        # If no objects
        # -------------------------
        if len(intents) == 0:
            return {
                "bev_past": bev_past,
                "bev_future": bev_future,
                "intent_label": torch.zeros(0, dtype=torch.long),
                "boxes": torch.zeros(0, 4),
                "motion_target": torch.zeros(0, self.future_frames, 2)
            }

        return {
            "bev_past": bev_past,
            "bev_future": bev_future,
            "intent_label": torch.tensor(intents, dtype=torch.long),
            "boxes": torch.tensor(boxes, dtype=torch.float),
            "motion_target": torch.tensor(motion_targets, dtype=torch.float)
        }