import cv2
import torch


class NuScenesLoader:
    """
    Returns:
        frames: list of T tensors [3,H,W]
        future_sample: nuScenes sample dict
    """

    def __init__(self, nusc, scene, seq_len=3, cam="CAM_FRONT"):
        self.nusc = nusc
        self.scene = scene
        self.seq_len = seq_len
        self.cam = cam

        self.sample_token = scene["first_sample_token"]

    def __iter__(self):

        current_token = self.sample_token

        while current_token != "":

            sample = self.nusc.get("sample", current_token)

            frames = []
            temp_token = current_token

            # Collect past frames
            for _ in range(self.seq_len):

                temp_sample = self.nusc.get("sample", temp_token)
                cam_token = temp_sample["data"][self.cam]
                cam_data = self.nusc.get("sample_data", cam_token)

                img_path = self.nusc.dataroot + "/" + cam_data["filename"]
                img = cv2.imread(img_path)

                if img is None:
                    break

                img = cv2.resize(img, (256, 256))
                img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

                frames.append(img)

                if temp_sample["prev"] == "":
                    break

                temp_token = temp_sample["prev"]

            if len(frames) == self.seq_len:
                frames = frames[::-1]  # chronological order
                yield frames, sample

            if sample["next"] == "":
                break

            current_token = sample["next"]
