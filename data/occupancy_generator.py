import numpy as np


class OccupancyGenerator:

    def __init__(
        self,
        nusc,
        x_range=(-32, 32),
        y_range=(-32, 32),
        resolution=0.5
    ):
        self.nusc = nusc

        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.resolution = resolution

        self.grid_size_x = int((self.x_max - self.x_min) / resolution)
        self.grid_size_y = int((self.y_max - self.y_min) / resolution)

    # def generate(self, sample):

    
    #     grid = np.zeros((self.grid_size_y, self.grid_size_x))

    #     for ann_token in sample["anns"]:
    #         ann = self.nusc.get("sample_annotation", ann_token)

    #         if ann["num_lidar_pts"] == 0:
    #             continue

    #         x, y, _ = ann["translation"]

    #         if not (self.x_min <= x <= self.x_max and
    #                 self.y_min <= y <= self.y_max):
    #             continue

    #         ix = int((x - self.x_min) / self.resolution)
    #         iy = int((y - self.y_min) / self.resolution)

    #         if 0 <= ix < self.grid_size_x and 0 <= iy < self.grid_size_y:
    #             grid[iy, ix] = 1.0

    #     print("Occupied cells:", grid.sum())

    #     return grid
    def generate(self, sample):

        grid = np.zeros((self.grid_size_y, self.grid_size_x))

        cam_token = sample["data"]["CAM_FRONT"]
        sample_data = self.nusc.get("sample_data", cam_token)

        ego_pose_token = sample_data["ego_pose_token"]
        ego_pose = self.nusc.get("ego_pose", ego_pose_token)

        ego_translation = np.array(ego_pose["translation"])

        for ann_token in sample["anns"]:

            ann = self.nusc.get("sample_annotation", ann_token)

            if ann["num_lidar_pts"] == 0:
                continue

            global_xyz = np.array(ann["translation"])
            local_xyz = global_xyz - ego_translation

            x, y = local_xyz[0], local_xyz[1]

            if not (self.x_min <= x <= self.x_max and
                    self.y_min <= y <= self.y_max):
                continue

            ix = int((x - self.x_min) / self.resolution)
            iy = int((y - self.y_min) / self.resolution)

            radius = 2

            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):

                    nx = ix + dx
                    ny = iy + dy

                    if 0 <= nx < self.grid_size_x and 0 <= ny < self.grid_size_y:
                        grid[ny, nx] = 1.0

        return grid

