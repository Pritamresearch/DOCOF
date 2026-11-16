import os
import torch
from torch.utils.data import Dataset


class DOCOFTorchDatasetV2(Dataset):

    def __init__(self, root, split="train"):

        self.root = os.path.join(root, split)
        self.files = sorted(os.listdir(self.root))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        data = torch.load(
            os.path.join(self.root, self.files[idx])
        )

        return data