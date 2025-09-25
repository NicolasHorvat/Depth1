import os
import torch
from torch.utils.data import Dataset
import tifffile

class CanyonDataset(Dataset):
    def __init__(self, rgb_folder, depth_folder):
        self.rgb_folder = rgb_folder
        self.depth_folder = depth_folder
        self.rgb_files = sorted(os.listdir(rgb_folder))
        self.depth_files = sorted(os.listdir(depth_folder))

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        rgb_path = os.path.join(self.rgb_folder, self.rgb_files[idx])
        depth_path = os.path.join(self.depth_folder, self.depth_files[idx])

        rgb = tifffile.imread(rgb_path) / 255.0
        depth = tifffile.imread(depth_path).astype('float32')

        rgb_tensor = torch.tensor(rgb).permute(2,0,1).float()  # [C,H,W]
        depth_tensor = torch.tensor(depth).unsqueeze(0)         # [1,H,W]

        return rgb_tensor, depth_tensor
