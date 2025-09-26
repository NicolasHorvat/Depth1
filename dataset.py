import os
import torch
from torch.utils.data import Dataset, Subset
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
    
class CanyonDatasetWithPrior(CanyonDataset):
    def __getitem__(self, idx):
        # Get original RGB and depth
        rgb, depth = super().__getitem__(idx)
        
        _, H, W = depth.shape
        # Take the middle line
        middle_line = depth[:, H//2, :]  # [1, W]
        # Expand to full height to match image dimensions
        prior = middle_line.unsqueeze(1).expand(-1, H, -1)  # [1, H, W]

        # Concatenate RGB and prior along channel dimension
        rgb_with_prior = torch.cat([rgb, prior], dim=0)  # [4, H, W]

        return rgb_with_prior, depth



def split_dataset(dataset, train_frac=0.7, val_frac=0.2, test_frac=0.1):
    """
    Split the Dataset into train, validation, and test sets.
    """
    total_len = len(dataset)
    train_len = int(train_frac * total_len)
    val_len = int(val_frac * total_len)
    test_len = total_len - train_len - val_len

    indices = torch.randperm(total_len).tolist()  # shuffle indices

    train_indices = indices[:train_len]
    val_indices = indices[train_len:train_len+val_len]
    test_indices = indices[train_len+val_len:]

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    test_subset = Subset(dataset, test_indices)

    return train_subset, val_subset, test_subset