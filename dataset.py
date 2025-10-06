import os
import torch
from torch.utils.data import Dataset, Subset, ConcatDataset
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

        rgb_tensor = torch.tensor(rgb).permute(2,0,1).float()   # [H, W, C] -> [C,H,W]
        depth_tensor = torch.tensor(depth).unsqueeze(0)         # [H, W]    -> [1,H,W]

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
    
class CanyonDatasetWithPrior1(CanyonDataset):
    def __getitem__(self, idx):
        rgb, depth = super().__getitem__(idx)  # [3,H,W] and [1,H,W]
        _, H, W = depth.shape

        # Initialize prior with zeros
        prior = torch.zeros_like(depth)

        # Pick 100 random known points
        num_points = 100
        ys = torch.randint(0, H, (num_points,))
        xs = torch.randint(0, W, (num_points,))
        known_points = torch.stack([ys, xs], dim=1)
        known_depths = depth[0, ys, xs]

        # Assign each pixel the depth of the nearest known point (naive loop)
        for y in range(H):
            for x in range(W):
                # Compute distances to all known points
                dists = (ys - y)**2 + (xs - x)**2
                nearest_idx = torch.argmin(dists)
                prior[0, y, x] = known_depths[nearest_idx]

        # Concatenate
        rgb_with_prior = torch.cat([rgb, prior], dim=0)  # [4,H,W]
        return rgb_with_prior, depth, known_points



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




def combine_canyons(paths, dataset_class = CanyonDataset, n = 100, split=(0.7, 0.2, 0.1), seed=None):
    """
    Load multiple CanyonDatasets, sample n images from each, split into train/val/test, and combine.
    
    Args:
        paths (rgb_folder, depth_folder)
        n (int): Number of images to sample from each canyon
        split_fractions (tuple): Fractions for train, val, test split (must sum to 1.0)
    
    Returns:
        train_combined, val_combined, test_combined
    """

    train_sets = []
    val_sets = []
    test_sets = []

    for rgb_folder, depth_folder in paths:

        dataset = dataset_class(rgb_folder, depth_folder)

        # sample
        n = min(n, len(dataset))
        indices = torch.randperm(len(dataset))[:n].tolist()
        subset = Subset(dataset, indices)

        # Split
        train_subset, val_subset, test_subset = split_dataset(subset, *split)

        # append
        train_sets.append(train_subset)
        val_sets.append(val_subset)
        test_sets.append(test_subset)

    # Combine
    train_combined = ConcatDataset(train_sets)
    val_combined = ConcatDataset(val_sets)
    test_combined = ConcatDataset(test_sets)

    return train_combined, val_combined, test_combined