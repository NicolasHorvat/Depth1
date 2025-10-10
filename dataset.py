import os
import cv2
import torch
from torch.utils.data import Dataset, Subset, ConcatDataset
import tifffile
import numpy as np
import pickle


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
    


class CanyonDatasetWithSavedPriors(Dataset):
    def __init__(self, rgb_folder, depth_folder, priors_folder):
        """
        Dataset that loads RGB, depth, and precomputed NN/Gaussian priors.

        Args:
            rgb_folder (str): Path to RGB images.
            depth_folder (str): Path to depth maps.
            priors_root (str): Path to priors folder
        """
        self.rgb_folder = rgb_folder
        self.depth_folder = depth_folder
        self.rgb_files = sorted(os.listdir(rgb_folder))
        self.depth_files = sorted(os.listdir(depth_folder))

        canyon_name = os.path.basename(os.path.dirname(rgb_folder))
        folder_name = os.path.basename(rgb_folder)

        # Paths to saved priors
        self.nn_priors_path = os.path.join(priors_folder, f"{canyon_name}_{folder_name}_nn_priors")
        self.gauss_priors_path = os.path.join(priors_folder, f"{canyon_name}_{folder_name}_gauss_priors")

        self.nn_files = sorted(os.listdir(self.nn_priors_path))
        self.gauss_files = sorted(os.listdir(self.gauss_priors_path))

        if not (len(self.rgb_files) == len(self.depth_files) == len(self.nn_files) == len(self.gauss_files)):
            raise ValueError("Mismatch in number of RGB, depth, or prior files.")

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        # File paths
        rgb_path = os.path.join(self.rgb_folder, self.rgb_files[idx])
        depth_path = os.path.join(self.depth_folder, self.depth_files[idx])
        nn_path = os.path.join(self.nn_priors_path, self.nn_files[idx])
        gauss_path = os.path.join(self.gauss_priors_path, self.gauss_files[idx])

        # Load files
        rgb = tifffile.imread(rgb_path) / 255.0
        depth = tifffile.imread(depth_path).astype('float32')
        nn_prior = np.load(nn_path).astype('float32')
        gauss_prior = np.load(gauss_path).astype('float32')

        # Convert to tensors
        rgb_tensor = torch.tensor(rgb).permute(2, 0, 1).float()       # [C, H, W]
        depth_tensor = torch.tensor(depth).unsqueeze(0).float()       # [1, H, W]
        nn_tensor = torch.tensor(nn_prior).unsqueeze(0).float()       # [1, H, W]
        gauss_tensor = torch.tensor(gauss_prior).unsqueeze(0).float() # [1, H, W]

        return rgb_tensor, depth_tensor, nn_tensor, gauss_tensor



# old
class CanyonDatasetSiftPriors(CanyonDataset):
    def __getitem__(self, idx):
        rgb, depth = super().__getitem__(idx)  # [3,H,W], [1,H,W]
        _, H, W = depth.shape

        # Convert RGB to numpy HxWx3 for OpenCV
        rgb_np = (rgb.permute(1,2,0).numpy() * 255).astype('uint8')
        gray = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY)

        # 16-quadrants
        rows = 4
        cols = 4
        n_keypoints_per_patch = 10
        keypoints_all = []

        row_height = H // rows
        col_width = W // cols

        sift = cv2.SIFT_create()

        for i in range(rows):
            for j in range(cols):
                y0 = i * row_height
                y1 = H if i == rows-1 else (i+1) * row_height
                x0 = j * col_width
                x1 = W if j == cols-1 else (j+1) * col_width

                patch_gray = gray[y0:y1, x0:x1]
                kps = sift.detect(patch_gray, None)
                kps = sorted(kps, key=lambda kp: kp.response, reverse=True)[:n_keypoints_per_patch]

                for kp in kps:
                    # Adjust coordinates to full image
                    kp_x = int(kp.pt[0]) + x0
                    kp_y = int(kp.pt[1]) + y0
                    keypoints_all.append((kp_y, kp_x))

        if len(keypoints_all) == 0:
            # fallback: center pixel
            keypoints_all = [(H//2, W//2)]

        ys = torch.tensor([kp[0] for kp in keypoints_all])
        xs = torch.tensor([kp[1] for kp in keypoints_all])

        known_points = torch.stack([ys, xs], dim=1)
        known_depths = depth[0, ys, xs]

        # Initialize prior
        prior = torch.zeros_like(depth)

        # Grid of coordinates
        yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        yy = yy.unsqueeze(0)  # [1,H,W]
        xx = xx.unsqueeze(0)  # [1,H,W]

        # Compute squared distances to keypoints
        dists = (yy - ys[:, None, None])**2 + (xx - xs[:, None, None])**2  # [num_points,H,W]

        nearest_idx = torch.argmin(dists, dim=0)
        prior[0] = known_depths[nearest_idx]

        # Concatenate prior to RGB
        rgb_with_prior = torch.cat([rgb, prior], dim=0)  # [4,H,W]

        return rgb_with_prior, depth#, known_points

# old  
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

# old
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
        known_depths = depth[0, ys, xs]  # [num_points]

        # Generate a grid of all pixel coordinates
        yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        yy = yy.unsqueeze(0)  # [1,H,W]
        xx = xx.unsqueeze(0)  # [1,H,W]

        # Compute squared distances from all pixels to all known points
        # (num_points, H, W)
        dists = (yy - ys[:, None, None])**2 + (xx - xs[:, None, None])**2

        # Find the nearest known point for each pixel
        nearest_idx = torch.argmin(dists, dim=0)  # [H, W]

        # Assign depth from nearest known point
        prior[0] = known_depths[nearest_idx]  # broadcasting

        # Concatenate prior
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