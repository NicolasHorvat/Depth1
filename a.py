import torch
import tifffile
import os
import matplotlib.pyplot as plt
import numpy as np

# Check CUDA
print("CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))




datasets = {
    "flatiron": r"H:\canyons\flatiron",
    "horse_canyon": r"H:\canyons\horse_canyon",
    "tiny_canyon": r"H:\canyons\tiny_canyon",
    "u_canyon": r"H:\canyons\u_canyon"
}


for name, base_path in datasets.items():
    # Sorted depth files
    depth_files = sorted(os.listdir(os.path.join(base_path, "depth")))
    first_depth_file = depth_files[0]
    first_depth = os.path.join(base_path, "depth", first_depth_file)
    
    # Sorted RGB files
    rgb_files = sorted(os.listdir(os.path.join(base_path, "imgs")))
    first_rgb_file = rgb_files[0]
    first_rgb = os.path.join(base_path, "imgs", first_rgb_file)
    
    # Print filenames to verify correspondence
    print(f"{name} - Depth file: {first_depth_file}")
    print(f"{name} - RGB file:   {first_rgb_file}")
    
    # Read images
    depth = tifffile.imread(first_depth)
    rgb   = tifffile.imread(first_rgb)
    
    # Print shapes
    print(f"{name} depth shape:", depth.shape)
    print(f"{name} RGB shape:   ", rgb.shape)
    
    # Normalize RGB to 0-1 for plotting
    rgb_norm = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    
    # Plot side by side
    plt.figure(figsize=(12,5))
    
    # RGB
    plt.subplot(1, 2, 1)
    plt.title(f"{name} RGB")
    plt.imshow(rgb_norm)
    plt.axis('off')
    
    # Depth
    plt.subplot(1, 2, 2)
    plt.title(f"{name} Depth")
    im = plt.imshow(depth, cmap='viridis')
    plt.axis('off')
    plt.colorbar(im, fraction=0.046, pad=0.04)  # adjust size and spacing
    
    plt.show()
