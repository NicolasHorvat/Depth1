import sys
import os

# reminder to start the virtual environment
if not sys.prefix.endswith('env1'):
    print("Warning: Not running inside 'env1' virtual environment")

# or start it like this
# venv (.\env1\Scripts\python.exe .\main.py)

import torch
from torch.utils.data import DataLoader
from dataset import CanyonDataset, CanyonDatasetWithPrior
from dataset import split_dataset
from train import train_model
from test import test_model
from model import DepthNet, DepthNetWithPrior

import matplotlib.pyplot as plt

# Paths
# canyons_path = r"C:\Users\nicol\ETH\Master_Thesis\canyons" # path to canyons data from FLSea VI
canyons_path = r"D:\canyons" # path to canyons data from FLSea VI on ssdmy SSD

# paths to the different folders
flatiron_path = os.path.join(canyons_path, "flatiron")
flatiron_depth_path = os.path.join(flatiron_path, "depth")
flatiron_imgs_path = os.path.join(flatiron_path, "imgs")
flatiron_seaErra_path = os.path.join(flatiron_path, "seaErra")

horse_canyon_path = os.path.join(canyons_path, "horse_canyon")
horse_canyon_depth_path = os.path.join(horse_canyon_path, "depth")
horse_canyon_imgs_path = os.path.join(horse_canyon_path, "imgs")
horse_canyon_seaErra_path = os.path.join(horse_canyon_path, "seaErra")

tiny_canyon_path = os.path.join(canyons_path, "tiny_canyon")
tiny_canyon_depth_path = os.path.join(tiny_canyon_path, "depth")
tiny_canyon_imgs_path = os.path.join(tiny_canyon_path, "imgs")
tiny_canyon_seaErra_path = os.path.join(tiny_canyon_path, "seaErra")

u_canyon_path = os.path.join(canyons_path, "u_canyon")
u_canyon_depth_path = os.path.join(u_canyon_path, "depth")
u_canyon_imgs_path = os.path.join(u_canyon_path, "imgs")
u_canyon_seaErra_path = os.path.join(u_canyon_path, "seaErra")

def main_DepthNet():

    print("\n----- START -----")

    print("\nLoading Data & Splitting Dataset...")
    full_dataset = CanyonDataset(rgb_folder=tiny_canyon_seaErra_path, depth_folder=tiny_canyon_depth_path)

    # for faster training/testing 
    num_imgs_list = [1000] # Array to train on diffenent sizes of the dataset

    all_losses = {}

    for num_imgs in num_imgs_list:
        print(f"\n(Using {num_imgs} images)")

        # randomly select a subset
        indices = torch.randperm(len(full_dataset))[:num_imgs].tolist() # selectrandom idices
        subset_rgb = [full_dataset.rgb_files[i] for i in indices]       # select coresponding rgb imgs 
        subset_depth = [full_dataset.depth_files[i] for i in indices]   # select coresponding depth imgs

        subset_dataset = CanyonDataset(rgb_folder=tiny_canyon_seaErra_path, depth_folder=tiny_canyon_depth_path)
        subset_dataset.rgb_files = subset_rgb
        subset_dataset.depth_files = subset_depth

        # split into training validation and test sets
        train_dataset, val_dataset, test_dataset = split_dataset(subset_dataset, 0.7, 0.2, 0.1)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # train
        print("\nTrain Dataset...")
        model, device, train_losses, val_losses = train_model(train_loader, val_loader,model = DepthNet(), num_epochs=5, name=f"{num_imgs}imgs")
        all_losses[num_imgs] = (train_losses, val_losses)
        
        # test
        print("\nTest Dataset...")
        test_model(model, test_loader, device, num_samples=3, name = f"{num_imgs}imgs")
                   
    # Plot Losses
    print("\nPlotting Loss Plot...")
    colors = ['b', 'g', 'r', 'k', 'c', 'm']
    plt.figure(figsize=(8,6))
    for i, (num_imgs, (train_losses, val_losses)) in enumerate(all_losses.items()):
        color = colors[i]
        plt.plot(train_losses, label=f"Train {num_imgs} imgs", linewidth=2, color = color)
        plt.plot(val_losses, '--', label=f"Val {num_imgs} imgs", linewidth=2, color = color)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.title("Training & Validation Loss Plot")
    plt.savefig("Train_Val_Loss_Plot.png")

    print("\n----- END -----")

def main_DepthNetWithPrior():

    print("\n----- START (WithPrior) -----")

    print("\nLoading Data & Splitting Dataset...")
    full_dataset = CanyonDatasetWithPrior(rgb_folder=tiny_canyon_seaErra_path, depth_folder=tiny_canyon_depth_path)

    # for faster training/testing
    num_imgs_list = [1000]

    all_losses = {}

    for num_imgs in num_imgs_list:
        print(f"\n(Using {num_imgs} images)")

        # randomly select a subset
        indices = torch.randperm(len(full_dataset))[:num_imgs].tolist()
        subset_rgb = [full_dataset.rgb_files[i] for i in indices]
        subset_depth = [full_dataset.depth_files[i] for i in indices]

        subset_dataset = CanyonDatasetWithPrior(rgb_folder=tiny_canyon_seaErra_path, depth_folder=tiny_canyon_depth_path)
        subset_dataset.rgb_files = subset_rgb
        subset_dataset.depth_files = subset_depth

        # split
        train_dataset, val_dataset, test_dataset = split_dataset(subset_dataset, 0.7, 0.2, 0.1)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # train
        print("\nTrain Dataset...")
        model, device, train_losses, val_losses = train_model(train_loader, val_loader,model=DepthNetWithPrior(), num_epochs=5, name=f"{num_imgs}imgs")
        all_losses[num_imgs] = (train_losses, val_losses)
        
        # test
        print("\nTest Dataset...")
        test_model(model, test_loader, device, num_samples=3, name = f"{num_imgs}imgs_prior")
                   
    # Plot Losses
    print("\nPlotting Loss Plot...")
    colors = ['b', 'g', 'r', 'k', 'c', 'm']
    plt.figure(figsize=(8,6))
    for i, (num_imgs, (train_losses, val_losses)) in enumerate(all_losses.items()):
        color = colors[i]
        plt.plot(train_losses, label=f"Train {num_imgs} imgs", linewidth=2, color = color)
        plt.plot(val_losses, '--', label=f"Val {num_imgs} imgs", linewidth=2, color = color)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.title("Training & Validation Loss Plot")
    plt.savefig("Train_Val_Loss_Plot.png")

    print("\n----- END (WithPrior) -----")

if __name__ == "__main__":
    if True:
        main_DepthNet()

    if True:
        main_DepthNetWithPrior()