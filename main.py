import sys
import os

# reminder to start the virtual environment
if not sys.prefix.endswith('Depth1_venv'):
    print("Warning: Not running inside 'Depth1_venv' virtual environment")

# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
# .\Depth1_venv\Scripts\Activate
# venv (.\env1\Scripts\python.exe .\main.py)

import torch
from torch.utils.data import DataLoader
from dataset import CanyonDataset, CanyonDatasetWithPrior
from dataset import split_dataset
from train import train_model
from test import test_model

import matplotlib.pyplot as plt
from datetime import datetime

from model import DepthNet, DepthNetWithPrior




# --------------------------------------------------------------------------------- #
#                                    Paths
# --------------------------------------------------------------------------------- #

# reminder to change the canyons folder path
print("\n(Reminder to change the canyons folder path)")
# canyons_path = r"C:\Users\nicol\ETH\Master_Thesis\canyons" # path to canyons data from FLSea VI
canyons_path = r"H:\canyons" # path to canyons data from FLSea VI on my SSD

# paths to the different folders in canyon
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



# --------------------------------------------------------------------------------- #
#                               Results Folder
# --------------------------------------------------------------------------------- #
# TODO: Could be done better (works for now)
# create a folder to save plots for organization
run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
results_folder = os.path.join("results", run_name)
os.makedirs(results_folder, exist_ok=True)
print(f"Saving results to: {results_folder}")



# --------------------------------------------------------------------------------- #
#                                 Functions
# --------------------------------------------------------------------------------- #

def test_saved_model(model_class, model_path, dataset_class, rgb_folder, depth_folder, results_folder, num_samples=5, name="saved"):
    """
    Load a trained model and test it. (Plots some Plots)
    Runs test.py for a selected model
    """

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Init model
    model = model_class().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded model weights from {model_path}")

    # test dataset
    dataset = dataset_class(rgb_folder=rgb_folder, depth_folder=depth_folder)
    dataset.rgb_files = sorted(dataset.rgb_files)
    dataset.depth_files = sorted(dataset.depth_files)

    _, _, test_dataset = split_dataset(dataset, 0.7, 0.2, 0.1)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Run test
    print("\nTesting loaded model...")
    test_model(model, test_loader, device, results_folder, num_samples=num_samples, name=name)



def main_DepthNet():
    '''
    Main function for training without any prior knowlage
    '''

    print("\n----- START (without Prior) -----")

    print("\nLoading Data & Splitting Dataset...")
    # select data to be used (and sort):
    dataset = CanyonDataset(rgb_folder=tiny_canyon_seaErra_path, depth_folder=tiny_canyon_depth_path)
    dataset.rgb_files = sorted(dataset.rgb_files)
    dataset.depth_files = sorted(dataset.depth_files)

    # for faster training/testing 
    num_imgs_list = [1000] # Array to train on diffenent sizes of the dataset

    all_losses = {}

    for num_imgs in num_imgs_list:
        print(f"\n(Using {num_imgs} images)")

        # randomly select a subset
        indices = torch.randperm(len(dataset))[:num_imgs].tolist() # selectrandom idices
        subset_rgb = [dataset.rgb_files[i] for i in indices]       # select coresponding rgb imgs 
        subset_depth = [dataset.depth_files[i] for i in indices]   # select coresponding depth imgs

        dataset.rgb_files = subset_rgb
        dataset.depth_files = subset_depth

        # split into training validation and test sets
        train_dataset, val_dataset, test_dataset = split_dataset(dataset, 0.7, 0.2, 0.1)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # train
        print("\nTrain Dataset...")
        model, device, train_losses, val_losses = train_model(train_loader, val_loader, results_folder, model = DepthNet(), num_epochs=3, name=f"{num_imgs}imgs")
        all_losses[num_imgs] = (train_losses, val_losses)
        
        # test
        print("\nTest Dataset...")
        test_model(model, test_loader, device, results_folder, num_samples=3, name = f"{num_imgs}imgs")

                   
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
    plt.savefig(os.path.join(results_folder, "Train_Val_Loss_Plot.png"))

    print("\n----- END (Without Prior)-----")

def main_DepthNetWithPrior():

    print("\n----- START (With Prior) -----")

    print("\nLoading Data & Splitting Dataset...")
    dataset = CanyonDatasetWithPrior(rgb_folder=tiny_canyon_seaErra_path, depth_folder=tiny_canyon_depth_path)
    dataset.rgb_files = sorted(dataset.rgb_files)
    dataset.depth_files = sorted(dataset.depth_files)

    # for faster training/testing
    num_imgs_list = [1000]

    all_losses = {}

    for num_imgs in num_imgs_list:
        print(f"\n(Using {num_imgs} images)")

        # randomly select a subset
        indices = torch.randperm(len(dataset))[:num_imgs].tolist()
        subset_rgb = [dataset.rgb_files[i] for i in indices]
        subset_depth = [dataset.depth_files[i] for i in indices]

        dataset.rgb_files = subset_rgb
        dataset.depth_files = subset_depth

        # split
        train_dataset, val_dataset, test_dataset = split_dataset(subset_dataset, 0.7, 0.2, 0.1)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # train
        print("\nTrain Dataset...")
        model, device, train_losses, val_losses = train_model(train_loader, val_loader, results_folder, model=DepthNetWithPrior(), num_epochs=3, name=f"{num_imgs}imgs_WithPrior")
        all_losses[num_imgs] = (train_losses, val_losses)
        
        # test
        print("\nTest Dataset...")
        test_model(model, test_loader, device, results_folder, num_samples=3, name = f"{num_imgs}imgs_prior")
                   
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
    plt.title("Training & Validation Loss Plot (WithPrior)")
    plt.savefig(os.path.join(results_folder, "Train_Val_Loss_Plot_WithPrior.png"))

    print("\n----- END (WithPrior) -----")

if __name__ == "__main__":

    # select what you want to run

    if True:
        main_DepthNet()

    if False:
        main_DepthNetWithPrior()

    if False:
        test_saved_model(
        model_class=DepthNet,
        model_path=os.path.join("results", "2025-09-29_17-00-51", "depth_model_100imgs.pth"),
        dataset_class=CanyonDataset,
        rgb_folder=tiny_canyon_seaErra_path,
        depth_folder=tiny_canyon_depth_path,
        results_folder=os.path.join("results", "2025-09-29_17-00-51"),
        num_samples=5,
        name="DepthNet_test"
    )