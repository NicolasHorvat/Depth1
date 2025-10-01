# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
#                          Depth Prediction Main File
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #

import sys
import os

# reminder to start the virtual environment
if not sys.prefix.endswith('Depth1_venv1'):
    print("Warning: Not running inside 'Depth1_venv1' virtual environment")

# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
# .\Depth1_venv1\Scripts\Activate
# venv (.\Depth1_venv1\Scripts\python.exe .\main.py)

import torch
from torch.utils.data import DataLoader
from dataset import CanyonDataset, CanyonDatasetWithPrior
from dataset import split_dataset, combine_canyons
from train import train_model
from test import test_model
from model import DepthNet, DepthNetWithPrior
from utils import LossPlot, print_title

import matplotlib.pyplot as plt
from datetime import datetime


# --------------------------------------------------------------------------------- #
#                                    Paths
# --------------------------------------------------------------------------------- #

# reminder to change the canyons folder path
print("\n(Reminder to change the canyons folder path)")

# canyons_path = r"C:\Users\nicol\ETH\Master_Thesis\canyons" # path to canyons data from FLSea VI
canyons_path = r"D:\canyons" # path to canyons data from FLSea VI on my SSD

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


canyon_imgs_depth_paths = [
    (flatiron_imgs_path, flatiron_depth_path),
    (horse_canyon_imgs_path, horse_canyon_depth_path),
    (tiny_canyon_imgs_path, tiny_canyon_depth_path),
    (u_canyon_imgs_path, u_canyon_depth_path)
]

canyon_seaErra_depth_paths = [
    (flatiron_seaErra_path, flatiron_depth_path),
    (horse_canyon_seaErra_path, horse_canyon_depth_path),
    (tiny_canyon_seaErra_path, tiny_canyon_depth_path),
    (u_canyon_seaErra_path, u_canyon_depth_path)
]



# --------------------------------------------------------------------------------- #
#                               Results Folder
# --------------------------------------------------------------------------------- #

# TODO: Could be done better (works for now)

# create a folder to save plots for organization
run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
results_folder = os.path.join("results", run_name)
os.makedirs(results_folder, exist_ok=True)
print(f"\nSaving results to: {results_folder}")



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
    Main function for training without any prior knowlage.
    Load Data, train, test
    '''

    print_title("START (without Prior)")

    # for faster training/testing etc
    n_samples_of_eache_canyon = [10] # Array to train on diffenent sizes of the dataset

    all_losses = {}

    for n in n_samples_of_eache_canyon:
        print(f"\n(Using {n} samples of each canyon dataset (FLSea))")

        # Split fractions (train,val,test)
        split = (0.7, 0.2, 0.1) 

        # Combine and split
        train_dataset, val_dataset, test_dataset = combine_canyons(
            paths = canyon_seaErra_depth_paths,
            dataset_class = CanyonDataset,
            n = n,
            split = split
        )

        # DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader   = DataLoader(val_dataset, batch_size=8, shuffle=True)
        test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # train
        print("\nTrain Dataset...")
        model, device, train_losses, val_losses = train_model(
            train_loader, val_loader, results_folder,
            model = DepthNet(),
            num_epochs=2,
            name=f"{n}imgs_each"
            )
        
        all_losses[n] = (train_losses, val_losses)
        
        # test
        print("\n\nTest Dataset...")
        test_model(
            model, test_loader, device, results_folder,
            num_samples=10,
            name = f"{n}imgs each"
            )

                   
    # Plot Losses
    LossPlot(all_losses, results_folder,
             "Training & Validation Loss Plot (WithoutPrior)", # title
             "Train_Val_Loss_Plot.png"  # file name
             )
        

    print_title("Finished! (without Prior)")

def main_DepthNetWithPrior():

    print_title("START (with Prior)")

    print("\nLoading Data & Splitting Dataset...")
    # for faster training/testing etc
    n_samples_of_eache_canyon = [10] # Array to train on diffenent sizes of the dataset

    all_losses = {}

    for n in n_samples_of_eache_canyon:
        print(f"\n(Using {n} samples of each canyon)")

        # Split fractions (train,val,test)
        split = (0.7, 0.2, 0.1) 

        # Combine and split
        train_dataset, val_dataset, test_dataset = combine_canyons(
            paths = canyon_seaErra_depth_paths,
            dataset_class = CanyonDatasetWithPrior,
            n = n,
            split = split
        )

        # DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader   = DataLoader(val_dataset, batch_size=8, shuffle=True)
        test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # train
        print("\nTrain Dataset...")
        model, device, train_losses, val_losses = train_model(
            train_loader, val_loader, results_folder,
            model = DepthNetWithPrior(),
            num_epochs=2,
            name=f"{n}imgs_each"
            )
        
        all_losses[n] = (train_losses, val_losses)
        
        # test
        print("\nTest Dataset...")
        test_model(
            model, test_loader, device, results_folder,
            num_samples=10,
            name = f"{n}imgs each"
            )

                   
    # Plot Losses
    LossPlot(all_losses, results_folder,
             "Training & Validation Loss Plot (WithoutPrior)", # title
             "Train_Val_Loss_Plot.png"  # file name
             )
        
    print_title("Finished (with Prior)")

if __name__ == "__main__":

    # select what you want to run

    if True:
        main_DepthNet()

    if True:
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
    

    # for testing stuff
    if False:
        print_title("I should be centered")