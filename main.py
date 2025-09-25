import os
import torch
from torch.utils.data import DataLoader
from dataset import CanyonDataset
from model import SimpleDepthNet
from train import train_model
from test import test_model

# Paths
canyons_path = r"C:\Users\nicol\ETH\Master_Thesis\canyons" # path to canyons data from FLSea VI

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

def main():

    print("\nStart programm...")

    print("\nLoad Dataset...")
    dataset = CanyonDataset(rgb_folder=flatiron_imgs_path, depth_folder=flatiron_depth_path)

    print("only using 500 images right now")
    dataset.rgb_files = dataset.rgb_files[:500]
    dataset.depth_files = dataset.depth_files[:500]
    
    print("\nTrain Dataset...")
    model, val_loader, device = train_model(dataset, num_epochs=2, batch_size=4)
    
    print("\nTest Dataset...")
    test_dataset = dataset
    test_dataset.rgb_files = test_dataset.rgb_files[-50:]
    test_dataset.depth_files = test_dataset.depth_files[-50:]
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    test_model(model, test_loader, device, num_samples = 3)

    print("\nEnd programm...")

if __name__ == "__main__":
    main()

