# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #
#                          Depth Prediction Main File
# --------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------- #

'''
TODO:       
            Pretrained model
            Sift features as Prior
            K-fold Crossvalidation
            Saveing Best Model -> whene val loss starts to increase - done
            Update requirements.txt

'''


import sys
import os

# reminder to start the virtual environment
if not (sys.prefix.endswith('Depth1_venv') or sys.prefix.endswith('Depth1_venv1') or sys.prefix.endswith('venv')):
    print("Warning: Not running inside the virtual environment")

# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
# .\venv\Scripts\Activate
# .\Depth1_venv\Scripts\Activate
# .\Depth1_venv1\Scripts\Activate
# venv (.\Depth1_venv1\Scripts\python.exe .\main.py)

import torch
from torch.utils.data import DataLoader
from dataset import CanyonDataset, CanyonDatasetWithPrior, CanyonDatasetWithPrior1
from dataset import combine_canyons
from train import train_model
from test import test_model
from model import *
from utils import *



# --------------------------------------------------------------------------------- #
#                                    Paths
# --------------------------------------------------------------------------------- #

# reminder to change the canyons folder path
print("\n(Reminder to change the canyons folder path)")

# canyons_path = r"C:\Users\nicol\ETH\Master_Thesis\canyons" # path to canyons data from FLSea VI
canyons_path = r"H:\canyons" # path to canyons data from FLSea VI on my SSD                                    <-------  Dataset Path !!!

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
#                              Main Function
# --------------------------------------------------------------------------------- #

def train_test_model(dataset_paths = canyon_seaErra_depth_paths,
                     model_class = UNet_3channels,
                     model_name = None,
                     n_list = [40],
                     num_epochs = 2,
                     batch_size = 4,
                     split = (0.7,0.2,0.1),
                     dataset_class = CanyonDataset,
                     test_num_samples = 10):
    """
    Train and test a model.

    Args:
        model_class: UNet_3channels or UNet_4channels
        model_name: for naming stuff (if none, a name is automaticaly given based on the arguments)
        dataset_class: CanyonDataset or CanyonDatasetWithPrior
        n_samples_of_each_canyon: zb [10,100,1000] -> will train multiple models on spezified dataset sizes
        test_num_samples: number of samples to visualize in test
    """
    model_named = False
    if model_name is not None:
        model_named = True

    num_canyons = len(dataset_paths)

    for n in n_list:

        # model name
        if model_named is False:
            model_name = f"{model_class.__name__}_{dataset_class.__name__}_{n}imgs"


        print_title(f"START (model name: {model_name})")
        n_per_canyon = n // num_canyons
        print(f"\nUsing total {n} images -> {n_per_canyon} per canyon")

        # Load and split dataset
        train_dataset, val_dataset, test_dataset = combine_canyons(
            paths = dataset_paths,
            dataset_class = dataset_class,
            n = n_per_canyon,
            split = split
        )

        train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
        val_loader   = DataLoader(val_dataset, batch_size = batch_size, shuffle = True)
        test_loader  = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

        # Results folder
        results_folder_path = create_results_folder(model_name = model_name)

        # Train
        model = model_class()
        model, device, train_losses, val_losses, train_time= train_model(
            train_loader,
            val_loader,
            model=model,
            num_epochs=num_epochs,
            results_folder_path = results_folder_path,
            model_name = model_name
            )

        # Test
        test_loss = test_model(
            model, test_loader, device, results_folder_path,
            num_samples = test_num_samples,
            model_name = model_name
        )

        # Plot Losses
        LossPlot(train_losses, val_losses, results_folder_path, model_name)

        # log
        log(model_name, train_time, test_loss)

    print_title(f"END ({model_name})")


# TODO: Simplify
def test_saved_model(model_class, model_path, test_dataset, device='cuda', batch_size=1,
                     num_samples=5, folder=None, model_name=None):
    """
    Test model
    """

    os.makedirs(folder, exist_ok=True)

    # Load model
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model_class().to(device)
    model.load_state_dict(torch.load(model_path, map_location = device))
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    test_mse = test_model(
        model = model,
        test_loader = test_loader,
        device = device,
        results_folder = folder,
        num_samples = num_samples,
        model_name = model_name
    )

    print(f"Finished testing {model_name}, Average masked MSE: {test_mse:.4f}")
    return test_mse


if __name__ == "__main__":

    # ------------------------------------------------
    #        select what you want to run
    # ------------------------------------------------

    model_class_list_3inChs = [
        UNet_3inChs_1L_12bc,
        UNet_3inChs_2L_12bc,
        UNet_3inChs_3L_12bc,

        UNet_3inChs_1L_24bf,
        UNet_3inChs_2L_24bf,
        UNet_3inChs_3L_24bf
        ]
    model_class_list_4inChs = [
        UNet_4inChs_1L_12bc,
        UNet_4inChs_2L_12bc,
        UNet_4inChs_3L_12bc,

        UNet_4inChs_1L_24bf,
        UNet_4inChs_2L_24bf,
        UNet_4inChs_3L_24bf
        ]
    model_class_list_4inChs_test = [
        UNet_4inChs_1L_12bc,
        UNet_4inChs_1L_24bf,
        UNet_4inChs_2L_12bc,
        UNet_4inChs_2L_24bf
        ]

    if True:
        for model_class in model_class_list_3inChs:
            count_parameters(model_class())
        print("\n")
        for model_class in model_class_list_4inChs:
            count_parameters(model_class())

    # ---------- Flexible U-Net TEST -----------------
    if True:

        for model_class in model_class_list_4inChs_test:
            train_test_model(
                dataset_paths = canyon_seaErra_depth_paths,
                model_class = model_class,
                model_name = None,
                n_list = [40],
                num_epochs = 3,
                batch_size = 4,
                split = (0.7,0.2,0.1),
                dataset_class = CanyonDatasetWithPrior1,
                test_num_samples = 5
                )


    # ---------- U-Net, 3 Channels -----------------
    if False:
        train_test_model(
            dataset_paths = canyon_seaErra_depth_paths,
            model_class = UNet_3channels,
            model_name = None,
            n_list = [10],
            num_epochs = 4,
            batch_size = 4,
            split = (0.7,0.2,0.1),
            dataset_class = CanyonDataset,
            test_num_samples = 10
            )

    # ---------- U-Net, 4 Channels -----------------
    if False:
        train_test_model(
            dataset_paths = canyon_seaErra_depth_paths,
            model_class = UNet_4channels,
            model_name = None,
            n_list = [10],
            num_epochs = 4,
            batch_size = 4,
            split = (0.7,0.2,0.1),
            dataset_class = CanyonDatasetWithPrior,
            test_num_samples = 10
            )

    # ---------- U-Net, 4 Channels 256 -------------
    if False:
        train_test_model(
            dataset_paths = canyon_seaErra_depth_paths,
            model_class = UNet_4channels_256,
            model_name = None,
            n_list = [5, 10, 100, 500, 1000],
            num_epochs = 10,
            batch_size = 4,
            split = (0.7,0.2,0.1),
            dataset_class = CanyonDatasetWithPrior1,
            test_num_samples = 10
            )
    
    # ---------- ResNet U-Net, 3 Channels ----------
    if False:
        train_test_model(
            dataset_paths = canyon_seaErra_depth_paths,
            model_class = ResNetUNet,
            model_name = None,
            n_list = [10],
            num_epochs = 4,
            batch_size = 4,
            split = (0.7,0.2,0.1),
            dataset_class = CanyonDataset,
            test_num_samples = 10
            )
    
    # ---------- Testing Saved Model ---------------
    if False:
        _, _, test_dataset = combine_canyons(
            paths = canyon_seaErra_depth_paths,
            dataset_class = CanyonDatasetWithPrior1,
            n = 100,
            split = (0.7,0.2,0.1)
        )
        test_saved_model(
            model_class = UNet_4channels,
            model_path = r"H:\Depth1\results\Results_UNet_4channels_10imgsPerCanyon_4_2025.10.05_23-39\UNet_4channels_10imgsPerCanyon_4.pth",
            test_dataset = test_dataset,
            device = 'cuda',
            num_samples = 5,
            folder=r"H:\Depth1\results\Results_UNet_4channels_10imgsPerCanyon_4_2025.10.05_23-39"
        )

    # ---------- Testing Random Stuff --------------
    if False:
        print_title("I should be centered")

        _, _, test_dataset = combine_canyons(
            paths = canyon_seaErra_depth_paths,
            dataset_class = CanyonDatasetWithPrior1,
            n = 10,
            split = (0.7,0.2,0.1)
        )
        dataset = test_dataset
        plot_rgb_depth_prior(dataset, idx=0)


    print_thingy() # ^-^