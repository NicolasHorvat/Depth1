import os
import matplotlib.pyplot as plt
from datetime import datetime
import csv

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from dataset import split_dataset

# --------------------------------------------------------------------------------- #
#                            Some Utility Functions
# --------------------------------------------------------------------------------- #

def LossPlot(train_losses, val_losses, results_folder, model_name):
    plt.figure(figsize=(8,6))
    plt.plot(train_losses, label="Train", linewidth=2, color='b')
    plt.plot(val_losses, '--', label="Val", linewidth=2, color='r')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.title(f"Training & Validation Loss Plot ({model_name})")
    plt.savefig(os.path.join(results_folder, f"Train_Val_Loss_Plot_({model_name}).png"))
    plt.close()


def create_results_folder(model_name = "noName"):
    '''
    Creates a results folder for the plots,model ect
    Args: self explenatory
    Return: results_folder_path
    '''
    
    run_name = f"Results_{model_name}_{datetime.now().strftime('%Y.%m.%d_%H-%M')}"
    results_folder_path = os.path.join("results", run_name)
    os.makedirs(results_folder_path, exist_ok=True)
    print(f"\n-> Saving results to: {results_folder_path}")
    
    return results_folder_path


def save_model_at(model, folder_path, model_name = "noName"):
    '''
    saves the model in the specified folder path
    '''
    os.makedirs(folder_path, exist_ok=True)
    model_path = os.path.join(folder_path, f"{model_name}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\n-> Model saved at {model_path}")


def log(model_name, train_time, test_loss):
    """
    Logs Training information
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # CSV log
    csv_file = os.path.join(script_dir, "training_runs_log.csv")
    csv_exists = os.path.exists(csv_file)
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not csv_exists:
            writer.writerow(["DateTime", "ModelName", "TrainingTime_sec", "FinalMSE"])
        writer.writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
                         model_name, 
                         f"{train_time:.2f}", 
                         f"{test_loss:.6f}"])
    
    # TXT log
    txt_file = os.path.join(script_dir, "training_runs_log.txt")
    with open(txt_file, 'a') as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                f"Model: {model_name} | Training Time: {train_time:.2f}s or {train_time/60:.2f}min or {train_time/3600:.2f}h | Test Loss: {test_loss:.6f}\n")


def test_saved_model(model_class, model_path, dataset_class, rgb_folder, depth_folder, results_folder, num_samples=5, name="saved"):
    """
    Load a trained model and test it. (Plots some Plots)
    Runs test.py for a selected model
    """
    from test import test_model

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



def print_title(title):
    length = 100
    dashes = "/" * length
    spaces = " " * ((length - len(title)) // 2)
    print(f"\n{dashes}")
    print(f"\n{spaces}{title}")
    print(f"\n{dashes}")


def print_thingy():
    print("\n")
    print("          (\_(\        ")
    print("          (-O-O)       ")
    print("    \\\\\\\\_o(  >0)_////  ")
    print(f"\n")


import matplotlib.pyplot as plt

def plot_rgb_depth_prior(dataset, idx=0):
    # Get sample
    rgb_with_prior, depth, known_points = dataset[idx]  # ensure __getitem__ returns known_points
    
    # RGB image
    rgb = rgb_with_prior[:3].permute(1,2,0)  # [H,W,3]
    
    # Prior channel
    prior = rgb_with_prior[3]
    
    fig, axes = plt.subplots(1,3, figsize=(18,6))
    
    # RGB
    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    axes[0].axis('off')
    for (y, x) in known_points:
        axes[0].scatter(x, y, color='red', s=50, edgecolors='white')
    
    # Depth
    axes[1].imshow(depth.squeeze(), cmap='viridis')
    axes[1].set_title("Depth map")
    axes[1].axis('off')
    for (y, x) in known_points:
        axes[1].scatter(x, y, color='red', s=50, edgecolors='white')
    
    # Prior
    axes[2].imshow(prior, cmap='viridis')
    axes[2].set_title("Prior channel")
    axes[2].axis('off')
    for (y, x) in known_points:
        axes[2].scatter(x, y, color='red', s=50, edgecolors='white')
    
    plt.show()



def rmse_loss(pred, target, mask=None):
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    return torch.sqrt(nn.MSELoss(reduction = 'mean')(pred, target))

def silog_loss(pred, target, mask=None, eps=1e-6, lam=0.85):
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    pred = torch.clamp(pred, min=eps)
    target = torch.clamp(target, min=eps)
    
    d = torch.log(pred) - torch.log(target)
    return torch.sqrt(torch.mean(d**2) - lam * (torch.mean(d))**2)

def combined_loss(pred, target, mask=None):
    rmse = rmse_loss(pred, target, mask)
    silog = silog_loss(pred, target, mask)
    return 0.4 * rmse + 0.6 * silog




def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_name = type(model).__name__
    print(f"(Model: {model_name}) Total trainable parameters: {total_params:,}")


